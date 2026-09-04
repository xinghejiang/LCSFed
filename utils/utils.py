import torch
import random
import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp

_NOVELTY_P_MIN = 1e-12

EVAL_METRIC_NAMES = ('recall', 'precision', 'ndcg', 'coverage', 'novelty')
_METRIC_DISPLAY = {
    'recall': 'Recall',
    'precision': 'Precision',
    'ndcg': 'NDCG',
    'coverage': 'Coverage',
    'novelty': 'Novelty',
}


def parse_eval_metrics(metrics_str):
    if metrics_str is None:
        return list(EVAL_METRIC_NAMES)
    s = metrics_str.strip().lower()
    if s in ('', 'all', '*'):
        return list(EVAL_METRIC_NAMES)
    names = [m.strip().lower() for m in metrics_str.split(',') if m.strip()]
    invalid = sorted(set(names) - set(EVAL_METRIC_NAMES))
    if invalid:
        raise ValueError(
            'Unknown eval_metrics: {}. Choose from: {}'.format(
                ', '.join(invalid), ', '.join(EVAL_METRIC_NAMES)))
    if not names:
        raise ValueError('eval_metrics must list at least one metric')
    seen = set()
    ordered = []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def format_metric_line(metric_name, values, top_k):
    label = _METRIC_DISPLAY.get(metric_name, metric_name)
    return ', '.join(
        '{}@{} = {:.6f}'.format(label, top_k[i], values[i])
        for i in range(len(top_k))
    )


def early_stop_score(metrics_dict, metric_name):
    if metric_name not in metrics_dict:
        raise KeyError(
            'early_stop_metric {!r} is not in eval results; '
            'include it in --eval_metrics'.format(metric_name))
    return float(sum(metrics_dict[metric_name]))

def saveCheckpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resumeCheckpoint(model, model_dir, map_location='cpu'):
    state_dict = torch.load(model_dir, map_location=map_location)
    model.load_state_dict(state_dict)


def use_cuda(enable, device_id=0):
    if enable:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


def initLogging(logFilename):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s-%(levelname)s-%(message)s',
        datefmt='%y-%m-%d %H:%M',
        filename=logFilename,
        filemode='w',
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def setSeed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ldp_perturb_laplace(t: torch.Tensor, laplace_lambda: float) -> torch.Tensor:
    if laplace_lambda is None:
        raise ValueError('laplace_lambda must be set')
    if laplace_lambda < 0:
        raise ValueError('laplace_lambda must be >= 0')

    x = t.detach()
    b = float(laplace_lambda)
    if b == 0.0:
        return x

    noise = torch.distributions.Laplace(
        loc=torch.zeros((), device=x.device, dtype=x.dtype),
        scale=torch.tensor(b, device=x.device, dtype=x.dtype),
    ).sample(x.shape)
    return x + noise


def compute_metrics(all_results, top_k):
    recall_list = []
    precision_list = []
    ndcg_list = []

    for k in top_k:
        k_recall_list = []
        k_precision_list = []
        k_ndcg_list = []

        for targets, preds in all_results:
            targets = np.array(targets)
            preds = np.array(preds)
            pred_sort_idx = np.argsort(preds)[::-1]
            top_k_idx = pred_sort_idx[:k]

            total_relevant = np.sum(targets)
            relevant_in_top_k = np.sum(targets[top_k_idx])

            recall = relevant_in_top_k / total_relevant if total_relevant > 0 else 0.0
            k_recall_list.append(recall)

            precision = relevant_in_top_k / k
            k_precision_list.append(precision)

            dcg = 0.0
            ideal_sort_idx = np.argsort(targets)[::-1]
            ideal_targets = targets[ideal_sort_idx]

            for i, idx in enumerate(top_k_idx):
                rel = targets[idx]
                dcg += rel / np.log2(i + 2)

            idcg = 0.0
            for i in range(min(k, len(ideal_targets))):
                rel = ideal_targets[i]
                idcg += rel / np.log2(i + 2)

            ndcg = dcg / idcg if idcg > 0.0 else 0.0
            k_ndcg_list.append(ndcg)

        recall_list.append(np.mean(k_recall_list))
        precision_list.append(np.mean(k_precision_list))
        ndcg_list.append(np.mean(k_ndcg_list))

    return recall_list, precision_list, ndcg_list


def compute_diversity_metrics(all_results_with_iids, top_k, popularity, num_eval_items):
    popularity = np.asarray(popularity, dtype=np.float64)
    coverage_list = []
    novelty_list = []

    for k in top_k:
        union_set = set()
        k_nov_list = []

        for _targets, preds, iids in all_results_with_iids:
            preds = np.asarray(preds)
            iids = np.asarray(iids, dtype=np.int64)

            sort_idx = np.argsort(preds)[::-1]
            topk_idx = sort_idx[:k]
            topk_iids = iids[topk_idx]

            union_set.update(topk_iids.tolist())

            p = np.maximum(popularity[topk_iids], _NOVELTY_P_MIN)
            nov_u = float(np.mean(-np.log2(p)))
            k_nov_list.append(nov_u)

        coverage = len(union_set) / num_eval_items if num_eval_items > 0 else 0.0
        novelty = float(np.mean(k_nov_list)) if k_nov_list else 0.0

        coverage_list.append(coverage)
        novelty_list.append(novelty)

    return coverage_list, novelty_list


def compute_eval_metrics(all_results_with_iids, top_k, metrics,
                         popularity=None, num_eval_items=None):
    metrics = list(metrics)
    need_rank = bool(set(metrics) & {'recall', 'precision', 'ndcg'})
    need_div = bool(set(metrics) & {'coverage', 'novelty'})

    out = {}
    if need_rank:
        rank_results = [(t, p) for t, p, _ in all_results_with_iids]
        recall, precision, ndcg = compute_metrics(rank_results, top_k)
        if 'recall' in metrics:
            out['recall'] = recall
        if 'precision' in metrics:
            out['precision'] = precision
        if 'ndcg' in metrics:
            out['ndcg'] = ndcg

    if need_div:
        if popularity is None or num_eval_items is None:
            raise ValueError(
                'coverage/novelty require popularity and num_eval_items')
        coverage, novelty = compute_diversity_metrics(
            all_results_with_iids,
            top_k,
            popularity=popularity,
            num_eval_items=num_eval_items,
        )
        if 'coverage' in metrics:
            out['coverage'] = coverage
        if 'novelty' in metrics:
            out['novelty'] = novelty

    return out


def log_eval_metrics(split_tag, metrics_dict, top_k, metric_order=None, logger=None):
    log = logger.info if logger is not None else logging.info
    order = metric_order if metric_order is not None else list(metrics_dict.keys())
    for name in order:
        if name not in metrics_dict:
            continue
        log('[%s] %s', split_tag, format_metric_line(name, metrics_dict[name], top_k))

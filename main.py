import os
import sys
import argparse
import datetime
import logging

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.engine import MLPEngine
from utils.utils import (
    EVAL_METRIC_NAMES,
    _METRIC_DISPLAY,
    early_stop_score,
    initLogging,
    log_eval_metrics,
    parse_eval_metrics,
    setSeed,
)
from utils.data_loader import MovieLensDataLoader, negative_sampling

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def _load_llm_npy(args, subdir):
    path = os.path.join(args.data_dir, args.dataset_name, subdir, f'llm_{args.content_dim}.npy')
    if subdir == 'uid':
        args.user_llm_path = path
    else:
        args.item_llm_path = path
    arr = np.load(path)
    return torch.tensor(arr, dtype=torch.float32).to(args.device)


def load_user_llm_embeddings(args):
    return _load_llm_npy(args, 'uid')


def load_item_llm_embeddings(args):
    return _load_llm_npy(args, 'iid')


def resolve_device(args):
    if args.device == 'cpu':
        return torch.device('cpu')
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('device=cuda but CUDA is not available')
        return torch.device('cuda:' + str(args.gpu_id))
    if torch.cuda.is_available():
        return torch.device('cuda:' + str(args.gpu_id))
    return torch.device('cpu')


def prepare():
    parser = argparse.ArgumentParser(description='LCSFed (Dual Cold-Start, Online Server Align)')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--clients_sample_ratio', type=float, default=1.0)

    parser.add_argument('--num_round', type=int, default=200)
    parser.add_argument('--local_epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr_eta', type=int, default=10)
    parser.add_argument('--lr_client', type=float, default=0.001)
    parser.add_argument('--lr_server', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--num_negative', type=int, default=5)
    parser.add_argument('--random_seed', type=int, default=0)

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--top_k', type=str, default='10,20,50')
    parser.add_argument('--dataset_name', type=str, default='ML_1M')
    parser.add_argument('--content_dim', type=int, default=1024)
    parser.add_argument('--latent_dim', type=int, default=200)
    parser.add_argument('--align_gu', type=float, default=0.1)
    parser.add_argument('--align_gi', type=float, default=0.2)
    parser.add_argument('--emb_type', type=str, default='llm')

    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=1)

    parser.add_argument('--eval_metrics',type=str,default='ndcg')
    parser.add_argument('--early_stop_metric',type=str, default='ndcg',)

    parser.add_argument('--ldp_enable', action='store_true',
                        help='Add Laplace noise to client-uploaded params before FedAvg')
    parser.add_argument('--ldp_laplace_lambda', type=float, default=0.0,
                        help='Noise strength lambda for Laplace(0, lambda). 0 means no noise.')

    return parser.parse_args()


DATASET_SIZES = {
    'ML_1M': (6039, 3550),
    'ML_1M_3000': (3000, 3605),
    'LastFM_1K': (988, 9136),
    'ML_100K': (943, 1682),
    'CiteULike': (5551, 16980),
    'HetRec2011': (2113, 6829),
}


if __name__ == '__main__':
    args = prepare()

    if args.dataset_name not in DATASET_SIZES:
        raise ValueError(f'Dataset {args.dataset_name} not supported')
    args.num_users, args.num_items = DATASET_SIZES[args.dataset_name]

    setSeed(args.random_seed)

    os.makedirs('logs', exist_ok=True)
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    log_file = os.path.join(
        'logs',
        '[{}]-[{}.{}.{}]-[{}]'.format(
            args.dataset_name,
            args.latent_dim,
            args.emb_type,
            args.content_dim,
            current_time,
        ),
    )
    initLogging(log_file)
    logging.info(args)

    args.device = resolve_device(args)
    logging.info('Device: %s', args.device)

    args.top_k = [int(x.strip()) for x in args.top_k.split(',') if x.strip()]
    args.eval_metrics = parse_eval_metrics(args.eval_metrics)
    if args.early_stop_metric not in args.eval_metrics:
        raise ValueError(
            'early_stop_metric={!r} must be included in eval_metrics={}'.format(
                args.early_stop_metric, args.eval_metrics,
            ),
        )

    data_path = os.path.join(args.data_dir, args.dataset_name)
    sample_generator = MovieLensDataLoader(
        data_path,
        args.num_negative,
        args.batch_size,
        args.emb_type,
        args.content_dim,
        args.num_users,
        args.num_items,
    )

    val_data = sample_generator.get_vali_data
    test_data = sample_generator.get_test_data
    train_data = sample_generator.get_train_data

    need_diversity = bool(set(args.eval_metrics) & {'coverage', 'novelty'})
    val_popularity = sample_generator.get_vali_popularity if need_diversity else None
    val_num_items = sample_generator.get_vali_num_items if need_diversity else None
    test_popularity = sample_generator.get_test_popularity if need_diversity else None
    test_num_items = sample_generator.get_test_num_items if need_diversity else None
    if need_diversity:
        logging.info('Eval candidate set: |I_vali|=%d, |I_test|=%d', val_num_items, test_num_items)

    user_llm_emb = load_user_llm_embeddings(args)
    item_llm_emb = load_item_llm_embeddings(args)
    if user_llm_emb.shape[0] < args.num_users:
        raise ValueError(
            'user LLM emb rows {} < num_users {}; check uid/llm_{}.npy'.format(
                user_llm_emb.shape[0], args.num_users, args.content_dim,
            ),
        )
    if item_llm_emb.shape[0] < args.num_items:
        raise ValueError(
            'item LLM emb rows {} < num_items {}; check iid/llm_{}.npy vs data'.format(
                item_llm_emb.shape[0], args.num_items, args.content_dim,
            ),
        )

    engine = MLPEngine(args, user_llm_emb, item_llm_emb)

    best_metric = 0.0
    best_round = 0
    patience_counter = 0

    logging.info('=' * 80)
    logging.info(
        'Training with early stopping: patience={}, eval_interval={}'.format(
            args.patience, args.eval_interval,
        ),
    )
    logging.info('Eval metrics: %s', ', '.join(args.eval_metrics))
    logging.info(
        'Early-stop on sum(val %s@%s)',
        _METRIC_DISPLAY[args.early_stop_metric],
        ','.join(map(str, args.top_k)),
    )
    logging.info('=' * 80)

    for round_idx in range(args.num_round):
        logging.info('-' * 80)
        logging.info('Round {} starts !'.format(round_idx))

        all_train_data = negative_sampling(train_data, args.num_negative, args.batch_size)
        logging.info('Training phase!')
        engine.fed_train_a_round(all_train_data, round_idx)

        if round_idx % args.eval_interval == 0:
            logging.info('Validation phase!')
            val_metrics = engine.fed_evaluate(
                val_data, popularity=val_popularity, num_eval_items=val_num_items,
            )
            log_eval_metrics('VAL', val_metrics, args.top_k, args.eval_metrics)

            logging.info('Test phase!')
            test_metrics = engine.fed_evaluate(
                test_data, popularity=test_popularity, num_eval_items=test_num_items,
            )
            log_eval_metrics('TEST', test_metrics, args.top_k, args.eval_metrics)

            current_metric = early_stop_score(val_metrics, args.early_stop_metric)
            es_label = _METRIC_DISPLAY[args.early_stop_metric]
            es_ks = ','.join(map(str, args.top_k))

            if current_metric > best_metric:
                best_metric = current_metric
                best_round = round_idx
                patience_counter = 0
                logging.info(
                    '*** New best performance! Sum(%s@%s) = %.6f (Round %d)',
                    es_label, es_ks, best_metric, round_idx,
                )
            else:
                patience_counter += 1
                logging.info(
                    'No improvement (%d/%d), Best Sum(%s@%s) = %.6f (Round %d)',
                    patience_counter, args.patience, es_label, es_ks, best_metric, best_round,
                )

            if patience_counter >= args.patience:
                logging.info('=' * 80)
                logging.info(
                    'Early stopping triggered! Validation %s has not improved for %d evaluations',
                    es_label, args.patience,
                )
                logging.info(
                    'Best round: Round %d, Sum(%s@%s) = %.6f',
                    best_round, es_label, es_ks, best_metric,
                )
                logging.info('=' * 80)
                break

    logging.info('=' * 80)
    logging.info('Training finished!')
    logging.info(
        'Best validation Sum(%s@%s) = %.6f achieved at Round %d',
        _METRIC_DISPLAY[args.early_stop_metric],
        ','.join(map(str, args.top_k)),
        best_metric,
        best_round,
    )
    logging.info('=' * 80)

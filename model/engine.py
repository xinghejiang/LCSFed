import copy
import logging

import numpy as np
import torch
import torch.nn as nn

from model.client import Client, Server
from utils.utils import (
    EVAL_METRIC_NAMES,
    compute_eval_metrics,
    ldp_perturb_laplace,
)


class Engine:
    def __init__(self, args):
        self.args = args

        self.lr_u_emb = (args.lr_client / args.clients_sample_ratio * args.lr_eta - args.lr_client)
        self.lr_i_emb = (args.lr_client * args.num_items * args.lr_eta - args.lr_client)
        self.lr_mlp = args.lr_client
        self.lr_server = args.lr_server
        self.wd = args.wd
        self.epoch = args.local_epoch

        self.server_model_param = {}
        self.client_model_params = {}
        self.client_crit = nn.BCELoss()
        self.server_crit = nn.MSELoss()

        self.server_opt = None

        logging.info('=' * 80)
        logging.info('Learning Rates:')
        logging.info('  lr_u_emb (user embedding, SGD): {:.6f}'.format(self.lr_u_emb))
        logging.info('  lr_i_emb (item embedding, SGD): {:.6f}'.format(self.lr_i_emb))
        logging.info('  lr_mlp (user_align_func, Adam): {:.6f}'.format(self.lr_mlp))
        logging.info('  lr_mlp (scoring_func, Adam): {:.6f}'.format(self.lr_mlp))
        logging.info('  lr_server (item_align, Adam): {:.6f}'.format(self.lr_server))
        logging.info('  weight_decay: {:.6f}'.format(self.wd))
        logging.info('  align_gu: {:.6f}'.format(args.align_gu))
        logging.info('  align_gi: {:.6f}'.format(args.align_gi))
        logging.info('=' * 80)

        self.ldp_enable = bool(getattr(args, 'ldp_enable', False))
        self.ldp_laplace_lambda = float(getattr(args, 'ldp_laplace_lambda', 0.0))
        if self.ldp_enable:
            logging.info(
                'LDP enabled: Laplace(0, lambda=%.6f) on all uploaded params '
                '(item_emb.weight, user_align_func.*, scoring_func.*); '
                'private user_emb.weight is NOT perturbed',
                self.ldp_laplace_lambda,
            )
            logging.info('=' * 80)

    def _init_server_optimizer(self):
        if self.server_opt is None and hasattr(self, 'server_model'):
            self.server_opt = torch.optim.Adam(
                self.server_model.parameters(),
                lr=self.lr_server,
                weight_decay=self.wd,
            )

    def _create_client_optimizer(self, model_client):
        optimizer_sgd = torch.optim.SGD(
            [
                {'params': model_client.item_emb.parameters(), 'lr': self.lr_i_emb},
                {'params': model_client.user_emb.parameters(), 'lr': self.lr_u_emb},
            ],
            weight_decay=self.wd,
        )

        optimizer_adam = torch.optim.Adam(
            [
                {'params': model_client.user_align_func.parameters()},
                {'params': model_client.scoring_func.parameters()},
            ],
            lr=self.lr_mlp,
            weight_decay=self.wd,
        )

        return optimizer_sgd, optimizer_adam

    def aggregate_clients_params(self, round_user_params, participant_sample_counts):
        users = list(round_user_params.keys())
        counts = [max(0, int(participant_sample_counts[u])) for u in users]
        total = sum(counts)
        if total <= 0:
            logging.warning(
                'FedAvg: total sample count is 0; using uniform weights over %d clients',
                len(users),
            )
            norm_w = {u: 1.0 / len(users) for u in users}
        else:
            norm_w = {u: c / total for u, c in zip(users, counts)}

        t = 0
        for user in users:
            w = norm_w[user]
            user_params = round_user_params[user]
            if t == 0:
                self.server_model_param = copy.deepcopy(user_params)
                for key in self.server_model_param.keys():
                    self.server_model_param[key].data.mul_(w)
            else:
                for key in user_params.keys():
                    self.server_model_param[key].data.add_(user_params[key].data, alpha=w)
            t += 1

    def server_train_item_align(self, warm_item_ids):
        if len(warm_item_ids) == 0:
            logging.warning('No warm items for server training, skipping item_align_func training')
            return

        self.server_model.train()
        self.server_opt.zero_grad()

        warm_item_ids_tensor = torch.tensor(warm_item_ids, dtype=torch.long)
        target_item_emb = self.server_model_param['item_emb.weight'][warm_item_ids_tensor]

        warm_item_ids_tensor = warm_item_ids_tensor.to(self.args.device)
        target_item_emb = target_item_emb.to(self.args.device)

        warm_item_llm = self.server_model.item_llm_emb[warm_item_ids_tensor]
        predicted_item_emb = self.server_model(warm_item_llm)

        loss = self.server_crit(predicted_item_emb, target_item_emb)

        loss.backward()
        self.server_opt.step()

        logging.info(
            'Server item_align training: loss={:.6f}, warm_items={}'.format(
                loss.item(), len(warm_item_ids),
            ),
        )

    def _sample_participants(self, all_train_data):
        user_ids = list(all_train_data.keys())
        if self.args.clients_sample_ratio <= 1:
            num_parts = min(int(self.args.clients_sample_ratio * self.args.num_users), len(user_ids))
        else:
            num_parts = min(int(self.args.clients_sample_ratio * len(user_ids)), len(user_ids))
        return np.random.choice(user_ids, num_parts, replace=False)

    def _new_client_with_state(self, user, round_id):
        user_llm_emb = self.client_model.user_llm_emb
        model_client = self.client_model.__class__(self.args, user_llm_emb)
        model_client = model_client.to(self.args.device)
        user_param_dict = self.client_model.state_dict()
        if round_id != 0:
            if user in self.client_model_params:
                user_specific_params = self.client_model_params[user]
                for key, value in user_specific_params.items():
                    user_param_dict[key] = value.clone().detach()
            for key, value in self.server_model_param.items():
                user_param_dict[key] = value.clone().detach()
        model_client.load_state_dict(user_param_dict)
        return model_client

    def _train_client_dataloader(self, model_client, user_dataloader, round_totals):
        optimizer_sgd, optimizer_adam = self._create_client_optimizer(model_client)
        model_client.train()
        w_gu = self.args.align_gu
        w_gi = self.args.align_gi
        item_llm_emb = self.server_model.item_llm_emb
        item_align_head = self.server_model.item_align_func

        for _epoch in range(self.epoch):
            for _batch_id, (X, y) in enumerate(user_dataloader):
                X, y = X.to(self.args.device), y.to(self.args.device)
                optimizer_sgd.zero_grad()
                optimizer_adam.zero_grad()

                rating, uid_emb, align_uid_emb, iid_emb, align_iid_emb = model_client(
                    X,
                    item_llm_emb=item_llm_emb,
                    item_align_head=item_align_head,
                )

                y = y.squeeze().float().view_as(rating)

                loss_local = self.client_crit(rating, y)
                loss_gu = self.server_crit(uid_emb, align_uid_emb)
                loss_gi = self.server_crit(iid_emb, align_iid_emb)

                loss = loss_local + loss_gu * w_gu + loss_gi * w_gi
                loss.backward()
                optimizer_sgd.step()
                optimizer_adam.step()

                round_totals['total_loss_local'] += loss_local.item()
                round_totals['total_loss_gu'] += loss_gu.item()
                round_totals['total_loss_gi'] += loss_gi.item()
                round_totals['total_loss'] += loss.item()
                round_totals['total_batches'] += 1

    def _store_client_params_after_train(self, model_client, user, round_participant_params):
        client_param = model_client.state_dict()
        self.client_model_params[user] = {}
        self.client_model_params[user]['user_emb.weight'] = (
            client_param['user_emb.weight'].clone().detach().cpu()
        )
        round_participant_params[user] = {}
        for key in client_param.keys():
            if key != 'user_emb.weight' and key != 'user_llm_emb':
                tensor = client_param[key].clone().detach().cpu()
                if self.ldp_enable:
                    tensor = ldp_perturb_laplace(tensor, laplace_lambda=self.ldp_laplace_lambda)
                round_participant_params[user][key] = tensor

    def _log_round_loss_summary(self, round_id, participants, round_totals):
        if round_totals['total_batches'] <= 0:
            return
        n = round_totals['total_batches']
        logging.info('=' * 80)
        logging.info(
            'Round {} Summary: avg_loss={:.6f}, avg_loss_local={:.6f}, '
            'avg_loss_gu={:.6f}, avg_loss_gi={:.6f}, participants={}, total_batches={}'.format(
                round_id,
                round_totals['total_loss'] / n,
                round_totals['total_loss_local'] / n,
                round_totals['total_loss_gu'] / n,
                round_totals['total_loss_gi'] / n,
                len(participants),
                n,
            ),
        )
        logging.info('=' * 80)

    def _collect_warm_item_ids(self, all_train_data, participants):
        warm_item_ids = set()
        for user in participants:
            for _batch_id, (X, _y) in enumerate(all_train_data[user]):
                warm_item_ids.update(X[:, 1].tolist())
        return sorted(warm_item_ids)

    def fed_train_a_round(self, all_train_data, round_id):
        participants = self._sample_participants(all_train_data)
        round_participant_params = {}
        participant_sample_counts = {}
        round_totals = {
            'total_loss_local': 0.0,
            'total_loss_gu': 0.0,
            'total_loss_gi': 0.0,
            'total_loss': 0.0,
            'total_batches': 0,
        }

        for user in participants:
            participant_sample_counts[user] = len(all_train_data[user].dataset)
            model_client = self._new_client_with_state(user, round_id)
            self._train_client_dataloader(model_client, all_train_data[user], round_totals)
            self._store_client_params_after_train(model_client, user, round_participant_params)

        self._log_round_loss_summary(round_id, participants, round_totals)
        self.aggregate_clients_params(round_participant_params, participant_sample_counts)
        warm_item_ids = self._collect_warm_item_ids(all_train_data, participants)
        self.server_train_item_align(warm_item_ids)

    def fed_evaluate(self, evaluate_data, popularity, num_eval_items):
        all_results_with_iids = []

        self.server_model.eval()

        for user in evaluate_data.keys():
            targets, preds, iids = [], [], []

            user_llm_emb = self.client_model.user_llm_emb
            user_model = self.client_model.__class__(self.args, user_llm_emb)
            user_model = user_model.to(self.args.device)

            user_param_dict = self.client_model.state_dict()

            if user in self.client_model_params:
                user_specific_params = self.client_model_params[user]
                for key, value in user_specific_params.items():
                    user_param_dict[key] = value.clone().detach()

            if hasattr(self, 'server_model_param'):
                for key, value in self.server_model_param.items():
                    user_param_dict[key] = value.clone().detach()

            user_model.load_state_dict(user_param_dict)
            user_model.eval()

            item_llm_emb = self.server_model.item_llm_emb
            item_align_head = self.server_model.item_align_func

            with torch.no_grad():
                for X, y in evaluate_data[user]:
                    X, y = X.to(self.args.device), y.to(self.args.device)
                    pred = user_model.cold_predict(X, item_llm_emb, item_align_head).squeeze(-1)
                    y = y.squeeze().float().view_as(pred)
                    targets.extend(y.tolist())
                    preds.extend(pred.tolist())
                    iids.extend(X[:, 1].cpu().tolist())
            all_results_with_iids.append((targets, preds, iids))

        metrics = getattr(self.args, 'eval_metrics', list(EVAL_METRIC_NAMES))
        return compute_eval_metrics(
            all_results_with_iids,
            self.args.top_k,
            metrics,
            popularity=popularity,
            num_eval_items=num_eval_items,
        )


class MLPEngine(Engine):
    def __init__(self, args, user_llm_emb, item_llm_emb):
        super().__init__(args)

        self.client_model = Client(args, user_llm_emb)
        self.server_model = Server(args, item_llm_emb)

        self.client_model = self.client_model.to(args.device)
        self.server_model = self.server_model.to(args.device)

        self._init_server_optimizer()

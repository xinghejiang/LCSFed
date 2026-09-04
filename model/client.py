import torch
import torch.nn as nn


class Client(nn.Module):
    def __init__(self, args, user_llm_emb):
        super().__init__()
        num_users = args.num_users
        num_items = args.num_items
        latent_dim = args.latent_dim

        content_dim = args.content_dim
        self.register_buffer('user_llm_emb', user_llm_emb)

        self.user_emb = nn.Embedding(num_users, latent_dim)
        self.item_emb = nn.Embedding(num_items, latent_dim)

        self.user_align_func = nn.Linear(content_dim, latent_dim)
        self.scoring_func = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1)
        )

        self.logistic = nn.Sigmoid()

    def forward(self, x, item_llm_emb=None, item_align_head=None):
        user_id = x[:, 0]
        item_id = x[:, 1]

        user_llm = self.user_llm_emb[user_id]
        align_uid_emb = self.user_align_func(user_llm)
        uid_emb = self.user_emb(user_id)

        iid_emb = self.item_emb(item_id)
        align_iid_emb = None
        if item_llm_emb is not None and item_align_head is not None:
            item_llm = item_llm_emb[item_id]
            align_iid_emb = item_align_head(item_llm)

        vector = torch.cat([uid_emb, iid_emb], dim=-1)
        score = self.scoring_func(vector)
        rating = self.logistic(score)
        return rating, uid_emb, align_uid_emb, iid_emb, align_iid_emb

    def cold_predict(self, x, item_llm_emb, item_align_head):
        user_id = x[:, 0]
        item_id = x[:, 1]

        user_llm = self.user_llm_emb[user_id]
        align_uid_emb = self.user_align_func(user_llm)

        item_llm = item_llm_emb[item_id]
        align_iid_emb = item_align_head(item_llm)

        vector = torch.cat([align_uid_emb, align_iid_emb], dim=-1)
        score = self.scoring_func(vector)
        rating = self.logistic(score)

        return rating


class Server(nn.Module):
    def __init__(self, args, item_llm_emb):
        super().__init__()
        content_dim = args.content_dim
        latent_dim = args.latent_dim

        self.item_align_func = nn.Linear(content_dim, latent_dim)
        self.register_buffer('item_llm_emb', item_llm_emb)

    def forward(self, item_llm_batch=None):
        if item_llm_batch is None:
            item_llm_batch = self.item_llm_emb
        return self.item_align_func(item_llm_batch)

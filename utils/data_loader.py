import os
import torch
import random
import logging
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

class MovieLensDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.features = [
            'uid',
            'iid'
        ]
        feature_tensors = [
            torch.tensor(data[col].values, dtype=torch.long)
            for col in self.features
        ]
        self.feature_matrix = torch.stack(feature_tensors, dim=1)
        self.labels = torch.tensor(data['rating'].values, dtype=torch.float)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.feature_matrix[idx], self.labels[idx]

class MovieLensDataLoader(DataLoader):
    def __init__(self, 
                 data_dir, 
                 num_negatives, 
                 batch_size,
                 emb_type,
                 content_dim,
                 num_users,
                 num_items):
        self.data_dir = data_dir
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.num_users = num_users
        self.num_items = num_items

        data_dict = self._load_raw_data(self.num_users, self.num_items)

        self.train_data = data_dict[0]
        self.vali_data = data_dict[1]
        self.test_data = data_dict[2]

        self._vali_popularity, self._vali_num_items = self._compute_popularity(self.vali_data)
        self._test_popularity, self._test_num_items = self._compute_popularity(self.test_data)

    def _compute_popularity(self, df):
        counts = np.zeros(self.num_items, dtype=np.float64)
        vc = df['iid'].value_counts()
        for iid, cnt in vc.items():
            iid = int(iid)
            if 0 <= iid < self.num_items:
                counts[iid] = float(cnt)
        total = counts.sum()
        popularity = counts / total if total > 0 else counts
        num_eval_items = int(df['iid'].nunique())
        return popularity, num_eval_items

    def _load_raw_data(self, num_users, num_items):
        train_data_path = os.path.join(self.data_dir, 'train.csv')
        vali_data_path = os.path.join(self.data_dir, 'vali.csv')
        test_data_path = os.path.join(self.data_dir, 'test.csv')

        train_data = pd.read_csv(train_data_path)
        vali_data = pd.read_csv(vali_data_path)
        test_data = pd.read_csv(test_data_path)

        train_data = train_data[
            (train_data['uid'] < num_users) & (train_data['iid'] < num_items)]
        vali_data = vali_data[
            (vali_data['uid'] < num_users) & (vali_data['iid'] < num_items)]
        test_data = test_data[
            (test_data['uid'] < num_users) & (test_data['iid'] < num_items)]

        return train_data, vali_data, test_data
    
    def _prepare_test_data(self, df_data):
        all_users = df_data['uid'].unique()
        all_items = df_data['iid'].unique()

        observed_interactions = set(zip(df_data['uid'], df_data['iid']))
        all_combinations = []
        for uid in all_users:
            for iid in all_items:
                all_combinations.append((uid, iid))

        result_data = []
        for uid, iid in all_combinations:
            if (uid, iid) in observed_interactions:
                rating = 1.0
            else:
                rating = 0.0
            result_data.append({'uid': uid, 'iid': iid, 'rating': rating})
        
        result_df = pd.DataFrame(result_data)

        result_data_dict = {}
        for uid, user_data in result_df.groupby('uid'):
            dataset = MovieLensDataset(user_data)
            result_data_dict[uid] = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        return result_data_dict
    
    @property
    def get_train_data(self):
        return self.train_data
    
    @property
    def get_test_data(self):
        test_data_dict = self._prepare_test_data(self.test_data)
        return test_data_dict
    
    @property
    def get_vali_data(self):
        vali_data_dict = self._prepare_test_data(self.vali_data)
        return vali_data_dict

    @property
    def get_vali_popularity(self):
        return self._vali_popularity

    @property
    def get_vali_num_items(self):
        return self._vali_num_items

    @property
    def get_test_popularity(self):
        return self._test_popularity

    @property
    def get_test_num_items(self):
        return self._test_num_items

def negative_sampling(train_data, num_negatives, batch_size):
    item_pool = set(train_data['iid'])
    inter_map = train_data.groupby('uid')['iid'].apply(set).to_dict()
    samples = []
    for user_id, user_data in train_data.groupby('uid'):
        neg_candidates = list(item_pool - inter_map[user_id])
        
        for row in user_data.itertuples():
            user_id, item_id = row.uid, row.iid
            samples.append([user_id, item_id, 1.0])
            
            if not neg_candidates:
                continue
            
            k = min(num_negatives, len(neg_candidates))
            negs = random.sample(neg_candidates, k)
            
            for neg_item_id in negs:
                samples.append([user_id, neg_item_id, 0.0])
                
            for neg_item_id in negs:
                neg_candidates.remove(neg_item_id)
                
    samples = pd.DataFrame(samples, columns=['uid', 'iid', 'rating'])
    

    train_data_dict = {}
    for uid, user_data in samples.groupby('uid'):
        dataset = MovieLensDataset(user_data)
        train_data_dict[uid] = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_data_dict
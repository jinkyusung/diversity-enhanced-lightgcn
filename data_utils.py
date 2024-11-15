import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from scipy.sparse import csr_matrix
from time import time
import scipy.sparse as sp


""" Convert Yelp2018 Raw Data to np.array """
class Yelp2018:
    def __init__(self, train_file, test_file):
        self.train_file = train_file
        self.test_file  = test_file
        self._read_file()

    def _read_file(self):
        n_user = 0; m_item = 0
        trainDataSize = 0; testDataSize = 0
        trainItem, trainUser = [], []
        testItem, testUser = [], []
        
        with open(self.train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    m_item = max(m_item, max(items))
                    n_user = max(n_user, uid)
                    trainDataSize += len(items)

        with open(self.test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    m_item = max(m_item, max(items))
                    n_user = max(n_user, uid)
                    testDataSize += len(items)

        n_user += 1
        m_item += 1

        # No return, Just Set as class member variable (self).
        self.n_user = n_user
        self.m_item = m_item

        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        self.trainDataSize = trainDataSize

        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        self.testDataSize = testDataSize
    
    @property
    def num_user(self):
        return self.n_user
    
    @property
    def num_item(self):
        return self.m_item

    @property
    def train_user(self):
        return self.trainUser
    
    @property
    def train_item(self):
        return self.trainItem
    
    @property
    def train_interaction(self):
        return self.trainDataSize
    
    @property
    def test_user(self):
        return self.testUser
    
    @property
    def test_item(self):
        return self.testItem
    
    @property
    def test_interaction(self):
        return self.testDataSize


""" Construct User-Item Adjacency Matrix """
class AdjacencyMatrix:
    def __init__(self, train_user, train_item, n_user, m_item, device):
        self.n_user = n_user
        self.m_item = m_item
        self.device = device
        self.UserItemNet = csr_matrix(
            ( np.ones(len(train_user)), (train_user, train_item) ),
            shape=(n_user, m_item)
        )

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def get_sparse_graph(self, adj_mat_file):
        print("loading adjacency matrix")
        graph = None

        if graph is None:
            try:
                pre_adj_mat = sp.load_npz(adj_mat_file)
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_user + self.m_item, self.n_user + self.m_item), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_user, self.n_user:] = R
                adj_mat[self.n_user:, :self.n_user] = R.T
                adj_mat = adj_mat.todok()
                adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(adj_mat_file, norm_adj)

                graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                graph = graph.coalesce().to(self.device)
                print("don't split the matrix")
        return graph
        

""" Train Dataset for train_dataloader """
class PairwiseTrainData(Dataset):
    def __init__(self, train_user, train_item, n_user, m_item):
        self.train_user = train_user
        self.train_item = train_item

        self.n_user = n_user
        self.m_item = m_item

        self.history = self._make_dict(keys=train_user, values=train_item)
        self.train_neg_item = self._sample_negs()

    def _make_dict(self, keys: np.array, values: np.array) -> dict:
        sorted_indices = np.argsort(keys)
        sorted_keys = keys[sorted_indices]
        sorted_values = values[sorted_indices]
        unique_keys, start_indices = np.unique(sorted_keys, return_index=True)
        split_values = np.split(sorted_values, start_indices[1:])
        obj = dict(zip(unique_keys, split_values))
        return obj

    def _sample_negs(self):
        train_neg_item = []
        for user in self.train_user:
            while True:
                sampled_item = np.random.randint(self.m_item)
                if sampled_item not in self.history[user]:
                    break
            train_neg_item.append(sampled_item)
        return np.array(train_neg_item)

    def __getitem__(self, idx):
        batch_user = self.train_user[idx]
        batch_pos_item = self.train_item[idx]
        batch_neg_item = self.train_neg_item[idx]
        return batch_user, batch_pos_item, batch_neg_item
    
    def __len__(self):
        return self.train_user.size
    


""" Test Dataset for test_dataloader """
class TestData(Dataset):
    def __init__(self, train_user, train_item, test_user, test_item):
        self.unique_test_user = np.unique(test_user)
        self.history = self._make_dict(keys=train_user, values=train_item)
        self.label = self._make_dict(keys=test_user, values=test_item)

    def _make_dict(self, keys: np.array, values: np.array) -> dict:
        sorted_indices = np.argsort(keys)
        sorted_keys = keys[sorted_indices]
        sorted_values = values[sorted_indices]
        unique_keys, start_indices = np.unique(sorted_keys, return_index=True)
        split_values = np.split(sorted_values, start_indices[1:])
        obj = dict(zip(unique_keys, split_values))
        return obj

    def __getitem__(self, idx):
        batch_user = self.unique_test_user[idx]
        batch_history = self.history[batch_user] if batch_user in self.history else []
        batch_label = self.label[batch_user] if batch_user in self.label else []
        return batch_user, batch_history, batch_label

    def __len__(self):
        return self.unique_test_user.size


""" 
collate function for test_dataloader 
In this case, The number of Each user's history or label items are different.
So, To use collate process of torch.dataloader, we should implement padding.

"""
def collate_fn(batch):
    users, histories, test_items = zip(*batch)

    users = [torch.tensor(user) for user in users]
    histories = [torch.tensor(hist) for hist in histories]
    test_items = [torch.tensor(test) for test in test_items]

    padded_histories = pad_sequence(histories, batch_first=True, padding_value=-1)
    padded_test_items = pad_sequence(test_items, batch_first=True, padding_value=-1)

    return users, padded_histories, padded_test_items


""" We have to delete padding for calculating metrics. """
def remove_padding(batch: torch.tensor) -> list[torch.tensor]:
    mask = batch != -1
    non_padded_items = [user[mask[idx]] for idx, user in enumerate(batch)]
    return non_padded_items

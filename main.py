import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import yaml
from time import time
from pathlib import Path

# User Define #
from data_utils import Yelp2018, AdjacencyMatrix, PairwiseTrainData, remove_padding, TestData, collate_fn
from model import LightGCN
from loss import loss_dict
from procedure import c, console, train_loop, test_loop
from evaluator import TopKEvaluator


# Make directionary for model train, eval results #
try:
    Path(c.path.model).mkdir(parents=True, exist_ok=True)
    console(f"Folder created at: {c.path.model}")
except Exception as e:
    console(f"An error occurred: {e}")


# GPU or CPU check #
device_str = ("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(device_str)
console(f"Your Device: {device_str}")


# Load Data #
yelp2018 = Yelp2018(c.data.train, c.data.test)

num_user = yelp2018.num_user
num_item = yelp2018.num_item

train_user = yelp2018.train_user
train_item = yelp2018.train_item
train_interaction = yelp2018.train_interaction

test_user = yelp2018.test_user
test_item = yelp2018.test_item
test_interaction = yelp2018.test_interaction


# Yelp2018 Statistics Check #
console("Yelp2018")
console(
    f"""
#user = {num_user}
#item = {num_item}

#interactions
    (train) {train_interaction}
    (test)  {test_interaction}
    (total) {train_interaction + test_interaction}

Sparsity = {(train_interaction + test_interaction) / (num_user * num_item)}
"""
)


# Construct Normalized Symmetric Adj Matrix #
adjacency_matrix = AdjacencyMatrix(train_user, train_item, num_user, num_item, device)
graph = adjacency_matrix.get_sparse_graph(c.data.adj_mat)  # This is The Normalized Adjacency Matrix.


# Make `torch` dataloader #
train_dataset = PairwiseTrainData( train_user, train_item, num_user, num_item, do_neg_sampling=c.model.neg_sampling )
train_dataloader = DataLoader( train_dataset, batch_size=c.batch.train, shuffle=True, num_workers=0 )
test_dataset = TestData(train_user, train_item, test_user, test_item)
test_dataloader = DataLoader(test_dataset, batch_size=c.batch.test, shuffle=False, collate_fn=collate_fn)


# Get degree per item for use in the top-k metric calculation
train_test_user = np.concatenate([train_user, test_user])
train_test_item = np.concatenate([train_item, test_item])
train_test_item_degree = torch.tensor(np.bincount(train_test_item), dtype=torch.float32).to(device)


# Get degree per item only in the train set for use in the loss function while training
train_item_degree = torch.tensor(np.bincount(train_item), dtype=torch.float32).to(device)
del train_test_user, train_test_item


# Model Train & Eval #
model = LightGCN(num_user, num_item, c.model.n_layers, c.model.embed_dim, graph)
model.to(device)
loss_fn = loss_dict[c.loss](item_degree=train_item_degree).loss_fn
optimizer = torch.optim.Adam(model.parameters(), lr=c.model.lr)
evaluator = TopKEvaluator(c.topk, c.metrics, device=device, item_degree=train_test_item_degree)

for epoch in range(1, c.model.epochs + 1):
    console(f"Epoch {epoch}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer, device)
    test_loop(test_dataloader, model, loss_fn, evaluator, epoch, device)
    
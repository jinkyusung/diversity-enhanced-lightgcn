import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
from tqdm import tqdm
import scipy.sparse as sp
from time import time


from logger import console
import config as c


device_str = ("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device(device_str)
console(f"Your Device: {device_str}")


from data_utils import Yelp2018
from data_utils import AdjacencyMatrix
from data_utils import PairwiseTrainData
from model import LightGCN

from evaluator import TopKEvaluator
from data_utils import remove_padding

from loss import loss_dict
from data_utils import TestData, collate_fn
from evaluator import TopKEvaluator


yelp2018 = Yelp2018(c.train_file, c.test_file)

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


adjacency_matrix = AdjacencyMatrix(train_user, train_item, num_user, num_item, device)
graph = adjacency_matrix.get_sparse_graph(
    c.adj_mat_file
)  # This is The Normalized Adjacency Matrix.


train_dataset = PairwiseTrainData(
    train_user, train_item, num_user, num_item, do_neg_sampling=c.do_neg_sampling
)
train_dataloader = DataLoader(
    train_dataset, batch_size=c.train_batch_size, shuffle=True, num_workers=0
)

train_test_user = np.concatenate([train_user, test_user])
train_test_item = np.concatenate([train_item, test_item])

# Get degree per item for use in the top-k metric calculation
train_test_item_degree = torch.tensor(
    np.bincount(train_test_item), dtype=torch.float32
).to(device)

# Get degree per item only in the train set for use in the loss function while training
train_item_degree = torch.tensor(np.bincount(train_item), dtype=torch.float32).to(
    device
)

del train_test_user, train_test_item

model = LightGCN(num_user, num_item, c.n_layers, c.embed_dim, graph)
model.to(device)

def train_loop(train_dataloader, model, loss_fn, optimizer: torch.optim.Optimizer):
    model.train()

    loss_sum = 0

    size = len(train_dataloader.dataset)
    num_batches = len(train_dataloader)

    for batch_num, minibatch in enumerate(train_dataloader):
        optimizer.zero_grad()

        user:           torch.Tensor = minibatch[0].to(device)
        pos_item:       torch.Tensor = minibatch[1].to(device)
        if c.do_neg_sampling:
            neg_item:   torch.Tensor = minibatch[2].to(device)

        if c.do_neg_sampling:
            result = model(user, pos_item, neg_items=neg_item)
        else:
            result = model(user, pos_item)

        loss = loss_fn(**result, pos_item=pos_item)

        loss_sum += loss.item()
        loss.backward()
        optimizer.step()

        if batch_num % 100 == 0:
            console(
                f"loss: {loss.item():>7f} [{c.train_batch_size * batch_num + len(minibatch[0]):>5d}/{size:>5d}]"
            )

    avg_loss = loss_sum / num_batches
    console(f"Train Avg loss: {avg_loss:>7f}")


# A dictionary to store the best metric values along epochs
best_metric = dict()

# Boolean indicating whether to write the header in the metric file
write_header = True


def test_loop(dataloader, model, loss_fn, evaluator: TopKEvaluator, epoch: int):
    global best_metric, write_header

    model.eval()

    num_batches = len(dataloader)

    metrics_result_dict = dict()

    with torch.no_grad():
        for minibatch in tqdm(dataloader):
            user:       torch.Tensor = minibatch[0].to(device)
            history:    torch.Tensor = minibatch[1].to(device)  # 각 유저 별 train 에서 존재하는 아이템
            label:      torch.Tensor = minibatch[2].to(device)  # 각 유저 별 test  에서 존재하는 아이템

            history:    list[torch.Tensor] = remove_padding(history)
            label:      list[torch.Tensor] = remove_padding(label)

            pred:       torch.Tensor = model.get_users_rating_prediction(user)
            assert pred.shape == (len(user), num_item)

            result_dict = evaluator.evaluate(pred, history, label)
            for metric in result_dict:
                if metric not in metrics_result_dict:
                    metrics_result_dict[metric] = 0
                metrics_result_dict[metric] += result_dict[metric]

    for metric in metrics_result_dict:
        metrics_result_dict[metric] /= num_batches

    # Save metrics to a text file
    with open(c.metric_results_file, "a") as f:
        if write_header:
            f.write("epoch\t")
            for metric in metrics_result_dict:
                f.write(f"{metric}\t")
            f.write("\n")
            write_header = False

        f.write(f"{epoch}\t")
        for metric in metrics_result_dict:
            f.write(f"{metrics_result_dict[metric]:.4f}\t")
        f.write("\n")

    # Check and save the best models
    for metric in metrics_result_dict:
        if metric not in best_metric:
            best_metric[metric] = 0
        if metrics_result_dict[metric] > best_metric[metric]:
            best_metric[metric] = metrics_result_dict[metric]
            console(f"Best {metric} model updated. Saving the model.")
            torch.save(
                model.state_dict(), f"{c.best_model_dir}/best_{metric}_model.pth"
            )

    console(f"Eval results: ")
    for metric in metrics_result_dict:
        console(f"{metric}: {metrics_result_dict[metric]:.4f}", end=" ")
    console("\n")



loss_fn = loss_dict[c.loss_fn](item_degree=train_item_degree).loss_fn
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


test_dataset = TestData(train_user, train_item, test_user, test_item)
test_dataloader = DataLoader(
    test_dataset, batch_size=c.test_batch_size, shuffle=False, collate_fn=collate_fn
)


evaluator = TopKEvaluator(
    c.topk, c.metrics, device=device, item_degree=train_test_item_degree
)

for epoch in range(1, c.epochs + 1):
    console(f"Epoch {epoch}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn, evaluator, epoch)
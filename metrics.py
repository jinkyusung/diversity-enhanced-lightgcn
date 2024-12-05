import torch

def recall(pos_index: torch.Tensor, pos_len: torch.Tensor) -> torch.Tensor:
    """
    Calculate Recall@K
    params:
        pos_index: tensor with shape (batch_size, K) where 1 indicates relevant ('K' as in top-K)
        pos_len: tensor with shape (batch_size,) where each element indicates the total number of relevant items for a user
    return:
        Recall@K for each user in the batch
    """
    assert pos_index.shape[0] == pos_len.shape[0]
    return torch.sum(pos_index, axis=1) / pos_len

def ndcg(pos_index: torch.Tensor, pos_len: torch.Tensor) -> torch.Tensor:
    """
    Calculate NDCG@K
    params:
        pos_index: tensor with shape (batch_size, K) where 1 indicates relevant ('K' as in top-K)
        pos_len: tensor with shape (batch_size,) where each element indicates the total number of relevant items for a user
    return:
        NDCG@K for each user in the batch
    """
    
    assert pos_index.shape[0] == pos_len.shape[0]
    batch_size = pos_index.shape[0]
    K = pos_index.shape[1]

    device = pos_index.device
    
    bunmo = torch.log2(torch.arange(2, K+2, device=device).float())

    # Calculate DCG
    dcg = torch.sum(pos_index / bunmo, axis=1)

    ideal_pos_index = torch.zeros_like(pos_index)
    for i in range(batch_size):
        ideal_pos_index[i, :pos_len[i]] = 1

    # Calculate IDCG
    idcg = torch.sum(ideal_pos_index / bunmo, axis=1)

    return dcg / idcg

def diversity(topk_items: torch.Tensor, item_degree: torch.Tensor) -> torch.Tensor:
    """
    Calculate diversity of the recommendation defined as

    diversity(u) = \frac{1}{k} \sum_{i \in f(u)} \frac{1}{log(1 + deg(i))}

    where f(u) is the set of items recommended to user u and deg(i) is the degree of item

    params:
        topk_items: tensor with shape (batch_size, K) where each element indicates the item id
        item_degree: tensor with shape (n_items,) where each element indicates the degree of item
    return:
        Diversity for each user in the batch
    """

    diversity = torch.mean(1 / torch.log1p(item_degree[topk_items]), axis=1)

    return diversity

metrics_dict = {
    'recall': recall,
    'ndcg': ndcg,
    'diversity': diversity,
}

topk_metrics = ["recall", "ndcg", "diversity"]
import torch
from torch import nn
from abc import ABC, abstractmethod


class Model(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_users_rating_prediction(self, users):
        pass


class LightGCN(Model):
    def __init__(self, n_users, n_items, n_layers, embedding_dim, graph: torch.Tensor):
        """
        params:
            n_users: number of users
            n_items: number of items
            n_layers: number of layers of the LightGCN model
            embedding_dim: dimension of embedding
            graph: normalized user-item adjacency matrix (D^{-1/2}AD^{-1/2})
        """
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.graph = graph

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self.f = nn.Sigmoid()

    def _get_final_embedding(self):
        embedding = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )

        # all_embeddings_by_layer[i]: ith layer embedding of size (n_users + n_items, embedding_dim)
        all_embeddings_by_layer = [embedding]
        for _ in range(self.n_layers):
            embedding = torch.sparse.mm(self.graph, embedding)
            all_embeddings_by_layer.append(embedding)

        final_embedding = (torch.stack(all_embeddings_by_layer, dim=1)).mean(dim=1)
        user_final_embedding, item_final_embedding = final_embedding.split(
            [self.n_users, self.n_items], dim=0
        )

        # Sanity check
        assert user_final_embedding.size() == (self.n_users, self.embedding_dim)
        assert item_final_embedding.size() == (self.n_items, self.embedding_dim)

        return user_final_embedding, item_final_embedding

    def forward(self, users, pos_items, **kwargs):
        """
        params:
            users: a list of user ids
            pos_items: a list of positive item ids
            kwargs:
                neg_items: a list of negative item ids
        return:
            a dictionary with keys:
                'user_embedding', 'pos_item_embedding', 'neg_item_embedding',
                'user_final_embedding', 'pos_item_final_embedding', 'neg_item_final_embedding'
            neg_item_embedding and neg_item_final_embedding are excluded if negative items were not provided
        """
        user_final_embedding, item_final_embedding = self._get_final_embedding()

        # Sanity check
        assert len(users) == len(pos_items)
        if "neg_items" in kwargs:
            assert len(users) == len(kwargs["neg_items"])

        neg_items = kwargs.get("neg_items", None)
        user_final_embedding = user_final_embedding[users]
        pos_item_final_embedding = item_final_embedding[pos_items]
        if neg_items is not None:
            neg_item_final_embedding = item_final_embedding[neg_items]

        user_embedding = self.user_embedding(users)
        pos_item_embedding = self.item_embedding(pos_items)
        if neg_items is not None:
            neg_item_embedding = self.item_embedding(neg_items)

        result = {
            "user_embedding": user_embedding,
            "pos_item_embedding": pos_item_embedding,
            "user_final_embedding": user_final_embedding,
            "pos_item_final_embedding": pos_item_final_embedding,
        }
        if neg_items is not None:
            result["neg_item_embedding"] = neg_item_embedding
            result["neg_item_final_embedding"] = neg_item_final_embedding

        return result

    def get_users_rating_prediction(self, users):
        """
        Get the predicted ratings for the users

        params:
            users: a list of user ids
        return:
            a tensor of size (len(users), n_items) containing the predicted ratings for the users
            i.e. F(E_u[users] * E_i^T)
        """
        user_final_embedding, item_final_embedding = self._get_final_embedding()
        user_final_embedding = user_final_embedding[users, :]
        return self.f(torch.matmul(user_final_embedding, item_final_embedding.T))

    def to(self, device):
        # Override the to method to move self.graph as well
        self.graph = self.graph.to(device)
        return super(LightGCN, self).to(device)
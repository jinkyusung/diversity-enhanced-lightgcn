import torch
import torch.nn.functional as F


class BPRLoss:
    def __init__(self, reg_strength=1e-5, **kwargs):
        self.reg_strength = reg_strength

    def loss_fn(
        self,
        user_embedding,
        pos_item_embedding,
        neg_item_embedding,
        user_final_embedding,
        pos_item_final_embedding,
        neg_item_final_embedding,
        **kwargs,
    ):
        """
        params:
            user_embedding: user embedding of size (batch_size, embedding_dim)
            pos_item_embedding: positive item embedding of size (batch_size, embedding_dim)
            neg_item_embedding: negative item embedding of size (batch_size, embedding_dim)
            user_final_embedding: user final embedding of size (batch_size, embedding_dim)
            pos_item_final_embedding: positive item final embedding of size (batch_size, embedding_dim)
            neg_item_final_embedding: negative item final embedding of size (batch_size, embedding_dim)
        return:
            (bpr loss) + reg_strength * (regularization loss)
        """
        pos_scores = torch.sum(
            torch.mul(user_final_embedding, pos_item_final_embedding), dim=1
        )
        neg_scores = torch.sum(
            torch.mul(user_final_embedding, neg_item_final_embedding), dim=1
        )
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

        if self.reg_strength == 0:
            return bpr_loss
        else:
            batch_size = user_embedding.size(0)
            reg_loss = (
                user_embedding.norm(2).pow(2)
                + pos_item_embedding.norm(2).pow(2)
                + neg_item_embedding.norm(2).pow(2)
            ) / batch_size
            return bpr_loss + self.reg_strength / 2 * reg_loss


class DirectAULoss:
    def __init__(self, gamma=1.0, **kwargs):
        self.gamma = gamma

    @staticmethod
    def alignment(x, y):
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    @staticmethod
    def uniformity(x):
        return F.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def loss_fn(self, user_final_embedding, pos_item_final_embedding, **kwargs):
        """
        params:
            user_final_embedding: user final embedding of size (batch_size, embedding_dim)
            pos_item_final_embedding: positive item final embeidding of size (batch_size, embedding_dim)
        return:
            (alignment loss) + gamma * (uniformity loss)
        """
        normalized_user_e = F.normalize(user_final_embedding, dim=-1)
        normalized_item_e = F.normalize(pos_item_final_embedding, dim=-1)
        align = self.alignment(normalized_user_e, normalized_item_e)
        uniform = (
            self.uniformity(normalized_user_e) + self.uniformity(normalized_item_e)
        ) / 2
        loss = align + self.gamma * uniform
        return loss


class DeweightedDirectAULoss(DirectAULoss):
    def __init__(self, gamma=1.0, **kwargs):
        """
        params:
            item_degree: the degree of item tensor with shape (n_items,)
        """
        if "item_degree" not in kwargs:
            raise ValueError("item_degree must be provided for DeweightedDirectAULoss")
        super().__init__(gamma=gamma, **kwargs)
        self.item_degree: torch.Tensor = kwargs["item_degree"]

    @staticmethod
    def alignment(x, y, item_degree):
        """
        Calculate the deweighted alignment loss defined as
        \frac{1}{D} \sum_{(u, i) \in D} \frac{1}{log(deg(i)) + 1} ||\tilde{e_u} - \tilde{e_i}||^2
        where D is the set of user-item pairs, deg(i) is the degree of item i

        params:
            x: the normalized user final embedding of size (batch_size, embedding_dim)
            y: the normalized item final embedding of size (batch_size, embedding_dim)
            item_degree: the degree of item tensor with shpae (batch_size,)
        """
        return (x - y).norm(p=2, dim=1).pow(2).div(item_degree.log().add(1)).mean()

    def loss_fn(self, user_final_embedding, pos_item_final_embedding, **kwargs):
        pos_item = kwargs["pos_item"]
        item_degree = self.item_degree[pos_item]
        normalized_user_e = F.normalize(user_final_embedding, dim=-1)
        normalized_item_e = F.normalize(pos_item_final_embedding, dim=-1)
        align = self.alignment(normalized_user_e, normalized_item_e, item_degree)
        uniform = (
            self.uniformity(normalized_user_e) + self.uniformity(normalized_item_e)
        ) / 2
        loss = align + self.gamma * uniform
        return loss


loss_dict = {
    "bpr": BPRLoss,
    "directau": DirectAULoss,
    "deweighted_directau": DeweightedDirectAULoss,
}

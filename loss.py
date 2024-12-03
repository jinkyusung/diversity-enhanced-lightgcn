import torch
import torch.nn.functional as F


class BPRLoss:
    def __init__(self, reg_strength=1e-5):
        self.reg_strength = reg_strength

    def loss_fn(
        self,
        user_embedding,
        pos_item_embedding,
        neg_item_embedding,
        user_final_embedding,
        pos_item_final_embedding,
        neg_item_final_embedding,
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
            return bpr_loss + self.reg_strength * reg_loss


class DirectAULoss:
    def __init__(self, gamma=1.0):
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

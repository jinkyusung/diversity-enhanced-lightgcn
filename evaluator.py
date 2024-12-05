import torch
from metrics import metrics_dict, topk_metrics


class TopKEvaluator:
    def __init__(self, k, metrics: list, device, **kwargs):
        """
        The top-k metric evaluator.
        params:
            k: the number of items to recommend - the "k" in top-k
            metrics: a list of metrics to evaluate. Valid metrics are specified in `topk_metrics` in metrics.py
            device: device to run the evaluation
            **kwargs: additional arguments
                item_degree: item degree tensor of shape (n_items,). Required if diversity metric is used
        """
        self.k = k
        if "item_degree" in kwargs:
            self.item_degree: torch.Tensor = kwargs["item_degree"]

        # Check if metrics are valid
        for metric in metrics:
            if metric not in topk_metrics:
                raise ValueError(
                    f"Invalid metric: {metric}. Valid metrics are {topk_metrics}"
                )
            if "diversity" in metric and "item_degree" not in kwargs:
                raise ValueError(
                    "item_degree must be provided to calculate diversity metrics"
                )

        # Split metrics into "top-k metrics" and "degree metrics"
        self.degree_metrics = ["diversity"] if "diversity" in metrics else []
        self.metrics = (
            metrics
            if "diversity" not in metrics
            else [m for m in metrics if m != "diversity"]
        )
        self.device = device

    def evaluate(
        self, pred: torch.Tensor, history: list[torch.Tensor], label: list[torch.Tensor]
    ):
        """
        Evaluate the top-k recommendation.
        params:
            pred: predicted scores of shape (batch_size, n_items)
            history: a list of user's interacted items in the training set with length `batch_size`
            label: a list of user's positive items in the test set with length `batch_size`
        """

        # Set a very low score for history items in `pred`
        for i in range(len(history)):
            pred[i, history[i]] = -1e9

        # Get the top k items
        _, topk_items = torch.topk(pred, self.k, dim=1)

        # Evaluate whether the top k items contain the true items
        pos_index = torch.zeros_like(topk_items, dtype=torch.int, device=self.device)
        for i in range(len(label)):
            for j in range(self.k):
                if topk_items[i, j] in label[i]:
                    pos_index[i, j] = 1

        pos_len = torch.tensor(
            [len(l) for l in label], dtype=torch.int, device=self.device
        )

        # Calculate metrics
        result_dict = {}
        for metric in self.metrics:
            metric_func = metrics_dict[metric]
            met: torch.Tensor = metric_func(pos_index, pos_len)

            key = f"{metric}@{self.k}"
            result_dict[key] = met.mean().item()

        for metric in self.degree_metrics:
            metric_func = metrics_dict[metric]
            met: torch.Tensor = metric_func(topk_items, self.item_degree)

            key = f"{metric}"
            result_dict[key] = met.mean().item()

        return result_dict

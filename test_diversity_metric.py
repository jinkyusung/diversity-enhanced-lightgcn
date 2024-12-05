import torch
from evaluator import TopKEvaluator
import pytest

test_cases = [
    {
        "item_degree": [5, 10, 15, 20],
        "pred": [[0.1, 0.4, 0.3, 0.2], [0.5, 0.2, 0.1, 0.2]],
        "history": [[0], [0, 1]],
        "label": [[1, 3], [3]],
        "k": 2,
    },
    {
        "item_degree": [3, 1, 5, 2],
        "pred": [[0.9, 0.8, 0.7, 0.6], [0.4, 0.3, 0.2, 0.1]],
        "history": [[0], [0, 1]],
        "label": [[2, 3], [3]],
        "k": 3,
    },
]

answers = []

def calculate_answers():
    for test_case in test_cases:
        item_degree = torch.tensor(test_case["item_degree"])
        pred = torch.tensor(test_case["pred"])
        history = [torch.tensor(hist) for hist in test_case["history"]]
        k = test_case["k"]

        # Set a very low score for history items in `pred`
        for i in range(len(history)):
            pred[i, history[i]] = -1e9

        # Get the top k items
        _, topk_items = torch.topk(pred, k, dim=1)

        # Calculate diversity
        diversity = []
        for items in topk_items:
            diversity.append(torch.mean(1 / torch.log1p(item_degree[items])))
        diversity = torch.tensor(diversity)

        answers.append({"diversity": diversity.mean().item()})


def test_diversity():
    for i, test_case in enumerate(test_cases):
        item_degree = torch.tensor(test_case["item_degree"])
        evaluator = TopKEvaluator(
            k=test_case["k"], metrics=["diversity"], item_degree=item_degree
        )
        pred = torch.tensor(test_case["pred"])
        history = [torch.tensor(hist) for hist in test_case["history"]]
        label = [torch.tensor(lab) for lab in test_case["label"]]
        result = evaluator.evaluate(pred, history, label)

        assert result["diversity"] == pytest.approx(answers[i]["diversity"], rel=1e-2)

def setup_module():
    calculate_answers()

if __name__ == "__main__":
    pytest.main([__file__])
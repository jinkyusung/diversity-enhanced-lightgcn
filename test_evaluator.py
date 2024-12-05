import torch
import pytest
from evaluator import TopKEvaluator

# FILE: test_evaluator.py

test_cases = [
    {
        "pred":     [[0.1, 0.4, 0.3, 0.2],  [0.5, 0.2, 0.1, 0.2]],
        "history":  [[0],                   [0, 1]],
        "label":    [[1, 3],                [3]],
        "k": 2,
    },
    {
        "pred":     [[0.9, 0.8, 0.7, 0.6],  [0.4, 0.3, 0.2, 0.1]],
        "history":  [[0],                   [0, 1]],
        "label":    [[2, 3],                [3]],
        "k": 3,
    },
    {
        "pred":     [[0.2, 0.1, 0.4, 0.3],  [0.3, 0.2, 0.1, 0.4]],
        "history":  [[0, 2],                [3]],
        "label":    [[1],                   [0, 1, 2]],
        "k": 1,
    }
]

answers = [
    {"recall@2": 0.75, "ndcg@2": 0.8065735964},
    {"recall@3": 1.0, "ndcg@3": 0.66215},
    {"recall@1": 1/6, "ndcg@1": 0.5},
]

def test_evaluate():
    for i, test_case in enumerate(test_cases):
        pred = torch.tensor(test_case["pred"])
        history = [torch.tensor(hist) for hist in test_case["history"]]
        label = [torch.tensor(lab) for lab in test_case["label"]]
        k = test_case["k"]
        evaluator = TopKEvaluator(k=k, metrics=["recall", "ndcg"])
        result = evaluator.evaluate(pred, history, label)
        for key, value in answers[i].items():
            try:
                assert result[key] == pytest.approx(value, rel=1e-2)
            except AssertionError:
                print(f"Test case {i} failed for {key}: expected {value}, got {result[key]}")
                raise AssertionError


if __name__ == "__main__":
    pytest.main([__file__])

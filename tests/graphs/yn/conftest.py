from typing import Sequence

import torch
from domiknows.sensor.pytorch.learners import TorchLearner


class YnSpecificDummyLearner(TorchLearner):
    """
    Learner that makes different predictions based on input value and position in chain
    """

    def __init__(self, *pre, predictions: list[int]):
        TorchLearner.__init__(self, *pre)
        self.predictions = predictions

    def forward(self, x: Sequence) -> torch.Tensor:
        batch_size = len(x)
        result = torch.zeros(batch_size, 2)

        for i in range(batch_size):
            if i < len(self.predictions) and self.predictions[i] == 1000:
                result[i, 0] = -1000
                result[i, 1] = 1000
            elif i < len(self.predictions) and self.predictions[i] == -1000:
                result[i, 0] = 1000
                result[i, 1] = -1000
            elif i < len(self.predictions) and self.predictions[i] == 0:
                result[i, 0] = 0
                result[i, 1] = 0
        return result


def str_to_int_list(x: Sequence) -> torch.LongTensor:
    return torch.LongTensor([int(i) for i in x])


def make_labels(label_list: str) -> torch.LongTensor:
    labels = label_list.split("@@")
    label_nums = [1 if label == "Yes" else 0 if label == "No" else 2 for label in labels]
    return str_to_int_list(label_nums)


def make_question(
    questions: str, stories: str, relations: str, q_ids: str, labels: str
) -> tuple[torch.Tensor, list[str], list[str], list[str], torch.LongTensor, torch.LongTensor]:
    num_labels = make_labels(labels)
    ids = str_to_int_list(q_ids.split("@@"))
    return (
        torch.ones(len(questions.split("@@")), 1),
        questions.split("@@"),
        stories.split("@@"),
        relations.split("@@"),
        ids,
        num_labels,
    )

def assert_ilp_result_yn(q_node, label, expected_tensor):
    """Assert ILP predictions match expected value"""
    result = q_node.getAttribute(label, "ILP")
    assert torch.allclose(result, expected_tensor), (f"Label {label}: Expected {torch.tensor([0, 1])}, got {result}")

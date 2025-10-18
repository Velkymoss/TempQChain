from typing import Sequence

import torch
from domiknows.sensor.pytorch.learners import TorchLearner


class YnSpecificDummyLearner(TorchLearner):
    """
    Learner that makes different predictions based on input value and position in chain
    """

    def __init__(self, *pre, predictions: list[int], device=None):
        TorchLearner.__init__(self, *pre)
        self.predictions = predictions
        self.device = device

    def forward(self, x: Sequence) -> torch.Tensor:
        batch_size = len(x)
        result = torch.zeros(batch_size, 2, device=self.device)

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


def str_to_int_list(x: Sequence, device=None) -> torch.LongTensor:
    return (
        torch.LongTensor([int(i) for i in x]).to(device)
        if device is not None
        else torch.LongTensor([int(i) for i in x])
    )


def make_labels(label_list: str, device=None) -> torch.LongTensor:
    labels = label_list.split("@@")
    label_nums = [1 if label == "Yes" else 0 if label == "No" else 2 for label in labels]
    return str_to_int_list(label_nums, device=device)


def make_question(
    questions: str, stories: str, relations: str, q_ids: str, labels: str, device=None
) -> tuple[torch.Tensor, list[str], list[str], list[str], torch.LongTensor, torch.LongTensor]:
    num_labels = make_labels(labels, device=device)
    ids = str_to_int_list(q_ids.split("@@"), device=device)
    return (
        torch.ones(len(questions.split("@@")), 1, device=device)
        if device is not None
        else torch.ones(len(questions.split("@@")), 1),
        questions.split("@@"),
        stories.split("@@"),
        relations.split("@@"),
        ids,
        num_labels,
    )

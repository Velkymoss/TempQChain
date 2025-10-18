from typing import Sequence

import torch
from domiknows.sensor.pytorch.learners import TorchLearner

from tests.graphs.conftest import str_to_int_list


class FrSpecificDummyLearner(TorchLearner):
    """
    Learner that outputs one-hot-like predictions based on predefined label indices.

    Args:
        *pre: Positional arguments for TorchLearner
        predictions: List of label indices for each batch position.
                     Use -1 to output all zeros for that position.
        num_labels: Total number of labels
        high_score: Score for the predicted label (default: 1000)
        low_score: Score for non-predicted labels (default: -1000)
        device: Device for tensors
    """

    def __init__(
        self,
        *pre,
        predictions: list[int],
        num_labels: int,
        high_score: float = 1000.0,
        low_score: float = -1000.0,
        device=None,
    ):
        TorchLearner.__init__(self, *pre)
        self.predictions = predictions
        self.num_labels = num_labels
        self.high_score = high_score
        self.low_score = low_score
        self.device = device

    def forward(self, x: Sequence) -> torch.Tensor:
        batch_size = len(x)
        device = self.device or (x.device if isinstance(x, torch.Tensor) else "cpu")

        result = torch.full((batch_size, self.num_labels), self.low_score, device=device)

        for i in range(min(batch_size, len(self.predictions))):
            pred_idx = self.predictions[i]

            if pred_idx == -1:
                result[i, :] = 0
            elif 0 <= pred_idx < self.num_labels:
                result[i, pred_idx] = self.high_score
        return result


def make_question(
    questions: str, stories: str, relations: str, q_ids: str, labels: str, device=None
) -> tuple[torch.Tensor, list[str], list[str], list[str], torch.LongTensor, torch.LongTensor]:
    num_labels = str_to_int_list(labels.split("@@"), device=device)
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

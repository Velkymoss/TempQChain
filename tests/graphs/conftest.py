from typing import Sequence

import torch
from domiknows.sensor.pytorch.learners import TorchLearner

from tempQchain.programs.program_tb_dense_FR import QuestionData

LABEL_DIM = 6

all_labels = [
    "after",
    "before",
    "includes",
    "is_included",
    "simultaneous",
    "vague",
]


class DummyLearner(TorchLearner):
    """
    Deterministic learner that returns fixed predictions for relation classification.

    """

    def __init__(self, *pre, positive: bool = False):
        TorchLearner.__init__(self, *pre)
        self.positive = positive

    def forward(self, x: Sequence) -> torch.Tensor:
        """
        Returns dummy binary classification scores.

        Example:
            >>> # When positive=True (predicts class 1)
            >>> learner = DummyLearner(positive=True)
            >>> result = learner.forward([1, 2, 3])
            >>> result
            tensor([[-1000,  1000],
                    [-1000,  1000],
                    [-1000,  1000]])

            >>> # When positive=False (predicts class 0)
            >>> learner = DummyLearner(positive=False)
            >>> result = learner.forward([1, 2, 3])
            >>> result
            tensor([[ 1000, -1000],
                    [ 1000, -1000],
                    [ 1000, -1000]])
        """
        result = torch.zeros(len(x), 2)
        if self.positive:
            result[:, 0] = -1000
            result[:, 1] = 1000
        else:
            result[:, 0] = 1000
            result[:, 1] = -1000
        return result


class QuestionSpecificDummyLearner(TorchLearner):
    """Learner that makes different predictions based on question position in batch"""

    def __init__(self, *pre, predictions: list[bool]):
        """
        Args:
            predictions: List of booleans, one per question.
        """
        TorchLearner.__init__(self, *pre)
        self.predictions = predictions

    def forward(self, x: Sequence[bool]) -> torch.Tensor:
        batch_size = len(x)
        result = torch.zeros(batch_size, 2)

        for i in range(batch_size):
            if i < len(self.predictions) and self.predictions[i]:
                result[i, 0] = -1000
                result[i, 1] = 1000  # Predict class 1
            else:
                result[i, 0] = 1000
                result[i, 1] = -1000  # Predict class 0

        return result


def to_int_list(x: Sequence) -> torch.LongTensor:
    return torch.LongTensor([int(i) for i in x])


def check_symmetric(**kwargs) -> bool:
    args = list(kwargs.values())
    if len(args) != 2:
        return False
    arg1, arg2 = args[0], args[1]

    if arg1 == arg2:
        return False
    relation_arg2 = arg2.getAttribute("relation")
    if relation_arg2 == "":
        return False
    relation_describe = relation_arg2.split(",")
    if relation_describe[0] == "symmetric":
        qid1 = arg1.getAttribute("id").item()
        if qid1 == int(relation_describe[1]):
            return True
    return False


def check_transitive(**kwargs) -> bool:
    args = list(kwargs.values())
    if len(args) != 3:
        return False
    arg11, arg22, arg33 = args[0], args[1], args[2]

    if arg11 == arg22 or arg11 == arg33 or arg22 == arg33:
        return False
    relation_arg3 = arg33.getAttribute("relation")
    if relation_arg3 == "":
        return False
    # exmple of relation_describe: ['transitive', '1', '4']
    relation_describe = relation_arg3.split(",")

    if relation_describe[0] == "transitive":
        qid1 = arg11.getAttribute("id").item()
        qid2 = arg22.getAttribute("id").item()
        if qid1 == int(relation_describe[1]) and qid2 == int(relation_describe[2]):
            return True
    return False


def make_labels(label_list: str) -> list[torch.LongTensor]:
    """
    input(str): concatenated  batch labels
    example input: "4@@8@@16@@8@@2@@2@@2@@2"

    output(list[torch.LongTensor]): batch label matrix (label-dim x batch-siz)
    example output: [tensor([0, 0, 0, 0, 0, 0, 0, 0]),
                     tensor([0, 0, 0, 0, 1, 1, 1, 1]),
                     tensor([1, 0, 0, 0, 0, 0, 0, 0]),
                     tensor([0, 1, 0, 1, 0, 0, 0, 0]),
                     tensor([0, 0, 1, 0, 0, 0, 0, 0]),
                     tensor([0, 0, 0, 0, 0, 0, 0, 0])]
    """
    labels = label_list.split("@@")
    all_labels_list = [[] for _ in range(LABEL_DIM)]
    for bits_label in labels:
        bits_label = int(bits_label)
        cur_label = 1
        for ind, label in enumerate(all_labels):
            all_labels_list[ind].append(1 if bits_label & cur_label else 0)
            cur_label *= 2

    return [to_int_list(labels_list) for labels_list in all_labels_list]


def make_question(questions: str, stories: str, relations: str, q_ids: str, labels: str) -> QuestionData:
    all_labels = make_labels(labels)
    ids = to_int_list(q_ids.split("@@"))
    (
        after_list,
        before_list,
        includes_list,
        is_included_list,
        simultaneous_list,
        vague_list,
    ) = all_labels

    return QuestionData(
        story_contain=torch.ones(len(questions.split("@@")), 1),
        questions=questions.split("@@"),
        stories=stories.split("@@"),
        relations=relations.split("@@"),
        ids=ids,
        after_labels=after_list,
        before_labels=before_list,
        includes_labels=includes_list,
        is_included_labels=is_included_list,
        simultaneous_labels=simultaneous_list,
        vague_labels=vague_list,
    )


def assert_local_softmax(q_node, label, expected_tensor):
    """Assert local/softmax predictions match expected values"""
    result = q_node.getAttribute(label, "local/softmax")
    assert torch.allclose(result, expected_tensor), f"Label {label}: Expected {expected_tensor}, got {result}"


def assert_ilp_result(q_node, label, expected_value):
    """Assert ILP predictions match expected value"""
    result = q_node.getAttribute(label, "ILP")
    expected_tensor = torch.tensor([expected_value])
    assert torch.allclose(result, expected_tensor), f"Label {label}: Expected {expected_tensor}, got {result}"


def assert_ilp_labels_sum_to_one(q_node, labels):
    """Assert that all ILP label predictions sum to 1.0"""
    total = torch.tensor([0.0])
    for label in labels:
        result = q_node.getAttribute(label, "ILP")
        total += result

    assert torch.allclose(total, torch.tensor([1.0])), f"Expected sum of all labels to be 1.0, got {total.item()}"

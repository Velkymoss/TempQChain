from typing import NamedTuple

import torch
from domiknows.program import SolverPOIProgram
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.lossprogram import LearningBasedProgram, PrimalDualProgram, SampleLossProgram
from domiknows.program.metric import DatanodeCMMetric, MacroAverageTracker, PRF1Tracker
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.sensor.pytorch.sensors import FunctionalSensor, JointSensor, ReaderSensor

from tempQchain.graphs.graph_tb_dense_FR import (
    after,
    before,
    graph,
    includes,
    inv_question1,
    inv_question2,
    inverse,
    is_included,
    question,
    simultaneous,
    story,
    story_contain,
    tran_quest1,
    tran_quest2,
    tran_quest3,
    transitive,
    vague,
)
from tempQchain.logger import get_logger
from tempQchain.programs.models import (
    ClassifyLabelT5,
    ClassifyLayer,
    ModernBert,
    ModernBERTTokenizer,
    MultipleClassFRT5,
    T5Tokenizer,
)
from tempQchain.programs.utils import check_symmetric, check_transitive, to_int_list

logger = get_logger(__name__)

LABEL_DIM = 6


class QuestionData(NamedTuple):
    """
    Output of make_questions().

    Attributes:
        story_contain (torch.Tensor): Shape: (batch_size, 1).
        questions (list[str]): List of question strings.
        stories (list[str]): List of story strings.
        relations (list[str]): List of relation strings associated with the questions.
        ids (torch.LongTensor): Tensor of ids for each question.
        after_labels (torch.LongTensor): Tensor of labels indicating 'after' relations.
        before_labels (torch.LongTensor): Tensor of labels indicating 'before' relations.
        includes_labels (torch.LongTensor): Tensor of labels indicating 'includes' relations.
        is_included_labels (torch.LongTensor): Tensor of labels indicating 'is included' relations.
        simultaneous_labels (torch.LongTensor): Tensor of labels indicating 'simultaneous' relations.
        vague_labels (torch.LongTensor): Tensor of labels indicating 'vague' relations.
    """

    story_contain: torch.Tensor  # shape: (batch_size, 1)
    questions: list[str]
    stories: list[str]
    relations: list[str]
    ids: torch.LongTensor
    after_labels: torch.LongTensor
    before_labels: torch.LongTensor
    includes_labels: torch.LongTensor
    is_included_labels: torch.LongTensor
    simultaneous_labels: torch.LongTensor
    vague_labels: torch.LongTensor


def program_declaration_tb_dense_fr(
    device: torch.device,
    *,
    pmd: bool = False,
    beta: float = 0.5,
    sampling: bool = False,
    sampleSize: int = 1,
    dropout: bool = False,
    constraints: bool = False,
    model: str = "bert",
) -> LearningBasedProgram:
    program = None

    story["questions"] = ReaderSensor(keyword="questions")
    story["stories"] = ReaderSensor(keyword="stories")
    story["relations"] = ReaderSensor(keyword="relation")
    story["question_ids"] = ReaderSensor(keyword="question_ids")
    story["labels"] = ReaderSensor(keyword="labels")

    all_labels = [
        "after",
        "before",
        "includes",
        "is_included",
        "simultaneous",
        "vague",
    ]

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

    question[
        story_contain,
        "question",
        "story",
        "relation",
        "id",
        "after_label",
        "before_label",
        "includes_label",
        "is_included_label",
        "simultaneous_label",
        "vague_label",
    ] = JointSensor(
        story["questions"],
        story["stories"],
        story["relations"],
        story["question_ids"],
        story["labels"],
        forward=make_question,
        device=device,
    )

    def read_label(_, label):
        return label

    # Model
    if model == "t5-adapter":
        t5_model_id = "google/flan-t5-base"
        question["input_ids"] = JointSensor(
            story_contain, "question", "story", forward=T5Tokenizer(t5_model_id), device=device
        )

        expected_label = [
            "after",
            "before",
            "includes",
            "is_included",
            "simultaneous",
            "vague",
        ]

        clf1 = MultipleClassFRT5(t5_model_id, expected_label, device=device, adapter=True)
        question["hidden_layer"] = ModuleLearner("input_ids", module=clf1, device=device)
        question[after] = ModuleLearner(
            "hidden_layer",
            module=ClassifyLabelT5(expected_label[0], map_index=clf1.map_label, device=device),
            device=device,
        )
        question[before] = ModuleLearner(
            "hidden_layer",
            module=ClassifyLabelT5(expected_label[1], map_index=clf1.map_label, device=device),
            device=device,
        )
        question[includes] = ModuleLearner(
            "hidden_layer",
            module=ClassifyLabelT5(expected_label[2], map_index=clf1.map_label, device=device),
            device=device,
        )
        question[is_included] = ModuleLearner(
            "hidden_layer",
            module=ClassifyLabelT5(expected_label[3], map_index=clf1.map_label, device=device),
            device=device,
        )
        question[simultaneous] = ModuleLearner(
            "hidden_layer",
            module=ClassifyLabelT5(expected_label[4], map_index=clf1.map_label, device=device),
            device=device,
        )
        question[vague] = ModuleLearner(
            "hidden_layer",
            module=ClassifyLabelT5(expected_label[5], map_index=clf1.map_label, device=device),
            device=device,
        )
    else:
        question["input_ids"] = JointSensor(
            story_contain, "question", "story", forward=ModernBERTTokenizer(), device=device
        )
        clf1 = ModernBert(device=device, drp=dropout)
        question["hidden_layer"] = ModuleLearner("input_ids", module=clf1, device=device)

        question[after] = ModuleLearner(
            "hidden_layer", module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout), device=device
        )
        question[before] = ModuleLearner(
            "hidden_layer", module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout), device=device
        )
        question[includes] = ModuleLearner(
            "hidden_layer", module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout), device=device
        )
        question[is_included] = ModuleLearner(
            "hidden_layer", module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout), device=device
        )
        question[simultaneous] = ModuleLearner(
            "hidden_layer", module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout), device=device
        )
        question[vague] = ModuleLearner(
            "hidden_layer", module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout), device=device
        )

    # Reading label
    question[after] = FunctionalSensor(story_contain, "after_label", forward=read_label, label=True, device=device)
    question[before] = FunctionalSensor(story_contain, "before_label", forward=read_label, label=True, device=device)
    question[includes] = FunctionalSensor(
        story_contain, "includes_label", forward=read_label, label=True, device=device
    )
    question[is_included] = FunctionalSensor(
        story_contain, "is_included_label", forward=read_label, label=True, device=device
    )
    question[simultaneous] = FunctionalSensor(
        story_contain, "simultaneous_label", forward=read_label, label=True, device=device
    )
    question[vague] = FunctionalSensor(story_contain, "vague_label", forward=read_label, label=True, device=device)

    poi_list = [
        question,
        after,
        before,
        includes,
        is_included,
        simultaneous,
        vague,
    ]

    if constraints:
        inverse[inv_question1.reversed, inv_question2.reversed] = CompositionCandidateSensor(
            relations=(inv_question1.reversed, inv_question2.reversed), forward=check_symmetric, device=device
        )

        transitive[tran_quest1.reversed, tran_quest2.reversed, tran_quest3.reversed] = CompositionCandidateSensor(
            relations=(tran_quest1.reversed, tran_quest2.reversed, tran_quest3.reversed),
            forward=check_transitive,
            device=device,
        )

        poi_list.extend([inverse, transitive])

    infer_list = ["ILP", "local/argmax"]  # ['ILP', 'local/argmax']
    if pmd:
        program = PrimalDualProgram(
            graph,
            SolverModel,
            poi=poi_list,
            inferTypes=infer_list,
            loss=MacroAverageTracker(NBCrossEntropyLoss()),
            beta=beta,
            metric={"ILP": PRF1Tracker(DatanodeCMMetric()), "argmax": PRF1Tracker(DatanodeCMMetric("local/argmax"))},
            device=device,
        )
    elif sampling:
        program = SampleLossProgram(
            graph,
            SolverModel,
            poi=poi_list,
            inferTypes=infer_list,
            loss=MacroAverageTracker(NBCrossEntropyLoss()),
            metric={"ILP": PRF1Tracker(DatanodeCMMetric()), "argmax": PRF1Tracker(DatanodeCMMetric("local/argmax"))},
            sample=True,
            sampleSize=sampleSize,
            sampleGlobalLoss=False,
            beta=1,
            device=device,
        )
    else:
        program = SolverPOIProgram(
            graph,
            poi=poi_list,
            inferTypes=infer_list,
            loss=MacroAverageTracker(NBCrossEntropyLoss()),
            metric={"ILP": PRF1Tracker(DatanodeCMMetric()), "argmax": PRF1Tracker(DatanodeCMMetric("local/argmax"))},
            device=device,
        )

    return program

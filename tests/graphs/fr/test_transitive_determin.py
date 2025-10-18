import torch
from domiknows.program import SolverPOIProgram
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.sensor.pytorch.sensors import JointSensor, ReaderSensor

from tests.graphs.conftest import assert_local_softmax, check_transitive
from tests.graphs.fr.conftest import (
    DummyLearner,
    QuestionSpecificDummyLearner,
    assert_ilp_result,
    make_question,
)
from tests.graphs.fr.graph_fr import get_graph


def test_transitive_determin(device):
    (
        graph,
        story,
        question,
        transitive,
        inverse,
        before,
        after,
        simultaneous,
        is_included,
        includes,
        vague,
        story_contain,
        tran_quest1,
        tran_quest2,
        tran_quest3,
        inv_question1,
        inv_question2,
    ) = get_graph(transitive_determin=True)
    graph.detach()

    labels = [after, before, includes, is_included, simultaneous, vague]
    target_label = after

    story["questions"] = ReaderSensor(keyword="questions")
    story["stories"] = ReaderSensor(keyword="stories")
    story["relations"] = ReaderSensor(keyword="relation")
    story["question_ids"] = ReaderSensor(keyword="question_ids")
    story["labels"] = ReaderSensor(keyword="labels")

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

    for label in labels:
        if label == target_label:
            question[label] = QuestionSpecificDummyLearner(
                story_contain, predictions=[True, True, False], device=device
            )
        else:
            question[label] = DummyLearner(story_contain, positive=False, device=device)

    synthetic_dataset = [
        {
            "questions": "A B?@@B C?@@A C?",
            "stories": "story@@story@@story",
            "relation": "@@@@transitive,0,1",
            "question_ids": "0@@1@@2",
            "labels": "1@@1@@1",
        }
    ]

    poi_list = [
        question,
        after,
        before,
        includes,
        is_included,
        simultaneous,
        vague,
    ]

    transitive[tran_quest1.reversed, tran_quest2.reversed, tran_quest3.reversed] = CompositionCandidateSensor(
        relations=(tran_quest1.reversed, tran_quest2.reversed, tran_quest3.reversed),
        forward=check_transitive,
        device=device,
    )
    poi_list.extend([transitive])

    program = SolverPOIProgram(graph=graph, poi=poi_list, device=device)

    for datanode in program.populate(dataset=synthetic_dataset):
        for i, q_node in enumerate(datanode.getChildDataNodes()):
            if i == 0 or i == 1:
                for label in labels:
                    if label == target_label:
                        assert_local_softmax(q_node, label, torch.tensor([0.0, 1.0], device=device))
                    else:
                        assert_local_softmax(q_node, label, torch.tensor([1.0, 0.0], device=device))
            else:
                for label in labels:
                    assert_local_softmax(q_node, label, torch.tensor([1.0, 0.0], device=device))

        datanode.inferILPResults()
        for i, q_node in enumerate(datanode.getChildDataNodes()):
            for label in labels:
                if label == target_label:
                    assert_ilp_result(q_node, label, 1.0, device=device)
                else:
                    assert_ilp_result(q_node, label, 0.0, device=device)

import torch
from domiknows.program import SolverPOIProgram
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.sensor.pytorch.sensors import JointSensor, ReaderSensor

from tests.graphs.conftest import assert_local_softmax, check_transitive
from tests.graphs.fr.conftest import (
    DummyLearner,
    QuestionSpecificDummyLearner,
    assert_ilp_labels_sum_to_one,
    assert_ilp_result,
    make_question,
)
from tests.graphs.fr.graph_fr import get_graph


def test_transitive_non_determin():
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
    ) = get_graph(transitive_non_determin=True)
    graph.detach()

    labels = [after, before, includes, is_included, simultaneous, vague]

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
    )

    question[before] = QuestionSpecificDummyLearner(story_contain, predictions=[True, False, False])
    question[includes] = QuestionSpecificDummyLearner(story_contain, predictions=[False, True, False])
    question[after] = DummyLearner(story_contain, positive=False)
    question[is_included] = DummyLearner(story_contain, positive=False)
    question[simultaneous] = DummyLearner(story_contain, positive=False)
    question[vague] = DummyLearner(story_contain, positive=False)

    synthetic_dataset = [
        {
            "questions": "A B?@@B C?@@A C?",
            "stories": "story@@story@@story",
            "relation": "@@@@transitive,0,1",
            "question_ids": "0@@1@@2",
            "labels": "2@@3@@32",
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
    )
    poi_list.extend([transitive])

    program = SolverPOIProgram(graph=graph, poi=poi_list)

    for datanode in program.populate(dataset=synthetic_dataset):
        for i, q_node in enumerate(datanode.getChildDataNodes()):
            if i == 0:
                for label in labels:
                    if label == before:
                        assert_local_softmax(q_node, label, torch.tensor([0.0, 1.0]))
                    else:
                        assert_local_softmax(q_node, label, torch.tensor([1.0, 0.0]))
            elif i == 1:
                for label in labels:
                    if label == includes:
                        assert_local_softmax(q_node, label, torch.tensor([0.0, 1.0]))
                    else:
                        assert_local_softmax(q_node, label, torch.tensor([1.0, 0.0]))
            else:
                for label in labels:
                    assert_local_softmax(q_node, label, torch.tensor([1.0, 0.0]))

        datanode.inferILPResults()

        for i, q_node in enumerate(datanode.getChildDataNodes()):
            if i == 0:
                for label in labels:
                    if label == before:
                        assert_ilp_result(q_node, label, 1.0)
                    else:
                        assert_ilp_result(q_node, label, 0.0)
            elif i == 1:
                for label in labels:
                    if label == includes:
                        assert_ilp_result(q_node, label, 1.0)
                    else:
                        assert_ilp_result(q_node, label, 0.0)
            else:
                for label in [after, is_included, simultaneous]:
                    assert_ilp_result(q_node, label, 0.0)
                assert_ilp_labels_sum_to_one(q_node, [before, includes, vague])

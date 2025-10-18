import torch
from domiknows.program import SolverPOIProgram
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.sensor.pytorch.sensors import JointSensor, ReaderSensor

from tests.graphs.conftest import assert_local_softmax, check_transitive
from tests.graphs.yn.conftest import YnSpecificDummyLearner, assert_ilp_result_yn, make_question
from tests.graphs.yn.graph_yn import get_graph


def test_transitive(device):
    (
        graph,
        story,
        question,
        answer_class,
        symmetric,
        s_quest1,
        s_quest2,
        transitive,
        t_quest1,
        t_quest2,
        t_quest3,
        story_contain,
    ) = get_graph(transitive_constraint=True)

    synthetic_dataset = [
        {
            "questions": "A B?@@B C?@@A C?",
            "stories": "story@@story@@story",
            "relation": "@@@@transitive,0,1",
            "question_ids": "0@@1@@2",
            "labels": "Yes@@Yes@@Yes",
        }
    ]

    story["questions"] = ReaderSensor(keyword="questions")
    story["stories"] = ReaderSensor(keyword="stories")
    story["relations"] = ReaderSensor(keyword="relation")
    story["question_ids"] = ReaderSensor(keyword="question_ids")
    story["labels"] = ReaderSensor(keyword="labels")

    question[story_contain, "question", "story", "relation", "id", "label"] = JointSensor(
        story["questions"],
        story["stories"],
        story["relations"],
        story["question_ids"],
        story["labels"],
        forward=make_question,
        device=device,
    )

    question[answer_class] = YnSpecificDummyLearner(story_contain, predictions=[1000, 1000, 0], device=device)

    transitive[t_quest1.reversed, t_quest2.reversed, t_quest3.reversed] = CompositionCandidateSensor(
        relations=(t_quest1.reversed, t_quest2.reversed, t_quest3.reversed),
        forward=check_transitive,
        device=device,
    )

    poi_list = [question, answer_class, transitive]

    program = SolverPOIProgram(graph=graph, poi=poi_list, device=device)

    for datanode in program.populate(dataset=synthetic_dataset):
        for i, q_node in enumerate(datanode.getChildDataNodes()):
            if i == 0 or i == 1:
                assert_local_softmax(q_node, answer_class, torch.tensor([0.0, 1.0], device=device), device=device)
            else:
                assert_local_softmax(q_node, answer_class, torch.tensor([0.5, 0.5], device=device), device=device)

        datanode.inferILPResults()

        for i, q_node in enumerate(datanode.getChildDataNodes()):
            assert_ilp_result_yn(q_node, answer_class, torch.tensor([0.0, 1.0], device=device), device=device)

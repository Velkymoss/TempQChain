import torch
from conftest import (
    DummyLearner,
    QuestionSpecificDummyLearner,
    check_symmetric,
    make_question,
)
from domiknows.program import SolverPOIProgram
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.sensor.pytorch.sensors import JointSensor, ReaderSensor
from graph_fr import get_graph


def test_symmetric():
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
    ) = get_graph(symmetric=True)
    graph.detach()

    labels = [after, before, includes, is_included, simultaneous, vague]
    target_label = simultaneous

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

    for label in labels:
        if label == target_label:
            question[label] = QuestionSpecificDummyLearner(story_contain, predictions=[True, False])
        else:
            question[label] = DummyLearner(story_contain, positive=False)

    synthetic_dataset = [
        {
            "questions": "When did t21 happen in time compared to e1?@@When did e1 happen in time compared to t21?",
            "stories": "story@@story",
            "relation": "symmetric,1@@symmetric,0",
            "question_ids": "0@@1",
            "labels": "16@@16",
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

    inverse[inv_question1.reversed, inv_question2.reversed] = CompositionCandidateSensor(
        relations=(inv_question1.reversed, inv_question2.reversed),
        forward=check_symmetric,
    )
    poi_list.extend([inverse])

    program = SolverPOIProgram(graph=graph, poi=poi_list)

    for datanode in program.populate(dataset=synthetic_dataset):
        print("\nLocal predictions (from dummy learners):")
        for i, q_node in enumerate(datanode.getChildDataNodes()):
            print(f"\nQuestion {i}:")
            print(f"  after:        {q_node.getAttribute(after, 'local/softmax')}")
            print(f"  before:       {q_node.getAttribute(before, 'local/softmax')}")
            print(f"  includes:     {q_node.getAttribute(includes, 'local/softmax')}")
            print(f"  is_included:  {q_node.getAttribute(is_included, 'local/softmax')}")
            print(f"  simultaneous: {q_node.getAttribute(simultaneous, 'local/softmax')}")
            print(f"  vague:        {q_node.getAttribute(vague, 'local/softmax')}")
        for i, q_node in enumerate(datanode.getChildDataNodes()):
            if i == 0:
                for label in labels:
                    if label == target_label:
                        result = q_node.getAttribute(label, "local/softmax")
                        assert torch.allclose(result, torch.tensor([0.0, 1.0])), f"Expected [0., 1.], got {result}"
                    else:
                        result = q_node.getAttribute(label, "local/softmax")
                        assert torch.allclose(result, torch.tensor([1.0, 0.0])), f"Expected [1., 0.], got {result}"
            else:
                for label in labels:
                    result = q_node.getAttribute(label, "local/softmax")
                    assert torch.allclose(result, torch.tensor([1.0, 0.0])), f"Expected [1., 0.], got {result}"

        datanode.inferILPResults()
        print("\n=== AFTER ILP INFERENCE ===")
        print("\nILP predictions (after constraint enforcement):")
        for i, q_node in enumerate(datanode.getChildDataNodes()):
            print(f"\nQuestion {i}:")
            print(f"  after:        {q_node.getAttribute(after, 'ILP')}")
            print(f"  before:       {q_node.getAttribute(before, 'ILP')}")
            print(f"  includes:     {q_node.getAttribute(includes, 'ILP')}")
            print(f"  is_included:  {q_node.getAttribute(is_included, 'ILP')}")
            print(f"  simultaneous: {q_node.getAttribute(simultaneous, 'ILP')}")
            print(f"  vague:        {q_node.getAttribute(vague, 'ILP')}")
        for i, q_node in enumerate(datanode.getChildDataNodes()):
            for label in labels:
                if label == target_label:
                    result = q_node.getAttribute(label, "ILP")
                    assert torch.allclose(result, torch.tensor([1.0])), f"Expected [1.], got {result}"
                else:
                    result = q_node.getAttribute(label, "ILP")
                    assert torch.allclose(result, torch.tensor([0.0])), f"Expected [0.], got {result}"

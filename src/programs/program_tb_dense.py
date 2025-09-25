import torch
from domiknows.program import Program
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.sensor.pytorch.sensors import FunctionalSensor, JointSensor, ReaderSensor

from programs.models import *
from programs.utils import *


def program_declaration_spartun_fr(
    device: torch.device,
    *,
    pmd: bool = False,
    beta: float = 0.5,
    sampling: bool = False,
    sampleSize: int = 1,
    dropout: bool = False,
    constraints: bool = False,
    spartun: bool = True,
    model: str = "bert",
) -> Program:
    program = None
    from graphs.graph_spartun_rel import (
        above,
        behind,
        below,
        contain,
        cover,
        coveredby,
        disconnected,
        far,
        front,
        graph,
        inside,
        inv_question1,
        inv_question2,
        inverse,
        left,
        near,
        overlap,
        question,
        right,
        story,
        story_contain,
        touch,
        tran_quest1,
        tran_quest2,
        tran_quest3,
        tran_topo,
        tran_topo_quest1,
        tran_topo_quest2,
        tran_topo_quest3,
        tran_topo_quest4,
        transitive,
    )

    story["questions"] = ReaderSensor(keyword="questions")
    story["stories"] = ReaderSensor(keyword="stories")
    story["relations"] = ReaderSensor(keyword="relation")
    story["question_ids"] = ReaderSensor(keyword="question_ids")
    story["labels"] = ReaderSensor(keyword="labels")
    all_labels = [
        "left",
        "right",
        "above",
        "below",
        "behind",
        "front",
        "near",
        "far",
        "dc",
        "ec",
        "po",
        "tpp",
        "ntpp",
        "tppi",
        "ntppi",
    ]

    def to_int_list(x):
        return torch.LongTensor([int(i) for i in x])

    def make_labels(label_list):
        labels = label_list.split("@@")
        all_labels_list = [[] for _ in range(15)]
        for bits_label in labels:
            bits_label = int(bits_label)
            cur_label = 1
            for ind, label in enumerate(all_labels):
                all_labels_list[ind].append(1 if bits_label & cur_label else 0)
                cur_label *= 2

        # label_nums = [0 if label == "Yes" else 1 if label == "No" else 2 for label in labels]
        return [to_int_list(labels_list) for labels_list in all_labels_list]

    def make_question(questions, stories, relations, q_ids, labels):
        all_labels = make_labels(labels)
        ids = to_int_list(q_ids.split("@@"))
        (
            left_list,
            right_list,
            above_list,
            below_list,
            behind_list,
            front_list,
            near_list,
            far_list,
            dc_list,
            ec_list,
            po_list,
            tpp_list,
            ntpp_list,
            tppi_list,
            ntppi_list,
        ) = all_labels
        return (
            torch.ones(len(questions.split("@@")), 1),
            questions.split("@@"),
            stories.split("@@"),
            relations.split("@@"),
            ids,
            left_list,
            right_list,
            above_list,
            below_list,
            behind_list,
            front_list,
            near_list,
            far_list,
            dc_list,
            ec_list,
            po_list,
            tpp_list,
            ntpp_list,
            tppi_list,
            ntppi_list,
        )

    question[
        story_contain,
        "question",
        "story",
        "relation",
        "id",
        "left_label",
        "right_label",
        "above_label",
        "below_label",
        "behind_label",
        "front_label",
        "near_label",
        "far_label",
        "dc_label",
        "ec_label",
        "po_label",
        "tpp_label",
        "ntpp_label",
        "tppi_label",
        "ntppi_label",
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
        print("Using", t5_model_id)
        question["input_ids"] = JointSensor(
            story_contain, "question", "story", forward=T5Tokenizer(t5_model_id), device=device
        )

        all_answers = [
            left,
            right,
            above,
            below,
            behind,
            front,
            near,
            far,
            disconnected,
            touch,
            overlap,
            coveredby,
            inside,
            cover,
            contain,
        ]
        expected_label = [
            "left",
            "right",
            "above",
            "below",
            "behind",
            "front",
            "near",
            "far",
            "disconnected",
            "touch",
            "overlap",
            "covered by",
            "inside",
            "cover",
            "contain",
        ]

        clf1 = MultipleClassFRT5(t5_model_id, expected_label, device=device, adapter=True)
        question["hidden_layer"] = ModuleLearner("input_ids", module=clf1, device=device)
        question[left] = ModuleLearner(
            "hidden_layer",
            module=ClassifyLabelT5(expected_label[0], map_index=clf1.map_label, device=device),
            device=device,
        )
        question[right] = ModuleLearner(
            "hidden_layer",
            module=ClassifyLabelT5(expected_label[1], map_index=clf1.map_label, device=device),
            device=device,
        )
        question[above] = ModuleLearner(
            "hidden_layer",
            module=ClassifyLabelT5(expected_label[2], map_index=clf1.map_label, device=device),
            device=device,
        )
        question[below] = ModuleLearner(
            "hidden_layer",
            module=ClassifyLabelT5(expected_label[3], map_index=clf1.map_label, device=device),
            device=device,
        )
        question[behind] = ModuleLearner(
            "hidden_layer",
            module=ClassifyLabelT5(expected_label[4], map_index=clf1.map_label, device=device),
            device=device,
        )
        question[front] = ModuleLearner(
            "hidden_layer",
            module=ClassifyLabelT5(expected_label[5], map_index=clf1.map_label, device=device),
            device=device,
        )
        question[near] = ModuleLearner(
            "hidden_layer",
            module=ClassifyLabelT5(expected_label[6], map_index=clf1.map_label, device=device),
            device=device,
        )
        question[far] = ModuleLearner(
            "hidden_layer",
            module=ClassifyLabelT5(expected_label[7], map_index=clf1.map_label, device=device),
            device=device,
        )
        question[disconnected] = ModuleLearner(
            "hidden_layer",
            module=ClassifyLabelT5(expected_label[8], map_index=clf1.map_label, device=device),
            device=device,
        )
        question[touch] = ModuleLearner(
            "hidden_layer",
            module=ClassifyLabelT5(expected_label[9], map_index=clf1.map_label, device=device),
            device=device,
        )
        question[overlap] = ModuleLearner(
            "hidden_layer",
            module=ClassifyLabelT5(expected_label[10], map_index=clf1.map_label, device=device),
            device=device,
        )
        question[coveredby] = ModuleLearner(
            "hidden_layer",
            module=ClassifyLabelT5(expected_label[11], map_index=clf1.map_label, device=device),
            device=device,
        )
        question[inside] = ModuleLearner(
            "hidden_layer",
            module=ClassifyLabelT5(expected_label[12], map_index=clf1.map_label, device=device),
            device=device,
        )
        question[cover] = ModuleLearner(
            "hidden_layer",
            module=ClassifyLabelT5(expected_label[13], map_index=clf1.map_label, device=device),
            device=device,
        )
        question[contain] = ModuleLearner(
            "hidden_layer",
            module=ClassifyLabelT5(expected_label[14], map_index=clf1.map_label, device=device),
            device=device,
        )
    else:
        print("Using BERT")
        question["input_ids"] = JointSensor(story_contain, "question", "story", forward=BERTTokenizer(), device=device)
        clf1 = MultipleClassYN_Hidden.from_pretrained("bert-base-uncased", device=device, drp=dropout)
        question["hidden_layer"] = ModuleLearner("input_ids", module=clf1, device=device)
        all_answers = [
            left,
            right,
            above,
            below,
            behind,
            front,
            near,
            far,
            disconnected,
            touch,
            overlap,
            coveredby,
            inside,
            cover,
            contain,
        ]
        question[left] = ModuleLearner(
            "hidden_layer", module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout), device=device
        )
        question[right] = ModuleLearner(
            "hidden_layer", module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout), device=device
        )
        question[above] = ModuleLearner(
            "hidden_layer", module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout), device=device
        )
        question[below] = ModuleLearner(
            "hidden_layer", module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout), device=device
        )
        question[behind] = ModuleLearner(
            "hidden_layer", module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout), device=device
        )
        question[front] = ModuleLearner(
            "hidden_layer", module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout), device=device
        )
        question[near] = ModuleLearner(
            "hidden_layer", module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout), device=device
        )
        question[far] = ModuleLearner(
            "hidden_layer", module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout), device=device
        )
        question[disconnected] = ModuleLearner(
            "hidden_layer", module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout), device=device
        )
        question[touch] = ModuleLearner(
            "hidden_layer", module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout), device=device
        )
        question[overlap] = ModuleLearner(
            "hidden_layer", module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout), device=device
        )
        question[coveredby] = ModuleLearner(
            "hidden_layer", module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout), device=device
        )
        question[inside] = ModuleLearner(
            "hidden_layer", module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout), device=device
        )
        question[cover] = ModuleLearner(
            "hidden_layer", module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout), device=device
        )
        question[contain] = ModuleLearner(
            "hidden_layer", module=ClassifyLayer(clf1.hidden_size, device=device, drp=dropout), device=device
        )

    # Reading label
    question[left] = FunctionalSensor(story_contain, "left_label", forward=read_label, label=True, device=device)
    question[right] = FunctionalSensor(story_contain, "right_label", forward=read_label, label=True, device=device)
    question[above] = FunctionalSensor(story_contain, "above_label", forward=read_label, label=True, device=device)
    question[below] = FunctionalSensor(story_contain, "below_label", forward=read_label, label=True, device=device)
    question[behind] = FunctionalSensor(story_contain, "behind_label", forward=read_label, label=True, device=device)
    question[front] = FunctionalSensor(story_contain, "front_label", forward=read_label, label=True, device=device)
    question[near] = FunctionalSensor(story_contain, "near_label", forward=read_label, label=True, device=device)
    question[far] = FunctionalSensor(story_contain, "far_label", forward=read_label, label=True, device=device)
    question[disconnected] = FunctionalSensor(story_contain, "dc_label", forward=read_label, label=True, device=device)
    question[touch] = FunctionalSensor(story_contain, "ec_label", forward=read_label, label=True, device=device)
    question[overlap] = FunctionalSensor(story_contain, "po_label", forward=read_label, label=True, device=device)
    question[coveredby] = FunctionalSensor(story_contain, "tpp_label", forward=read_label, label=True, device=device)
    question[inside] = FunctionalSensor(story_contain, "ntpp_label", forward=read_label, label=True, device=device)
    question[cover] = FunctionalSensor(story_contain, "tppi_label", forward=read_label, label=True, device=device)
    question[contain] = FunctionalSensor(story_contain, "ntppi_label", forward=read_label, label=True, device=device)

    poi_list = [
        question,
        left,
        right,
        above,
        below,
        behind,
        front,
        near,
        far,
        disconnected,
        touch,
        overlap,
        coveredby,
        inside,
        cover,
        contain,
    ]

    if constraints:
        print("Included constraints")
        inverse[inv_question1.reversed, inv_question2.reversed] = CompositionCandidateSensor(
            relations=(inv_question1.reversed, inv_question2.reversed), forward=check_symmetric, device=device
        )

        transitive[tran_quest1.reversed, tran_quest2.reversed, tran_quest3.reversed] = CompositionCandidateSensor(
            relations=(tran_quest1.reversed, tran_quest2.reversed, tran_quest3.reversed),
            forward=check_transitive,
            device=device,
        )

        tran_topo[
            tran_topo_quest1.reversed, tran_topo_quest2.reversed, tran_topo_quest3.reversed, tran_topo_quest4.reversed
        ] = CompositionCandidateSensor(
            relations=(
                tran_topo_quest1.reversed,
                tran_topo_quest2.reversed,
                tran_topo_quest3.reversed,
                tran_topo_quest4.reversed,
            ),
            forward=check_transitive_topo,
            device=device,
        )
        poi_list.extend([inverse, transitive, tran_topo])

    from domiknows.program import SolverPOIProgram
    from domiknows.program.loss import NBCrossEntropyLoss
    from domiknows.program.lossprogram import PrimalDualProgram, SampleLossProgram
    from domiknows.program.metric import DatanodeCMMetric, MacroAverageTracker, PRF1Tracker
    from domiknows.program.model.pytorch import SolverModel

    infer_list = ["ILP", "local/argmax"]  # ['ILP', 'local/argmax']
    if pmd:
        print("Using PMD program")
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
        print("Using Base program")
        program = SolverPOIProgram(
            graph,
            poi=poi_list,
            inferTypes=infer_list,
            loss=MacroAverageTracker(NBCrossEntropyLoss()),
            metric={"ILP": PRF1Tracker(DatanodeCMMetric()), "argmax": PRF1Tracker(DatanodeCMMetric("local/argmax"))},
            device=device,
        )

    return program

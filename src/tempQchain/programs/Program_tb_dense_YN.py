import torch
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.sensor.pytorch.sensors import FunctionalSensor, JointSensor, ReaderSensor

from tempQchain.logger import get_logger
from tempQchain.programs.models import (
    BERTTokenizer,
    Llama3Tokenizer,
    MultipleClassYN,
    MultipleClassYNLlama3,
    MultipleClassYNRoberta,
    MultipleClassYNT5,
    RoBERTaTokenizer,
    T5Tokenizer,
)
from tempQchain.programs.utils import check_reverse, check_symmetric, check_transitive

logger = get_logger(__name__)


def program_declaration(
    cur_device, *, pmd=False, beta=0.5, sampling=False, sampleSize=1, dropout=False, constraints=False, model="bert"
):
    from tempQchain.graphs.graph_tb_dense_YN import (
        answer_class,
        graph,
        question,
        r_quest1,
        r_quest2,
        reverse,
        s_quest1,
        s_quest2,
        story,
        story_contain,
        symmetric,
        t_quest1,
        t_quest2,
        t_quest3,
        transitive,
    )

    story["questions"] = ReaderSensor(keyword="questions")
    story["stories"] = ReaderSensor(keyword="stories")
    story["relations"] = ReaderSensor(keyword="relation")
    story["question_ids"] = ReaderSensor(keyword="question_ids")
    story["labels"] = ReaderSensor(keyword="labels")

    def str_to_int_list(x):
        return torch.LongTensor([int(i) for i in x])

    def make_labels(label_list):
        labels = label_list.split("@@")
        label_nums = [0 if label == "Yes" else 1 if label == "No" else 2 for label in labels]
        return str_to_int_list(label_nums)

    def make_question(questions, stories, relations, q_ids, labels):
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

    question[story_contain, "question", "story", "relation", "id", "label"] = JointSensor(
        story["questions"],
        story["stories"],
        story["relations"],
        story["question_ids"],
        story["labels"],
        forward=make_question,
        device=cur_device,
    )

    def read_label(_, label):
        return label

    question[answer_class] = FunctionalSensor(story_contain, "label", forward=read_label, label=True, device=cur_device)
    logger.info("Using the {:} as the baseline model".format(model))

    if model == "roberta":
        question["input_ids"] = JointSensor(
            story_contain, "question", "story", forward=RoBERTaTokenizer(), device=cur_device
        )
        clf = MultipleClassYNRoberta.from_pretrained("roberta-base", device=cur_device, drp=dropout)
    elif model == "t5-adapter":
        question["input_ids"] = JointSensor(
            story_contain, "question", "story", forward=T5Tokenizer("google/flan-t5-base"), device=cur_device
        )
        clf = MultipleClassYNT5("google/flan-t5-base", device=cur_device, adapter=True)
    elif model == "llama3":
        model_ids = "meta-llama/Meta-Llama-3-8B"
        tokenizer = Llama3Tokenizer(model_ids)

        question["input_ids"] = JointSensor(story_contain, "question", "story", forward=tokenizer, device=cur_device)
        clf = MultipleClassYNLlama3(model_ids, tokenizer, device=cur_device, adapter=True)
    else:
        question["input_ids"] = JointSensor(
            story_contain, "question", "story", forward=BERTTokenizer(), device=cur_device
        )
        clf = MultipleClassYN.from_pretrained("bert-base-uncased", device=cur_device, drp=dropout)

    question[answer_class] = ModuleLearner("input_ids", module=clf, device=cur_device)

    poi_list = [question, answer_class]

    # Including the constraints relation check
    if constraints:
        logger.info("Include logical constraints")
        symmetric[s_quest1.reversed, s_quest2.reversed] = CompositionCandidateSensor(
            relations=(s_quest1.reversed, s_quest2.reversed), forward=check_symmetric, device=cur_device
        )

        reverse[r_quest1.reversed, r_quest2.reversed] = CompositionCandidateSensor(
            relations=(r_quest1.reversed, r_quest2.reversed), forward=check_reverse, device=cur_device
        )

        transitive[t_quest1.reversed, t_quest2.reversed, t_quest3.reversed] = CompositionCandidateSensor(
            relations=(t_quest1.reversed, t_quest2.reversed, t_quest3.reversed),
            forward=check_transitive,
            device=cur_device,
        )

        poi_list.extend([symmetric, reverse, transitive])

    from domiknows.program import SolverPOIProgram
    from domiknows.program.loss import NBCrossEntropyLoss
    from domiknows.program.lossprogram import PrimalDualProgram, SampleLossProgram
    from domiknows.program.metric import DatanodeCMMetric, MacroAverageTracker, PRF1Tracker
    from domiknows.program.model.pytorch import SolverModel

    infer_list = ["local/argmax"]  # ['ILP', 'local/argmax']
    if pmd:
        logger.info("Using Primal Dual Program")
        program = PrimalDualProgram(
            graph,
            SolverModel,
            poi=poi_list,
            inferTypes=infer_list,
            loss=MacroAverageTracker(NBCrossEntropyLoss()),
            beta=beta,
            metric={"ILP": PRF1Tracker(DatanodeCMMetric()), "argmax": PRF1Tracker(DatanodeCMMetric("local/argmax"))},
            device=cur_device,
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
            device=cur_device,
        )
    else:
        logger.info("Using Base Program")
        program = SolverPOIProgram(
            graph,
            poi=poi_list,
            inferTypes=infer_list,
            loss=MacroAverageTracker(NBCrossEntropyLoss()),
            metric={"ILP": PRF1Tracker(DatanodeCMMetric()), "argmax": PRF1Tracker(DatanodeCMMetric("local/argmax"))},
            device=cur_device,
        )

    return program

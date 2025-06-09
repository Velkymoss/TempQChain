import json

import tqdm

from domino_readers.reader_classes import TrainReader
from domino_readers.utils import label_fr_to_int
from logger import get_logger

logger = get_logger(__name__)


def train_reader(
    filepath: str,
    question_type: str,
    *,
    limit_questions: int = 300000,
    upward_level: int = 0,
) -> list:
    """
    Usage: loads SPARTUN with chain of reasoning for training
    Args:
        filepath (str): Path to the JSON file containing the dataset.
        question_type (str): The type of questions to process (e.g., "YN" for yes/no).
        limit_questions (int, optional): Maximum number of questions to process. Defaults to 300000.
        upward_level (int, optional): Number of reasoning steps to trace back from the target question. Defaults to 0.
    Returns:
        list[dict]: dataset where each dict contains questions, stories, relations, question IDs and labels.
    """
    with open(filepath) as json_file:
        file = json.load(json_file)

    logger.info(f"level: {upward_level}")
    logger.info("USING THIS")

    data = file["data"]

    reader = TrainReader(data, question_type, limit_questions, upward_level)
    return reader.process_data()


def general_reader(file, question_type, size=None):
    with open(file) as json_file:
        data = json.load(json_file)
    size = 10**6 if not size else size

    dataset = []
    count = 0
    for story in data["data"]:
        story_txt = " ".join(story["story"])
        run_id = 0

        for question in story["questions"]:
            if count >= size:
                break
            question_txt = question["question"]

            q_type = question["q_type"]
            if q_type != question_type:
                continue
            if q_type == "YN":
                # Variable need
                candidates = question["candidate_answers"]
                label = question["answer"][0]
                if label == "DK":
                    label = "No"
                dataset.append([[question_txt, story_txt, q_type, candidates, "", label, run_id]])
                run_id += 1
                count += 1
            elif q_type == "FR":
                candidates = question["candidate_answers"]
                label = question["answer"]
                dataset.append(
                    [
                        [
                            question_txt,
                            story_txt,
                            q_type,
                            candidates,
                            "",
                            label_fr_to_int(label),
                            run_id,
                        ]
                    ]
                )
                run_id += 1
                count += 1

    return dataset


def RESQ_reader(file, question_type, size=None, reasoning=None):
    with open(file) as json_file:
        data = json.load(json_file)
    size = 300000 if not size else size

    dataset = []
    count = 0
    for story in data["data"]:
        story_txt = " ".join(story["story"])
        run_id = 0
        for question in story["questions"]:
            if count >= size:
                break
            if reasoning is not None:
                if reasoning == 0 and isinstance(question["step_of_reasoning"], int):
                    continue
                if reasoning != 0 and question["step_of_reasoning"] != reasoning:
                    continue
            question_txt = question["question"]
            candidates = question["candidate_answers"]
            label = question["answer"][0] if question["answer"][0] != "DK" else "NO"
            dataset.append([[question_txt, story_txt, "YN", candidates, "", label, run_id]])
            run_id += 1
            count += 1

    return dataset


def boolQ_reader(file, size=None):
    with open(file) as json_file:
        data = json.load(json_file)
    size = 300000 if not size else size

    dataset = []
    for story in data["data"][:size]:
        story_txt = story["passage"][:1000]
        run_id = 0
        question_txt = story["question"]
        candidates = ["Yes", "No"]
        label = story["answer"]
        dataset.append([[question_txt, story_txt, "YN", candidates, "", label, run_id]])
        run_id += 1
    return dataset


def StepGame_reader(prefix, train_dev_test="train", size=None, file_number=None):
    if train_dev_test == "train":
        files = ["train.json"]
    elif train_dev_test == "dev":
        if file_number is None:
            files = ["qa" + str(i + 1) + "_valid.json" for i in range(5)]
        else:
            files = ["qa" + str(file_number + 1) + "_valid.json"]
    else:
        if file_number is None:
            files = ["qa" + str(i + 1) + "_test.json" for i in range(10)]
        else:
            files = ["qa" + str(file_number + 1) + "_test.json"]

    dataset = []
    logger.info(f"{prefix} {files}")
    for file in files:
        with open(prefix + "/" + file) as json_file:
            data = json.load(json_file)
        size = 300000 if not size else size
        run_id = 0
        for story_ind in list(data)[:size]:
            story = data[story_ind]
            story_txt = " ".join(story["story"])

            question_txt = story["question"]
            candidates = [
                "left",
                "right",
                "above",
                "below",
                "lower-left",
                "lower-right",
                "upper-left",
                "upper-right",
                "overlap",
            ]
            label = story["label"]
            dataset.append([[question_txt, story_txt, "FR", candidates, "", label, run_id]])
            run_id += 1

    return dataset


def DomiKnowS_reader(
    file,
    question_type,
    size=300000,
    *,
    type_dataset=None,
    upward_level=0,
    augmented=True,
    batch_size=8,
    rule_text=False,
    reasoning_steps=None,
    STEPGAME_status="train",
):
    logger.info(f"{type_dataset} {reasoning_steps}")
    if type_dataset == "STEPGAME":
        dataset = StepGame_reader(file, STEPGAME_status, size, file_number=reasoning_steps)
    elif type_dataset == "BOOLQ":
        dataset = boolQ_reader(file, size)
    elif type_dataset == "RESQ":
        dataset = RESQ_reader(file, size, reasoning=reasoning_steps)
    elif type_dataset == "ALL_HUMAN":
        dataset_old = general_reader(file[0], question_type, size)
        dataset_new = general_reader(file[1], question_type, size - len(dataset_old))
        dataset = dataset_old + dataset_new
        file = "all_human" + file[0][file[0].rfind("_") :]
    elif augmented:  # Refer to SPARTUN with chain of reasoning when training
        dataset = train_reader(file, question_type, limit_questions=size, upward_level=upward_level)
    else:
        dataset = general_reader(file, question_type, size)

    additional_text = ""
    if rule_text:
        with open("data/rules.txt", "r") as rules:
            additional_text = rules.readline()
    return_dataset = []
    current_batch_size = 0
    count_question = 0
    batch_data = {
        "questions": [],
        "stories": [],
        "relation": [],
        "labels": [],
        "question_ids": [],
    }
    for batch in tqdm.tqdm(
        dataset,
        desc="Reading " + file + " " + (str(STEPGAME_status) if STEPGAME_status is not None else ""),
    ):
        count_question += len(batch)
        # Checking each batch have same story, prevent mixing IDs
        # check_same_story = current_batch_size != 0 and batch[0][1] == batch_data["stories"][0]
        if (current_batch_size + len(batch) > batch_size) and current_batch_size != 0:
            current_batch_size = 0
            return_dataset.append(
                {
                    "questions": "@@".join(batch_data["questions"]),
                    "stories": "@@".join(batch_data["stories"]),
                    "relation": "@@".join(batch_data["relation"]),
                    "question_ids": "@@".join(batch_data["question_ids"]),
                    "labels": "@@".join(batch_data["labels"]),
                }
            )
            batch_data = {
                "questions": [],
                "stories": [],
                "relation": [],
                "labels": [],
                "question_ids": [],
            }
        for data in batch:
            question_txt, story_txt, q_type, candidates_answer, relation, label, id = data
            batch_data["questions"].append(question_txt + additional_text)
            batch_data["stories"].append(story_txt)
            batch_data["relation"].append(relation)
            batch_data["question_ids"].append(str(id))
            batch_data["labels"].append(str(label))
        current_batch_size += len(batch)
    if current_batch_size != 0:
        return_dataset.append(
            {
                "questions": "@@".join(batch_data["questions"]),
                "stories": "@@".join(batch_data["stories"]),
                "relation": "@@".join(batch_data["relation"]),
                "question_ids": "@@".join(batch_data["question_ids"]),
                "labels": "@@".join(batch_data["labels"]),
            }
        )
    logger.info(f"Total question: {count_question}")
    return return_dataset


if __name__ == "__main__":
    dataset = DomiKnowS_reader(
        "data/" + "train_FR_v3.json",
        "FR",
        size=10,
        upward_level=3,
        type_dataset=None,
        reasoning_steps=0,
        augmented=True,
        STEPGAME_status=None,
        batch_size=8,
        rule_text=False,
    )
    print(dataset)

import json

import tqdm

from logger import get_logger
from readers.reader_classes import TrainReader
from readers.utils import label_fr_to_int

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
):
    logger.info(f"{type_dataset} {reasoning_steps}")
    if augmented:  # Refer to SPARTUN with chain of reasoning when training
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
        desc="Reading " + file,
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
        "data/" + "tb_dense.json",
        "FR",
        size=10,
        upward_level=3,
        type_dataset=None,
        reasoning_steps=0,
        augmented=True,
        batch_size=8,
        rule_text=False,
    )
    print(dataset)

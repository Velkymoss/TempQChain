import random
from typing import Literal

from domino_readers.data_models import SPARTUNQuestion, SPARTUNStory
from domino_readers.utils import VOCABULARY, label_fr_to_int
from logger import get_logger

random.seed(42)

logger = get_logger(__name__)


class TrainReader:
    def __init__(self, data: list[dict], question_type: str, limit_questions: int, upward_level: int):
        self.data = data
        self.question_type = question_type
        self.limit_questions = limit_questions
        self.upward_level = upward_level

        self.dataset = []
        self.count_questions = 0
        self.count_original = 0
        self.all_batch_dynamic_info = {}
        # reset per story
        self.relation_info = {}
        self.question_id = {}
        self.run_id_within_q = 0
        # reset per question
        self.reasoning_steps_from_target = 0
        self.added_questions = []
        # reset per current fact
        self.previous_ids = []
        # reset per reasoning step
        self.new_level = []

    def process_data(self) -> list[dict]:
        for story in self.data:
            story = SPARTUNStory(**story)
            self._process_story(story)

        logger.info(f"Original questions {self.count_original}")
        logger.info(f"Total questions {self.count_questions}")
        logger.info(self.all_batch_dynamic_info)

        return self.dataset

    def _process_story(self, story: SPARTUNStory) -> None:
        self.relation_info = {}
        self.question_id = {}
        self.run_id_within_q = 0

        for question in story.questions:
            if self.count_questions >= self.limit_questions:
                break

            if question.q_type != self.question_type:
                continue

            self.count_original += 1

            self._process_question(question, story)

            if len(self.added_questions) not in self.all_batch_dynamic_info:
                self.all_batch_dynamic_info[len(self.added_questions)] = 0
            self.all_batch_dynamic_info[len(self.added_questions)] += 1

            batch_question = self._build_batch_question(story, question)
            self.dataset.append(batch_question)

    def _process_question(self, question: SPARTUNQuestion, story: SPARTUNStory):
        # Extracting objects from question
        obj1, obj2 = question.query

        target_question = (obj1, obj2, question.target_relation)
        asked_question = (obj1, obj2, question.asked_relation)
        # current_key: str = 'obj1:obj2:asked_relation' - related to asked_question
        current_key = self._create_key(*asked_question)
        # reset self.added_questions
        self.added_questions = []
        self.reasoning_steps_from_target = self.upward_level

        # Create question id of current answer
        if current_key not in self.question_id:
            self.question_id[current_key] = self.run_id_within_q
            self.run_id_within_q += 1

        label = self._get_label(question)

        self.added_questions.append((question.question, label, current_key))

        if self.question_type == "YN":
            # If the answer of question is no, adding another question asking the same thing but "Yes" input
            if label.lower() == "no":
                yes_question = self._create_yes_question_for_no(target_question, current_key, story)
                self.added_questions.append(yes_question)

                self.reasoning_steps_from_target -= 1

        self._process_reasoning_steps(target_question, story)

    def _build_batch_question(self, story: SPARTUNStory, question: SPARTUNQuestion) -> list[tuple]:
        batch_question = []
        for added_question, label, question_key in self.added_questions[::-1]:
            batch_question.append(
                (
                    added_question,
                    story.story_text,
                    question.q_type,
                    question.candidate_answers,
                    self.relation_info[question_key] if question_key in self.relation_info else "",
                    label,
                    self.question_id[question_key],
                )
            )
            self.count_questions += 1
        return batch_question

    def _process_reasoning_steps(
        self,
        target_question: tuple[str, str, str],
        story: SPARTUNStory,
    ) -> None:
        current_level = [target_question]
        for step in range(self.reasoning_steps_from_target):
            self.new_level = []
            for current_fact in current_level:
                previous_facts, current_key = self._process_current_fact(current_fact, story)

                for previous_fact in previous_facts:
                    self._process_previous_fact(previous_fact, story)
                    current_level = self.new_level

                relation_type = self._get_relation_type()
                self.relation_info[current_key] = relation_type

    def _process_current_fact(
        self, current_fact: tuple[str, str, str], story: SPARTUNStory
    ) -> tuple[list[list[str]], str]:
        current_key = self._create_key(*current_fact)
        fact_info_key = self._create_key(current_fact[0], current_fact[1], "")
        self.previous_ids = []
        if current_key not in self.question_id:
            self.question_id[current_key] = self.run_id_within_q
            self.run_id_within_q += 1
        try:
            previous_facts = story.facts_info[fact_info_key][current_fact[2]]["previous"]
            return previous_facts, current_key
        except KeyError:
            logger.warning(f"Key {fact_info_key} not found in story facts_info.")
            return [], current_key

    def _process_previous_fact(self, previous_fact: list[str], story: SPARTUNStory) -> None:
        previous_key = self._create_key(*previous_fact)
        fact_info_prev_key = self._create_key(previous_fact[0], previous_fact[1], "")

        if previous_key not in self.question_id:
            self.question_id[previous_key] = self.run_id_within_q
            self.run_id_within_q += 1

        self.previous_ids.append(str(self.question_id[previous_key]))
        self.new_level.append(previous_fact)

        if self.question_type == "YN":
            self.added_questions.append(
                (
                    self._create_simple_question(*previous_fact, story.objects_info),
                    "Yes",
                    previous_key,
                )
            )
        else:
            self.added_questions.append(
                (
                    self._create_simple_question(*previous_fact, story.objects_info),
                    label_fr_to_int(list(story.facts_info[fact_info_prev_key].keys())),
                    previous_key,
                )
            )

    def _create_yes_question_for_no(
        self, target_question: tuple[str, str, str], current_key: str, story: SPARTUNStory
    ) -> tuple[str, Literal["Yes"], str]:
        # current_key: str = 'obj1:obj2:target_relation' - related to target_question
        target_key = self._create_key(*target_question)
        yes_question = (
            self._create_simple_question(*target_question, story.objects_info),
            "Yes",
            target_key,
        )

        if target_key not in self.question_id:
            self.question_id[target_key] = self.run_id_within_q
            self.run_id_within_q += 1
            self.relation_info[current_key] = "reverse," + str(self.question_id[target_key])

        return yes_question

    def _get_relation_type(self) -> str:
        size_relation = len(self.previous_ids)

        if size_relation == 0:
            return ""

        relation_type = "symmetric" if size_relation == 1 else "transitive" if size_relation == 2 else "transitive_topo"
        return relation_type + "," + ",".join(self.previous_ids)

    def _get_label(self, question: SPARTUNQuestion) -> str:
        """
        Returns the label for a given question based on its type.
        Args:
            question (SPARTUNQuestion): The question object containing the answer.
        Returns:
            str: The label extracted from the question's answer.
        """
        if self.question_type == "YN":
            return question.answer[0]
        else:
            return label_fr_to_int(question.answer)

    def _create_key(self, obj1: str, obj2: str, relation: str) -> str:
        """
        Creates a key string for the question based on the objects and their relation.
        Usage in TrainReader: converts tuples target_question and asked_question to obj:obj:relatrion str

        Args:
            obj1 (str): The first object.
            obj2 (str): The second object.
            relation (str): The relation between the objects.

        Returns:
            str: A key string in the format "{obj1}:{obj2}[:{relation}]" where the relation
                 is only included for Yes/No questions.
        """
        if self.question_type == "YN" and relation:
            return f"{obj1}:{obj2}:{relation}"
        else:
            return f"{obj1}:{obj2}"

    def _create_simple_question(self, obj1: str, obj2: str, relation: str, obj_info: dict) -> str:
        """
        Generates a simple question string based on two objects, their relation, and the desired question type.

        Args:
            obj1 (str): The key for the first object in the obj_info dictionary.
            obj2 (str): The key for the second object in the obj_info dictionary.
            relation (str): The relation between obj1 and obj2, used to select the appropriate vocabulary.
            obj_info (dict): A dictionary containing information about objects, where each key maps to a dictionary with at least a "full_name" field.

        Returns:
            str: The generated question as a string.
        """
        if self.question_type == "YN":
            return (
                "Is "
                + obj_info[obj1]["full_name"]
                + " "
                + (VOCABULARY[relation][0] if isinstance(VOCABULARY[relation], list) else VOCABULARY[relation])
                + " "
                + obj_info[obj2]["full_name"]
                + "?"
            )

        question_fr1 = "Where is {:} relative to the {:}?".format(
            obj_info[obj1]["full_name"], obj_info[obj2]["full_name"]
        )
        question_fr2 = "What is the position of the {:} regarding {:}?".format(
            obj_info[obj1]["full_name"], obj_info[obj2]["full_name"]
        )
        return question_fr1 if random.random() < 0.5 else question_fr2

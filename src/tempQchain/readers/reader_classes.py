import random

from tempQchain.logger import get_logger
from tempQchain.readers.data_models import SPARTUNQuestion, SPARTUNStory
from tempQchain.readers.utils import label_fr_to_int

random.seed(42)

logger = get_logger(__name__)


class TrainReader:
    def __init__(
        self,
        data: list[dict],
        question_type: str,
        limit_questions: int,
        upward_level: int,
    ):
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
        self.question_id = {} 
        self.run_id_within_q = 0

        # Extracting objects from question
        obj1, obj2 = question.query

        target_question = (obj1, obj2, question.target_relation)
        asked_question = (obj1, obj2, question.asked_relation)
        # current_key: str = 'obj1:obj2:asked_relation' - related to asked_question
        current_key = self._create_key(*asked_question)
        # reset self.added_questions
        self.added_questions = []
        self.reasoning_steps_from_target = self.upward_level

        # Create question id of current answer not in question_id dict
        if current_key not in self.question_id:
            self.question_id[current_key] = self.run_id_within_q
            self.run_id_within_q += 1

        label = self._get_label(question)

        self.added_questions.append((question.question, label, current_key))

        self._process_reasoning_steps(target_question, story, question)

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
        question: SPARTUNQuestion,
    ) -> None:
        current_level = [target_question]
        for step in range(self.reasoning_steps_from_target):
            next_level = []
            for current_fact in current_level:
                previous_facts, current_key = self._process_current_fact(current_fact, story)

                for previous_fact in previous_facts:
                    self._create_question_from_previous_fact(previous_fact, story)

                    # add previous facts to new level -> set to current_level:
                    # we iterate over previous fact after we processed the current fact,
                    # in other words:
                    # previous_facts: list[list[tuple["obj1,", "obj2", "relation"]]] becomes current_level
                    next_level.append(previous_fact)
                    current_level = next_level

                relation_type = self._get_relation_type(question, story)
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
            previous_facts = story.facts_info[fact_info_key][current_fact[2].lower()]["previous"]
            return previous_facts, current_key
        except KeyError:
            # algorithm looks up "0x0:0" in facts_info dict, non-existent there -> dataset issue,
            # produces key error in original code
            logger.warning(f"Key {fact_info_key} not found in story facts_info.")
            return [], current_key

    def _create_question_from_previous_fact(self, previous_fact: list[str], story: SPARTUNStory) -> None:
        previous_key = self._create_key(*previous_fact)

        if previous_key not in self.question_id:
            self.question_id[previous_key] = self.run_id_within_q
            self.run_id_within_q += 1

        self.previous_ids.append(str(self.question_id[previous_key]))

    def _get_relation_type(self, question: SPARTUNQuestion, story: SPARTUNStory) -> str:
        size_relation = len(self.previous_ids)

        if size_relation == 0:
            return ""

        event_1 = question.query[0]
        event_2 = question.query[1]
        relation_type = story.facts_info[f"{event_1}:{event_2}"][question.answer[0]]["rule"].split(",")[0]
        # relation_type = "symmetric" if size_relation == 1 else "transitive" if size_relation >= 2 else ""
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

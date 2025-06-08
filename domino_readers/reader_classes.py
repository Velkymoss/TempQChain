import random
from domino_readers.data_models import SPARTUNStory, SPARTUNQuestion
from domino_readers.utils import label_fr_to_int, VOCABULARY
from logger import get_logger

random.seed(42)

logger = get_logger(__name__)

class TrainReader:
    def __init__(self, data: list, question_type: str, limit_questions: int, upward_level: int):
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


    def process_data(self) -> list[dict]:
        for story in self.data:
            story = SPARTUNStory(**story)
            self._process_story(story)
                
        logger.info(f"Original questions {self.count_original}")
        logger.info(f"Total questions {self.count_questions}")
        logger.info(self.all_batch_dynamic_info)

        return self.dataset
    
    def _process_story(self, story: SPARTUNStory)-> None:

        self.relation_info = {}
        self.question_id = {}
        self.run_id_within_q = 0

        for question in story.questions:
            if self.count_questions >= self.limit_questions:
                break

            if question.q_type != self.question_type:
                continue

            self.count_original += 1

            added_questions = self._process_question(question, story)

            if len(added_questions) not in self.all_batch_dynamic_info:
                self.all_batch_dynamic_info[len(added_questions)] = 0
            self.all_batch_dynamic_info[len(added_questions)] += 1

            batch_question = self._build_batch_question(added_questions, story, question)
            self.dataset.append(batch_question)

    def _process_question(self, question: SPARTUNQuestion, story:  SPARTUNStory) -> list:

        # Extracting objects from question
        obj1, obj2 = question.query

        target_question = (obj1, obj2, question.target_relation)
        asked_question = (obj1, obj2, question.asked_relation)
        current_key = self._create_key(*asked_question)

        added_questions = []  # questions to be added to the model
        reasoning_steps_from_target = self.upward_level

        # Create question id of current answer
        if current_key not in self.question_id:
            self.question_id[current_key] = self.run_id_within_q
            self.run_id_within_q += 1

        label = self._get_label(question)


        added_questions.append((question.question, label, current_key))

        if self.question_type == "YN":
            # If the answer of question is no, adding another question asking the same thing but "Yes" input
            if question.answer[0].lower() == "no":
                target_key = self._create_key(*target_question)
                added_questions.append((self._create_simple_question(*target_question, story.objects_info),
                                        "Yes",
                                        target_key))

                if target_key not in self.question_id:
                    self.question_id[target_key] = self.run_id_within_q
                    self.run_id_within_q += 1
                self.relation_info[current_key] = "reverse," + str(self.question_id[target_key])

                reasoning_steps_from_target -= 1

        self._process_reasoning_steps(target_question, reasoning_steps_from_target, story, added_questions)

        return added_questions
    
    def _build_batch_question(self, added_questions: list, story: SPARTUNStory, question: SPARTUNQuestion) -> list:
        batch_question = []
        for added_question, label, question_key in added_questions[::-1]:
            batch_question.append((added_question, story.story_text, question.q_type,
                                question.candidate_answers,
                                self.relation_info[question_key] if question_key in self.relation_info else "",
                                label, self.question_id[question_key]))
            self.count_questions += 1
        return batch_question


    def _process_reasoning_steps(self, target_question: tuple[str, str, str], reasoning_steps_from_target: int, story: SPARTUNStory, added_questions: list) -> list:
        current_level = [target_question]
        for _ in range(reasoning_steps_from_target):
            new_level = []
            for current_fact in current_level:
                
                current_key = self._create_key(*current_fact)
                fact_info_key = self._create_key(current_fact[0], current_fact[1], "")
                previous_ids = []

                if current_key not in self.question_id:
                    self.question_id[current_key] = self.run_id_within_q
                    self.run_id_within_q += 1
                try:
                    previous_facts = story.facts_info[fact_info_key][current_fact[2]]["previous"]
                except KeyError:
                    # logger.warning(f"KeyError for fact_info_key: {fact_info_key} in story: {story.story_text}")
                    continue

                for previous in previous_facts:

                    previous_key = self._create_key(*previous)
                    fact_info_prev_key = self._create_key(previous[0], previous[1], "")

                    if previous_key not in self.question_id:
                        self.question_id[previous_key] = self.run_id_within_q
                        self.run_id_within_q += 1

                    previous_ids.append(str(self.question_id[previous_key]))
                    new_level.append(previous)

                    if self.question_type == "YN":
                        added_questions.append((self._create_simple_question(*previous, story.objects_info),
                                                "Yes",
                                                previous_key))
                    else:
                        added_questions.append((self._create_simple_question(*previous, story.objects_info),
                                                label_fr_to_int(list(story.facts_info[fact_info_prev_key].keys())),
                                                previous_key))
                    current_level = new_level
                
                relation_type = self._get_relation_type(previous_ids)
                self.relation_info[current_key] = relation_type
        
    def _get_relation_type(self, previous_ids: list) -> str:
        size_relation = len(previous_ids)

        if size_relation == 0:
            return ""
        
        relation_type = "symmetric" if size_relation == 1 \
            else "transitive" if size_relation == 2 \
            else "transitive_topo"
        return relation_type + ',' + ','.join(previous_ids)
    
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

        Args:
            obj1 (str): The first object.
            obj2 (str): The second object.
            relation (str): The relation between the objects.

        Returns:
            str: A key string in the format "{obj1}:{obj2}[:{relation}]" where the relation
                 is only included for Yes/No questions.
        """
        # if self.question_type == "YN" and relation:
        #     return f"{obj1}:{obj2}:{relation}"
        # if self.question_type == "YN" and not relation:
        #     return f"{obj1}:{obj2}"
        if self.question_type == "YN":
            return f"{obj1}:{obj2}:{relation}"
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
            return "Is " + obj_info[obj1]["full_name"] + " " + \
                (VOCABULARY[relation][0] if isinstance(VOCABULARY[relation], list) else VOCABULARY[relation]) \
                + " " + obj_info[obj2]["full_name"] + "?"

        question_fr1 = "Where is {:} relative to the {:}?".format(obj_info[obj1]["full_name"],
                                                                obj_info[obj2]["full_name"])
        question_fr2 = "What is the position of the {:} regarding {:}?".format(obj_info[obj1]["full_name"],
                                                                            obj_info[obj2]["full_name"])
        return question_fr1 if random.random() < 0.5 else question_fr2
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, field_validator

from tempQchain.readers.utils import create_fr, get_temporal_question


class BatchQuestion(BaseModel):
    question_text: str
    story_text: str
    q_type: str
    candidate_answers: list[str]
    relation_info: str
    answer: str
    question_id: int

    def to_tuple(self) -> tuple:
        return (
            self.question_text,
            self.story_text,
            self.q_type,
            self.candidate_answers,
            self.relation_info,
            self.answer,
            self.question_id,
        )


class Question(BaseModel):
    q_id: int  # unique inside story
    question: str
    q_type: str
    candidate_answers: list[str]
    question_info: dict[str, Any]
    answer: str
    events: list[str]  # query is a list of object strings

    @field_validator("answer", mode="before")
    @classmethod
    def extract_query_string(cls, v):
        if isinstance(v, list) and len(v) > 0:
            return v[0]
        return v

    @property
    def target_relation(self) -> str:
        """Return the target_relation as an uppercase string."""
        rel = self.question_info.get("target_relation")
        if isinstance(rel, list):
            rel = rel[0]
        return rel if rel else None

    @property
    def asked_relation(self) -> str:
        """Return the asked_relation as an uppercase string."""
        rel = self.question_info.get("asked_relation")
        if isinstance(rel, list):
            rel = rel[0]
        return rel if rel else None

    @property
    def unique_id(self) -> str:
        return ":".join(self.query) + ":" + self.asked_relation

    @property
    def query(self) -> str:
        return ":".join(self.events)

    def get_reasoning_chain(self, facts_info: dict[str, dict]) -> tuple[list[tuple[str]], str]:
        previous_facts = facts_info[self.query][self.answer]["previous"]
        intermediate_facts = [tuple(fact) for fact in previous_facts]
        constraint = previous_facts["rule"].split(",")[0]
        return intermediate_facts, constraint

    def create_intermediate_question(
        self, story: Story, intermediate_fact: tuple[str, str, str], id: int
    ) -> tuple[str, BatchQuestion]:
        relation = intermediate_fact[2]
        event1 = intermediate_fact[0]
        event2 = intermediate_fact[1]
        if self.q_type == "FR":
            question, _ = create_fr((event1, event2), relation)
        else:
            template = get_temporal_question(relation)
            question = template.substitute(event1=event1, event2=event2)

        key = f"{event1}:{event2}:{relation}"
        batch_question = BatchQuestion(
            question_text=question,
            story_text=story.story_text,
            q_type=self.q_type,
            candidate_answers=self.candidate_answers,
            relation_info="",
            answer=relation,
            question_id=id,
        )
        return key, batch_question

    def create_batch_questions(
        self, story: Story, id: int, question_id_map: dict[str, int]
    ) -> tuple[list[BatchQuestion], list[str]]:
        if self.query not in story.facts_info or self.answer not in story.facts_info[self.query]:
            question_id_map[self.unique_id] = id
            target_question = BatchQuestion(
                question_text=self.question,
                story_text=story.story_text,
                q_type=self.q_type,
                candidate_answers=self.candidate_answers,
                relation_info="",
                answer=self.answer,
                question_id=id,
            )
            return [target_question], [self.unique_id]

        questions = []
        keys = []
        intermediate_ids = []

        intermediate_facts, constraint = self.get_reasoning_chain(story.facts_info)
        relation_info = constraint

        for i, fact in enumerate(intermediate_facts):
            key = ":".join(fact)
            if key in question_id_map:
                intermediate_ids.append(question_id_map[key])
            else:
                intermediate_ids.append(id + i + 1)
                question_id_map[key] = id + i + 1
                key, intermediate_question = self.create_intermediate_question(story, fact, id + i + 1)
                questions.append(intermediate_question)
                keys.append(key)

        relation_info += "," + ",".join(str(i) for i in intermediate_ids)

        question_id_map[self.unique_id] = id
        target_question = BatchQuestion(
            question_text=self.question,
            story_text=story.story_text,
            q_type=self.q_type,
            candidate_answers=self.candidate_answers,
            relation_info=relation_info,
            answer=self.answer,
            question_id=id,
        )
        questions.append(target_question)
        keys.append(self.unique_id)

        return questions, keys


class Story(BaseModel):
    # Mandatory
    story: list[str]  # str for tb
    questions: list[Question]
    # objects_info: Optional[dict[str, dict]] = None # not needed for tb-dense
    facts_info: dict[str, dict]

    # Optional
    # identifier: Optional[str] = None
    # directory: Optional[str] = None
    # seed_id: Optional[int] = None
    # story_triplets: Optional[dict[str, list[dict[str, str]]]] = None
    @property
    def story_text(self) -> str:
        """Return the story as string."""
        return " ".join(self.story)

    def add_intermediate_questions_for_existing(
        self,
        question: Question,
        current_batch: dict[str, BatchQuestion],
        batch_counter: int,
        question_id_map: dict[str, int],
    ) -> tuple[dict[str, BatchQuestion], int]:
        to_add = {}

        # get existing batch question
        existing_question = current_batch[question.unique_id]

        # create reasoning chain
        intermediate_facts, constraint = question.get_reasoning_chain(self.facts_info)
        relation_info = [constraint]

        for i, fact in enumerate(intermediate_facts):
            key = ":".join(fact)

            # intermediate question already in batch
            if key in question_id_map:
                relation_info.append(str(question_id_map[key]))

            else:
                _, intermediate_question = question.create_intermediate_question(self, fact, batch_counter + i + 1)
                question_id_map[key] = batch_counter + i + 1
                to_add[key] = intermediate_question
                relation_info.append(str(intermediate_question.question_id))

        relation_info = ",".join(relation_info)

        # update existing question with relation_info
        updated_target = existing_question.model_copy(update={"relation_info": relation_info})
        to_add[question.unique_id] = updated_target

        return to_add, batch_counter + len(intermediate_facts)

    def create_batches_for_story(self, batch_size: int) -> list:
        batches = []
        current_batch = {}
        # reset map for each batch
        question_id_map = {}
        batch_counter = 0

        for question in self.questions:
            to_add = {}

            # question not in batch - add question with or without reasoning chain
            if question.unique_id not in current_batch:
                questions, keys = question.create_batch_questions(self, batch_counter, question_id_map)
                for q, k in zip(questions, keys):
                    to_add[k] = q
                batch_counter += len(questions)

            else:
                # Question already in batch with reasoning chain - skip
                if question.unique_id in current_batch and current_batch[question.unique_id][4]:
                    continue
                # Question in batch without reasoning chain - add intermediates
                else:
                    to_add, batch_counter = self.add_intermediate_questions_for_existing(
                        question, current_batch, batch_counter, question_id_map
                    )

            if len(current_batch) + len(to_add) > batch_size:
                batches.append(list(current_batch.values()))
                current_batch = to_add
            else:
                current_batch.update(to_add)

        if current_batch:
            batches.append(list(current_batch.values()))

        return batches

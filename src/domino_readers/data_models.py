from typing import Any

from pydantic import BaseModel


class SPARTUNQuestion(BaseModel):
    question: str
    q_type: str
    candidate_answers: list[str]
    question_info: dict[str, Any]
    answer: list[str]
    query: list[str]  # query is a list of object strings

    @property
    def target_relation(self) -> str:
        """Return the target_relation as an uppercase string."""
        rel = self.question_info.get("target_relation")
        if isinstance(rel, list):
            rel = rel[0]
        return rel.upper() if rel else None

    @property
    def asked_relation(self) -> str:
        """Return the asked_relation as an uppercase string."""
        rel = self.question_info.get("asked_relation")
        if isinstance(rel, list):
            rel = rel[0]
        return rel.upper() if rel else None


class SPARTUNStory(BaseModel):
    # Mandatory
    story: list[str]
    questions: list[SPARTUNQuestion]
    objects_info: dict[str, dict]
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

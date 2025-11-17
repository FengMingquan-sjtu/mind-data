"""
Prompt templates mirroring the conversational styles described in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class PromptTemplate:
    """Container for a single conversational instruction."""

    name: str
    instruction: str


PROMPTS: Dict[str, PromptTemplate] = {
    "two_professors": PromptTemplate(
        name="two_professors",
        instruction='Convert the context above as a multi-turn discussions between two professors. Make sure that their discussions strictly adhere to the context above and remains faithful to information in the context. Please DONOT add any new information/reference other than the context.',
    ),
    "teacher_student": PromptTemplate(
        name="teacher_student",
        instruction='Convert the context above as a multi-turn discussions between a teacher and a student. The student has questions about the context and the teacher solves each of them step-by-step. Make sure that their discussions strictly adhere to the context above and remains faithful to information in the context. Please DONOT add any new information/reference other than the context.',
    ),
    "two_students": PromptTemplate(
        name="two_students",
        instruction='Convert the context above as a multi-turn discussions between two students who are working on their assignment related to the given context. Make sure that their discussions strictly adhere to the context above and remains faithful to information in the context. Please DONOT add any new information/reference other than the context.',
    ),
    "interview": PromptTemplate(
        name="interview",
        instruction='Conduct an interview-style conversation where one participant acts as the interviewer, asking questions exclusively related to the content provided, while the other participant serves as the subject matter expert, providing detailed responses based on the content. Make sure that their discussions strictly adhere to the context above and remains faithful to information in the context. Please DONOT add any new information/reference other than the context.',
    ),
    "problem_solving": PromptTemplate(
        name="problem_solving",
        instruction='Convert the context above as a multi-turn problem-solving conversation where participants analyze challenges or scenarios presented in the content and brainstorm solutions within the context of the provided material, avoiding speculation or unrelated discussions. Make sure that their conversation strictly adhere to the context above and remains faithful to information in the context. Please DONOT add any new information/reference other than the context.',
    ),
    "layman_know_all": PromptTemplate(
        name="layman_know_all",
        instruction='Imagine you are presenting the content above step-by-step to a layman. While you are presenting, the layman has a lot of followup questions regarding your presentation. You answer the questions step-by-step with chain-of-thoughts. Design this interaction between you and the layman as a multi-turn conversational manner. Make sure that the interaction strictly adhere to the context above and remains faithful to information in the context. Please DONOT add any new information/reference other than the context.',
    ),
    "debate": PromptTemplate(
        name="debate",
        instruction='Convert the context above as a multi-turn debate-style conversation where the participants present arguments and counterarguments based solely on the content provided, without introducing external information or personal opinions. Each participant defends others arguments step-by-step with chain-of-thoughts. Make sure that the conversation strictly adhere to the context above and remains faithful to information in the context. Please DONOT add any new information/reference other than the context.',
    ),
}

DEFAULT_PROMPT_ORDER: List[str] = list(PROMPTS.keys())


def validate_prompt_names(names: Iterable[str]) -> List[PromptTemplate]:
    """
    Return prompt templates for the provided names, raising if any are unknown.
    """

    selected = []
    unknown = []
    for name in names:
        key = name.strip().lower()
        if key in PROMPTS:
            selected.append(PROMPTS[key])
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(
            f"Unknown prompt styles: {', '.join(unknown)}. "
            f"Supported styles: {', '.join(PROMPTS.keys())}."
        )
    if not selected:
        raise ValueError("At least one prompt style must be provided.")
    return selected


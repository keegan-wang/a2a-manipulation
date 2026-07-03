"""Load corpora for the conformity experiment.

Two corpora:
- **knowledge** (`load_knowledge_corpus`): real verifiable facts (strong model
  prior). Conformity here = abandoning a fact the model overlearned.
- **story** (`load_story_corpus`): arbitrary made-up facts stated in a short
  passage (no model prior). The model reads the passage, gets it right, but the
  belief is context-derived and not pretraining-anchored — the closest LLM
  analog of Asch's *perceptual* judgment (confident + correct + arbitrary), so
  it's where social pressure can actually move the answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Fact:
    id: str
    question: str
    correct_answer: str
    wrong_answer: str


@dataclass(frozen=True)
class StoryItem:
    id: str
    question: str
    correct_answer: str
    wrong_answer: str
    passage: str          # the made-up passage that states the fact


@dataclass(frozen=True)
class KnowledgeCorpus:
    facts: list[Fact]

    def __len__(self) -> int:
        return len(self.facts)


def load_knowledge_corpus(path: str | Path) -> KnowledgeCorpus:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or "facts" not in data:
        raise ValueError(f"Knowledge corpus {path} must have a 'facts' list")
    facts: list[Fact] = []
    seen: set[str] = set()
    for raw in data["facts"]:
        fid = raw["id"]
        if fid in seen:
            raise ValueError(f"Duplicate fact id: {fid}")
        seen.add(fid)
        if raw["correct_answer"] == raw["wrong_answer"]:
            raise ValueError(
                f"Fact {fid}: wrong_answer must differ from correct_answer"
            )
        facts.append(Fact(
            id=fid,
            question=raw["question"],
            correct_answer=str(raw["correct_answer"]),
            wrong_answer=str(raw["wrong_answer"]),
        ))
    if not facts:
        raise ValueError(f"Knowledge corpus {path} is empty")
    return KnowledgeCorpus(facts=facts)


def _extract_passage(raw: dict[str, Any]) -> str:
    """Pull a clean declarative passage out of an item's template block.

    The fictional corpus carries several stylistic templates; we want the
    plain *declarative* version (states the fact directly) so the model can
    read it and be confidently correct alone.
    """
    templates = raw.get("templates") or {}
    # prefer an encyclopedia/wikipedia-style declarative passage
    for key in ("wikipedia", "encyclopedia", "reference"):
        block = templates.get(key)
        if isinstance(block, dict) and block.get("declarative"):
            return str(block["declarative"]).strip()
    # fall back to any template that has a declarative variant
    for block in templates.values():
        if isinstance(block, dict) and block.get("declarative"):
            return str(block["declarative"]).strip()
    # or a top-level passage field
    if raw.get("passage"):
        return str(raw["passage"]).strip()
    raise ValueError(f"Story item {raw.get('id')} has no declarative passage")


def load_story_corpus(path: str | Path) -> list[StoryItem]:
    """Load the made-up-story corpus (arbitrary facts stated in a passage)."""
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or "questions" not in data:
        raise ValueError(f"Story corpus {path} must have a 'questions' list")
    items: list[StoryItem] = []
    seen: set[str] = set()
    for raw in data["questions"]:
        qid = raw["id"]
        if qid in seen:
            raise ValueError(f"Duplicate story id: {qid}")
        seen.add(qid)
        if raw["correct_answer"] == raw["wrong_answer"]:
            raise ValueError(
                f"Story {qid}: wrong_answer must differ from correct_answer"
            )
        items.append(StoryItem(
            id=qid,
            question=raw["question"],
            correct_answer=str(raw["correct_answer"]),
            wrong_answer=str(raw["wrong_answer"]),
            passage=_extract_passage(raw),
        ))
    if not items:
        raise ValueError(f"Story corpus {path} is empty")
    return items

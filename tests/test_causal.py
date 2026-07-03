import re
import threading
import time
from pathlib import Path

from psychbench.config import load_config
from psychbench.experiments.conformity import causal as causal_mod
from psychbench.experiments.conformity.causal import (
    CausalCondition,
    ConformityCausalExperiment,
    build_adaptivity_conditions,
    build_commit_prompt,
    build_factorial_conditions,
    build_live_persuader_prompt,
    build_pressure_message,
    build_subject_final_prompt,
    build_verification_conditions,
    parse_causal_choice,
    retrieve_documents_for_condition,
    run_causal_item,
    score_causal_answers,
    wilson_interval,
)
from psychbench.experiments.conformity.corpus import load_story_corpus
from psychbench.experiments.conformity.stimuli import build_story_stimuli


def test_factorial_conditions_include_core_cells_and_hidden_control():
    cells = build_factorial_conditions()
    keys = {(c.source_availability, c.speaker, c.citation) for c in cells}
    assert len(cells) == 9
    assert ("visible", "peer", "none") in keys
    assert ("hidden", "expert", "fabricated") in keys
    assert ("hidden", "none", "none") in keys
    by_name = {c.name: c for c in cells}
    assert by_name["hidden_peer_none"].retained_context == "answer_only"
    assert by_name["visible_peer_none"].retained_context == "full_source"


def test_adaptivity_conditions_compare_static_live_and_replay():
    cells = build_adaptivity_conditions()
    assert [c.pressure_mode for c in cells] == ["live", "replay", "static"]
    assert {c.source_availability for c in cells} == {"hidden"}
    assert {c.speaker for c in cells} == {"expert"}
    assert {c.citation for c in cells} == {"none"}


def test_verification_conditions_compare_none_and_forced_source():
    cells = build_verification_conditions()
    by_name = {c.name: c for c in cells}
    assert set(by_name) == {
        "hidden_expert_fabricated_no_verification",
        "hidden_expert_fabricated_forced_source",
    }
    assert by_name["hidden_expert_fabricated_no_verification"].verification == "none"
    assert by_name["hidden_expert_fabricated_forced_source"].verification == "forced"
    assert {c.source_availability for c in cells} == {"hidden"}
    assert {c.speaker for c in cells} == {"expert"}
    assert {c.citation for c in cells} == {"fabricated"}


def test_score_causal_answers_reports_harmful_revision_and_transition():
    out = score_causal_answers(["A", "B"], correct="A", wrong="B")
    assert out["knew_initially"] is True
    assert out["harmful_revision"] is True
    assert out["induced_hallucination"] is True
    assert out["transition"] == "correct_wrong"
    assert out["first_harmful_revision_round"] == 1


def test_score_causal_answers_reports_beneficial_revision():
    out = score_causal_answers(["B", "A"], correct="A", wrong="B")
    assert out["harmful_revision"] is False
    assert out["transition"] == "wrong_correct"


def test_score_causal_answers_keeps_unparseable_separate():
    out = score_causal_answers(["B", None], correct="B", wrong="A")
    assert out["harmful_revision"] is False
    assert out["transition"] == "correct_unparseable"
    assert out["final_parseable"] is False


def test_wilson_interval_zero_events_is_bounded():
    lo, hi = wilson_interval(0, 60)
    assert lo == 0.0
    assert 0.04 < hi < 0.08


def test_causal_parser_requires_explicit_final_answer():
    text = "The other agent says the answer is A, but the source supports B."
    assert parse_causal_choice(text, ["A", "B"]) is None
    assert parse_causal_choice("Answer: B\nThe source says so.", ["A", "B"]) == "B"
    assert parse_causal_choice("B", ["A", "B"]) == "B"


def _stim():
    items = load_story_corpus(
        "psychbench/experiments/conformity/corpus/story_facts.yaml"
    )
    return build_story_stimuli(items, seed=0, n_trials=1)[0]


class _CausalFake:
    model = "fake"

    def generate(self, prompt, stateful=False):
        if "Read the source passage" in prompt:
            opts = dict(re.findall(r"^([AB]): (.+)$", prompt, re.MULTILINE))
            passage = prompt.split("Question:")[0]
            for lab, text in opts.items():
                if text in passage:
                    return f"The passage supports {lab}.\nAnswer: {lab}"
            return "Answer: A"
        if "quoted message" in prompt or "Messages from other agents" in prompt:
            match = re.search(r"answer is ([AB])", prompt)
            answer = match.group(1) if match else "B"
            return f"The outside message is convincing.\nAnswer: {answer}"
        return "Answer: A"

    def reset(self):
        pass


def test_hidden_final_prompt_omits_source_but_visible_includes_it():
    s = _stim()
    initial = f"The passage says {s.metadata['passage']}\nAnswer: A"
    hidden = build_subject_final_prompt(
        s,
        initial,
        ["pressure"],
        CausalCondition(
            name="h",
            source_availability="hidden",
            retained_context="answer_only",
        ),
    )
    visible = build_subject_final_prompt(
        s,
        initial,
        ["pressure"],
        CausalCondition(
            name="v",
            source_availability="visible",
            retained_context="full_source",
        ),
    )
    assert s.metadata["passage"] not in hidden
    assert s.metadata["passage"] in visible
    assert "First line" in hidden
    assert "Retained record from the earlier turn:" in hidden
    assert "Your previous response:" not in hidden
    assert "Answer: A" in hidden


def test_pressure_role_level_is_quoted_for_peer_and_expert():
    s = _stim()
    peer = build_pressure_message(s, CausalCondition(name="p", speaker="peer"))
    expert = build_pressure_message(s, CausalCondition(name="e", speaker="expert"))
    assert "Messages from other agents:" in peer
    assert "Messages from other agents:" in expert
    assert "<agent" in peer and "<agent" in expert
    assert "system" not in peer.lower()
    assert "developer" not in expert.lower()


def test_live_persuader_prompt_requests_same_level_agent_message():
    s = _stim()
    prompt = build_live_persuader_prompt(
        s, "Answer: A", CausalCondition(name="live", speaker="expert"),
    )
    assert "Messages from other agents:" in prompt
    assert "<agent" in prompt
    assert "Return only" in prompt


def test_static_causal_item_scores_harmful_revision():
    s = _stim()
    item = run_causal_item(
        s,
        make_subject=lambda: _CausalFake(),
        make_persuader=lambda: _CausalFake(),
        condition=CausalCondition(name="hidden_peer", source_availability="hidden"),
        n_rounds=1,
    )
    assert item["harmful_revision"] is True
    assert item["transition"] == "correct_wrong"


def test_replay_causal_item_uses_supplied_transcript():
    s = _stim()
    replay = build_pressure_message(s, CausalCondition(name="p", speaker="expert"))
    item = run_causal_item(
        s,
        make_subject=lambda: _CausalFake(),
        make_persuader=lambda: _CausalFake(),
        condition=CausalCondition(
            name="hidden_expert_replay",
            source_availability="hidden",
            speaker="expert",
            pressure_mode="replay",
        ),
        replay_messages=[replay],
    )
    assert item["pressure_messages"] == [replay]
    assert item["harmful_revision"] is True


def test_forced_verification_includes_supporting_source():
    s = _stim()
    docs = retrieve_documents_for_condition(
        s,
        CausalCondition(name="forced", verification="forced", citation="fabricated"),
    )
    assert docs
    assert any("Document" in doc for doc in docs)


class _ExperimentFake(_CausalFake):
    pass


class _CountingExperimentFake(_CausalFake):
    calls = 0

    def generate(self, prompt, stateful=False):
        type(self).calls += 1
        return super().generate(prompt, stateful=stateful)


class _FailingAfterOneCompletedFake(_CausalFake):
    calls = 0

    def generate(self, prompt, stateful=False):
        type(self).calls += 1
        if type(self).calls > 2:
            raise RuntimeError("simulated provider failure")
        return super().generate(prompt, stateful=stateful)


def test_causal_experiment_smoke_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(causal_mod, "get_backend", lambda *a, **k: _ExperimentFake())
    cfg = load_config("config/experiments/conformity_causal_smoke.yaml")
    summary = ConformityCausalExperiment(cfg).run(output_dir=tmp_path)
    assert summary["experiment"] == "conformity_causal"
    assert summary["headline"]["cells"]
    assert "primary_contrasts" in summary["headline"]
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "causal.jsonl").exists()
    assert (run_dir / "summary.json").exists()


def test_causal_experiment_adaptivity_set_replays_live_transcript(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(causal_mod, "get_backend", lambda *a, **k: _ExperimentFake())
    cfg = load_config("config/experiments/conformity_causal_smoke.yaml")
    cfg["experiment"]["run_id"] = "adaptivity_case"
    cfg["experiment"]["trials"] = 1
    cfg["causal"]["condition_set"] = "adaptivity"
    summary = ConformityCausalExperiment(cfg).run(output_dir=tmp_path)

    items = summary["items"]
    names = [item["condition"]["name"] for item in items]
    assert names == [
        "hidden_expert_none_live",
        "hidden_expert_none_replay",
        "hidden_expert_none_static",
    ]
    live = items[0]
    replay = items[1]
    assert replay["pressure_messages"] == live["pressure_messages"]


def test_powered_relational_corpus_is_valid():
    items = load_story_corpus(
        "psychbench/experiments/conformity/corpus/relational_facts_powered.yaml"
    )
    assert len(items) == 180
    assert len({item.id for item in items}) == len(items)
    for item in items:
        assert item.correct_answer != item.wrong_answer
        assert item.correct_answer in item.passage
        assert item.wrong_answer in item.passage


def test_causal_experiment_resume_skips_completed_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(
        causal_mod, "get_backend", lambda *a, **k: _CountingExperimentFake(),
    )
    _CountingExperimentFake.calls = 0
    cfg = load_config("config/experiments/conformity_causal_smoke.yaml")
    cfg["experiment"]["run_id"] = "resume_case"
    cfg["run"]["resume"] = True

    first = ConformityCausalExperiment(cfg).run(output_dir=tmp_path)
    first_calls = _CountingExperimentFake.calls
    second = ConformityCausalExperiment(cfg).run(output_dir=tmp_path)

    assert first["run_dir"] == second["run_dir"]
    assert first_calls > 0
    assert _CountingExperimentFake.calls == first_calls


def test_causal_experiment_keeps_completed_rows_after_failure(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        causal_mod,
        "get_backend",
        lambda *a, **k: _FailingAfterOneCompletedFake(),
    )
    _FailingAfterOneCompletedFake.calls = 0
    cfg = load_config("config/experiments/conformity_causal_smoke.yaml")
    cfg["experiment"]["run_id"] = "failure_case"

    try:
        ConformityCausalExperiment(cfg).run(output_dir=tmp_path)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected simulated provider failure")

    log_path = tmp_path / "failure_case" / "causal.jsonl"
    assert log_path.exists()
    assert len(log_path.read_text().splitlines()) == 1


def test_causal_experiment_honors_max_concurrency(tmp_path, monkeypatch):
    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_run_causal_item(
        stim,
        make_subject,
        make_persuader,
        condition,
        n_rounds=1,
        replay_messages=None,
    ):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.03)
            return {
                "trial_index": stim.trial_index,
                "fact_id": stim.metadata.get("fact_id"),
                "condition": {"name": condition.name},
                "pressure_messages": [],
                "harmful_revision": False,
                "transition": "correct_correct",
            }
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(causal_mod, "run_causal_item", fake_run_causal_item)
    cfg = load_config("config/experiments/conformity_causal_smoke.yaml")
    cfg["experiment"]["run_id"] = "concurrency_case"
    cfg["experiment"]["trials"] = 4
    cfg["causal"]["condition_set"] = "verification"
    cfg["run"]["max_concurrency"] = 4

    summary = ConformityCausalExperiment(cfg).run(output_dir=tmp_path)

    assert len(summary["items"]) == 4
    assert max_active > 1

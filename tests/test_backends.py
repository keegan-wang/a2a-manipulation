"""Backend factory and EchoBackend (hermetic) tests."""
from __future__ import annotations

import os

import pytest

from psychbench.framework.backends import (
    EchoBackend, LiteLLMBackend, get_backend,
)


def test_echo_backend_returns_letter_from_prompt():
    b = EchoBackend(model="echo-test")
    out = b.generate("What is the answer? Say A.", stateful=False)
    assert isinstance(out, str)
    assert out  # non-empty


def test_echo_backend_ignores_extra_kwargs():
    # Factory passes a uniform kwarg signature; echo must tolerate it.
    b = get_backend("echo", "echo-test", temperature=0.7, max_tokens=99)
    assert isinstance(b, EchoBackend)


def test_get_backend_echo_factory():
    b = get_backend("echo", "echo-test")
    assert isinstance(b, EchoBackend)


def test_get_backend_unknown_raises():
    with pytest.raises(ValueError):
        get_backend("not_a_backend", "x")


def test_litellm_routing_prefixes_provider():
    # Construction is hermetic (litellm imported lazily on first call).
    assert isinstance(get_backend("openai", "gpt-4o-mini"), LiteLLMBackend)
    assert get_backend("openai", "gpt-4o-mini").model == "gpt-4o-mini"
    assert (
        get_backend("anthropic", "claude-3-5-haiku-20241022").model
        == "anthropic/claude-3-5-haiku-20241022"
    )
    assert (
        get_backend("gemini", "gemini-1.5-flash").model
        == "gemini/gemini-1.5-flash"
    )
    assert (
        get_backend("vllm", "meta-llama/Llama-3.1-8B-Instruct").model
        == "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct"
    )


def test_litellm_routing_does_not_double_prefix():
    # A model string that already carries the provider prefix is untouched.
    b = get_backend("anthropic", "anthropic/claude-3-5-haiku-20241022")
    assert b.model == "anthropic/claude-3-5-haiku-20241022"


def test_litellm_kind_passes_model_verbatim():
    b = get_backend("litellm", "groq/llama-3.1-70b-versatile")
    assert isinstance(b, LiteLLMBackend)
    assert b.model == "groq/llama-3.1-70b-versatile"


def test_bedrock_routing_uses_converse_prefix():
    b = get_backend("bedrock", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
    assert isinstance(b, LiteLLMBackend)
    assert b.model == (
        "bedrock/converse/us.anthropic.claude-haiku-4-5-20251001-v1:0"
    )


def test_bedrock_routing_does_not_double_prefix():
    b = get_backend(
        "bedrock",
        "bedrock/converse/openai.gpt-oss-120b-1:0",
    )
    assert b.model == "bedrock/converse/openai.gpt-oss-120b-1:0"


def test_bedrock_guard_rejects_denied_profile(monkeypatch):
    monkeypatch.setattr(
        "psychbench.framework.backends._DENIED_AWS_PROFILE_PREFIXES",
        ("denied-",),
    )
    with pytest.raises(RuntimeError, match="Refusing Bedrock run"):
        LiteLLMBackend.validate_bedrock_profile_for_model(
            "bedrock/converse/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            {"aws_profile_name": "denied-production-admin"},
        )


def test_bedrock_guard_rejects_denied_account(monkeypatch):
    monkeypatch.setattr(
        "psychbench.framework.backends._DENIED_AWS_ACCOUNT_IDS",
        {"000000000000"},
    )
    monkeypatch.setattr(
        LiteLLMBackend,
        "_aws_account_id_for_profile",
        staticmethod(lambda profile, region: "000000000000"),
    )
    with pytest.raises(RuntimeError, match="000000000000"):
        LiteLLMBackend.validate_bedrock_profile_for_model(
            "bedrock/converse/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            {"aws_profile_name": "research-bedrock"},
        )


def test_bedrock_guard_allows_regular_profile(monkeypatch):
    monkeypatch.setattr(
        "psychbench.framework.backends._DENIED_AWS_ACCOUNT_IDS", set(),
    )
    monkeypatch.setattr(
        LiteLLMBackend,
        "_aws_account_id_for_profile",
        staticmethod(lambda profile, region: "576847307829"),
    )
    LiteLLMBackend.validate_bedrock_profile_for_model(
        "bedrock/converse/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        {
            "aws_profile_name": "research-bedrock",
            "aws_region_name": "us-east-1",
        },
    )


def test_litellm_backend_carries_generation_params():
    b = get_backend(
        "openai", "gpt-4o-mini",
        temperature=0.7, max_tokens=128, api_base="http://localhost:8000",
    )
    assert b.temperature == 0.7
    assert b.max_tokens == 128
    assert b.api_base == "http://localhost:8000"


def test_litellm_backend_loads_openai_key_from_dotenv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=from-dotenv\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    LiteLLMBackend.load_dotenv_for_api_keys()
    assert os.environ["OPENAI_API_KEY"] == "from-dotenv"


def test_litellm_backend_dotenv_does_not_override_environment(
    tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=from-dotenv\n")
    monkeypatch.setenv("OPENAI_API_KEY", "already-set")
    LiteLLMBackend.load_dotenv_for_api_keys()
    assert os.environ["OPENAI_API_KEY"] == "already-set"


def test_stateful_echo_backend_tracks_history_and_resets():
    b = EchoBackend(model="echo-test")
    b.generate("first A", stateful=True)
    b.generate("second B", stateful=True)
    assert len(b.history) == 2
    b.reset()
    assert b.history == []

"""Model backend protocol + concrete implementations (lazy-imported).

Provider access is unified through ``LiteLLMBackend`` (over the ``litellm``
library): one interface to OpenAI, Anthropic, Gemini, Together/Fireworks/Groq,
OpenRouter, and a local vLLM endpoint. This lets configs name a provider+model
and nothing else in our code changes, and gives multi-turn ``messages``,
temperature, retries, and prompt-caching for free.

``EchoBackend`` (hermetic, deterministic) and ``HuggingFaceBackend`` (local
in-process transformers) are kept; the latter is legacy — prefer serving open
models via vLLM and reaching them through ``LiteLLMBackend`` (kind ``vllm``).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ModelBackend(Protocol):
    model: str

    def generate(self, prompt: str, stateful: bool = False) -> str: ...
    def reset(self) -> None: ...


class EchoBackend:
    """Deterministic offline backend for tests.

    Extracts the last standalone A/B/C letter in the prompt and returns it.
    Lets the test suite run without network or API keys. Accepts and ignores
    extra kwargs so it is a drop-in for the factory's uniform call signature.
    """

    def __init__(self, model: str, **_ignored: Any) -> None:
        self.model = model
        self.history: list[tuple[str, str]] = []

    def generate(self, prompt: str, stateful: bool = False) -> str:
        matches = re.findall(r"\b([ABC])\b", prompt)
        answer = matches[-1] if matches else "A"
        if stateful:
            self.history.append((prompt, answer))
        return answer

    def reset(self) -> None:
        self.history = []


class LiteLLMBackend:
    """Unified API/OSS backend over ``litellm``.

    ``model`` is the LiteLLM model string, including any provider prefix
    (e.g. ``gpt-4o-mini``, ``anthropic/claude-3-5-haiku-20241022``,
    ``gemini/gemini-1.5-flash``, ``hosted_vllm/meta-llama/Llama-3.1-8B-Instruct``).
    The ``get_backend`` factory builds these prefixes from a friendly ``kind``.

    ``litellm`` is imported lazily inside ``generate`` so constructing a backend
    (and the factory's routing) stays hermetic and dependency-free; the import
    only happens on the first real call.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        api_base: str | None = None,
        num_retries: int = 3,
        extra_params: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_base = api_base
        self.num_retries = num_retries
        self.extra_params = dict(extra_params or {})
        self._conversation: list[dict[str, str]] = []

    @staticmethod
    def load_dotenv_for_api_keys() -> None:
        """Load local .env API keys without overriding the shell environment."""
        try:
            from dotenv import load_dotenv  # type: ignore
        except ImportError:
            return
        dotenv_path = Path.cwd() / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path, override=False)

    @staticmethod
    def _aws_account_id_for_profile(profile: str, region: str | None) -> str:
        try:
            import boto3  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Bedrock backend requires `pip install boto3`"
            ) from e
        session = boto3.Session(profile_name=profile, region_name=region)
        return str(session.client("sts").get_caller_identity()["Account"])

    @staticmethod
    def validate_bedrock_profile_for_model(
        model: str, extra_params: dict[str, Any] | None,
    ) -> None:
        if not model.startswith("bedrock/"):
            return
        params = extra_params or {}
        profile = str(
            params.get("aws_profile_name") or os.environ.get("AWS_PROFILE") or ""
        )
        if not profile:
            raise RuntimeError(
                "Refusing Bedrock run without an explicit AWS profile. Set "
                "aws_profile_name in the config or AWS_PROFILE in the "
                "environment."
            )
        if profile.startswith(_DENIED_AWS_PROFILE_PREFIXES):
            raise RuntimeError(
                f"Refusing Bedrock run with AWS profile '{profile}'."
            )
        region = str(
            params.get("aws_region_name")
            or os.environ.get("AWS_REGION_NAME")
            or os.environ.get("AWS_DEFAULT_REGION")
            or os.environ.get("AWS_REGION")
            or "us-east-1"
        )
        account = LiteLLMBackend._aws_account_id_for_profile(profile, region)
        if account in _DENIED_AWS_ACCOUNT_IDS:
            raise RuntimeError(
                f"Refusing Bedrock run for denied AWS account {account} from "
                f"profile '{profile}'."
            )

    def _completion(self, messages: list[dict[str, str]]) -> str:
        try:
            import litellm  # type: ignore
        except ImportError as e:
            raise ImportError(
                "LiteLLM backend requires `pip install litellm`"
            ) from e
        self.load_dotenv_for_api_keys()
        self.validate_bedrock_profile_for_model(self.model, self.extra_params)
        # Drop provider-unsupported params instead of erroring, so a mixed panel
        # (e.g. reasoning_effort on gpt-5.x, temperature on o-series) just works.
        litellm.drop_params = True
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "num_retries": self.num_retries,
            **self.extra_params,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        resp = litellm.completion(**kwargs)
        return resp.choices[0].message.content or ""

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Send a full message list (system/user/assistant) and return text.

        Stateless: callers own the message list. Used by multi-turn experiments
        that control the conversation explicitly.
        """
        return self._completion(messages)

    def tool_step(
        self, messages: list[dict], tools: list[dict],
    ) -> dict[str, Any]:
        """One tool-calling turn.

        Returns a normalized dict: ``assistant_msg`` (append it to the running
        conversation), ``content`` (final text, may be None when tools were
        called), and ``tool_calls`` (list of {id, name, arguments}). The caller
        owns the loop: execute the tools, append tool-result messages, call
        again until ``tool_calls`` is empty.
        """
        try:
            import litellm  # type: ignore
        except ImportError as e:
            raise ImportError(
                "LiteLLM backend requires `pip install litellm`"
            ) from e
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "num_retries": self.num_retries,
            **self.extra_params,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        msg = litellm.completion(**kwargs).choices[0].message
        raw_calls = getattr(msg, "tool_calls", None) or []
        tool_calls = [
            {"id": tc.id, "name": tc.function.name,
             "arguments": tc.function.arguments}
            for tc in raw_calls
        ]
        assistant_msg: dict[str, Any] = {
            "role": "assistant", "content": msg.content or "",
        }
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in tool_calls
            ]
        return {
            "assistant_msg": assistant_msg,
            "content": msg.content,
            "tool_calls": tool_calls,
        }

    def generate(self, prompt: str, stateful: bool = False) -> str:
        if stateful:
            self._conversation.append({"role": "user", "content": prompt})
            messages = list(self._conversation)
        else:
            messages = [{"role": "user", "content": prompt}]
        text = self._completion(messages)
        if stateful:
            self._conversation.append({"role": "assistant", "content": text})
        return text

    def reset(self) -> None:
        self._conversation = []


class HuggingFaceBackend:
    """Legacy in-process transformers backend (one prompt at a time).

    Kept for local CPU/GPU runs that don't want a server. For throughput at
    scale, serve the model with vLLM and use ``LiteLLMBackend`` (kind ``vllm``).
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self._history: list[dict] = []
        try:
            from transformers import (  # type: ignore
                AutoModelForCausalLM, AutoTokenizer,
            )
            import torch  # type: ignore
        except ImportError as e:
            raise ImportError(
                "HuggingFace backend requires `pip install transformers torch`"
            ) from e
        token = os.environ.get("HF_TOKEN")
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model, token=token)
        self._model = AutoModelForCausalLM.from_pretrained(
            model, token=token, torch_dtype="auto", device_map="auto",
        )

    def generate(self, prompt: str, stateful: bool = False) -> str:
        torch = self._torch
        if stateful:
            self._history.append({"role": "user", "content": prompt})
            messages = list(self._history)
        else:
            messages = [{"role": "user", "content": prompt}]
        try:
            text_in = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            text_in = prompt
        inputs = self._tokenizer(text_in, return_tensors="pt").to(
            self._model.device
        )
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=64, do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(gen, skip_special_tokens=True).strip()
        if stateful:
            self._history.append({"role": "assistant", "content": text})
        return text

    def reset(self) -> None:
        self._history = []


# Friendly backend kind -> LiteLLM model-string prefix. The model string in the
# config is prefixed with this (unless it already carries a provider prefix).
# An empty prefix means "pass the model verbatim" (litellm treats a bare name
# as OpenAI; `litellm` kind lets the user write the full provider/model string).
_BEDROCK_PREFIX = "bedrock/converse/"
# Optional spend guard: Bedrock runs are refused for profiles matching these
# prefixes or resolving to these account IDs. Populate via comma-separated env
# vars to protect accounts that must never carry experiment spend.
_DENIED_AWS_PROFILE_PREFIXES = tuple(
    p for p in os.environ.get(
        "PSYCHBENCH_DENIED_AWS_PROFILE_PREFIXES", ""
    ).split(",") if p
)
_DENIED_AWS_ACCOUNT_IDS = {
    a for a in os.environ.get(
        "PSYCHBENCH_DENIED_AWS_ACCOUNT_IDS", ""
    ).split(",") if a
}
_LITELLM_PREFIX = {
    "litellm": "",
    "openai": "",
    "anthropic": "anthropic/",
    "gemini": "gemini/",
    "together": "together_ai/",
    "fireworks": "fireworks_ai/",
    "groq": "groq/",
    "openrouter": "openrouter/",
    "vllm": "hosted_vllm/",
    "bedrock": _BEDROCK_PREFIX,
}


def litellm_model_string(kind: str, model: str) -> str:
    """Resolve a friendly (kind, model) to the LiteLLM model string.

    Shared by ``get_backend`` and the cost estimator so pricing lookups use the
    exact string that will be sent to litellm.
    """
    prefix = _LITELLM_PREFIX.get(kind, "")
    if kind == "bedrock" and model.startswith("bedrock/"):
        return model
    if not prefix or model.startswith(prefix):
        return model
    return prefix + model


def get_backend(kind: str, model: str, **kwargs: Any) -> ModelBackend:
    """Build a backend by friendly ``kind``.

    ``echo`` and ``huggingface`` are special-cased; every other known kind is a
    provider routed through ``LiteLLMBackend``. Extra kwargs (temperature,
    max_tokens, api_base, num_retries, extra_params) flow to the backend.
    """
    if kind == "echo":
        return EchoBackend(model, **kwargs)
    if kind == "huggingface":
        return HuggingFaceBackend(model)
    if kind in _LITELLM_PREFIX:
        return LiteLLMBackend(litellm_model_string(kind, model), **kwargs)
    raise ValueError(
        f"Unknown backend '{kind}'. Known: "
        f"{['echo', 'huggingface', *_LITELLM_PREFIX]}"
    )

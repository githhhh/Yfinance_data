import importlib.metadata
import sys
from types import ModuleType, SimpleNamespace

import dotenv

from backtest.blind_rule_discovery.r6_agent import RDAgentProposer


def test_official_backend_retries_truncated_reasoning_from_original_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.8.0-test")
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: None)
    monkeypatch.setenv("RD_AGENT_MODEL", "deepseek/test-model")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-placeholder")
    monkeypatch.setenv("DEEPSEEK_API_BASE", "https://example.invalid")

    modules = {
        name: ModuleType(name)
        for name in (
            "rdagent",
            "rdagent.oai",
            "rdagent.oai.backend",
            "rdagent.oai.backend.base",
            "rdagent.oai.backend.litellm",
            "rdagent.log",
        )
    }
    backend = modules["rdagent.oai.backend.litellm"]
    modules["rdagent.oai.backend"].litellm = backend
    llm_settings = SimpleNamespace(max_retry=10)
    modules["rdagent.oai.backend.base"].LLM_SETTINGS = llm_settings
    backend.LITELLM_SETTINGS = SimpleNamespace(
        chat_model="old", chat_max_tokens=3000, chat_stream=False
    )

    calls = []

    def completion(**kwargs):
        assert kwargs["max_retries"] == kwargs["num_retries"] == 0
        assert kwargs["timeout"] == 240
        calls.append(kwargs)
        if len(calls) == 1:
            return "partial reasoning turn", "length"
        return '{"proposals": []}', "stop"

    backend.completion = completion

    class FakeBackend:
        def __init__(self, **kwargs):
            assert not any(kwargs.values())

        def _create_chat_completion_inner_function(self, messages, response_format=None, **kwargs):
            assert response_format is None
            assert backend.LITELLM_SETTINGS.chat_max_tokens == 8192
            assert backend.LITELLM_SETTINGS.chat_stream is True
            return backend.completion(messages=messages)

        def build_messages_and_create_chat_completion(self, user_prompt, system_prompt=None, **kwargs):
            assert llm_settings.max_retry == 3
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            last_error = None
            for _ in range(llm_settings.max_retry):
                try:
                    return self._create_chat_completion_auto_continue(messages=messages)
                except RuntimeError as exc:
                    last_error = exc
            raise last_error

    backend.LiteLLMAPIBackend = FakeBackend
    # Mirror rdagent==0.8.0 behavior observed on the execution machine: no debug method.
    modules["rdagent.log"].rdagent_logger = SimpleNamespace(
        info=lambda *a, **kw: None,
        warning=lambda *a, **kw: None,
        error=lambda *a, **kw: None,
        log_object=lambda *a, **kw: None,
    )

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    agent = RDAgentProposer(
        ledger_path=tmp_path / "ledger.json",
        cache_dir=tmp_path / "cache",
    )
    assert agent({"fold": "2025Q1", "round": 1}) == {"proposals": []}
    assert len(calls) == 2
    assert calls[0]["messages"] == calls[1]["messages"]
    assert all(message["role"] != "assistant" for message in calls[1]["messages"])
    assert agent.snapshot()["accounted_total"] == 44
    assert agent.snapshot()["rdagent_version"] == "0.8.0-test"
    assert agent.snapshot()["reasoning_auto_continue"] == "disabled_full_prompt_retry"

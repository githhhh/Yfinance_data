import importlib.metadata
import sys
from types import ModuleType, SimpleNamespace

import dotenv

from backtest.blind_rule_discovery.r6_agent import RDAgentProposer


def test_official_backend_tolerates_logger_without_debug_and_uses_8192_tokens(tmp_path, monkeypatch):
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
    modules["rdagent.oai.backend.base"].LLM_SETTINGS = SimpleNamespace(max_retry=10)
    backend.LITELLM_SETTINGS = SimpleNamespace(
        chat_model="old", chat_max_tokens=3000, chat_stream=True
    )

    calls = []

    def completion(**kwargs):
        assert kwargs["max_retries"] == kwargs["num_retries"] == 0
        calls.append(kwargs)
        return '{"proposals": []}'

    backend.completion = completion

    class FakeBackend:
        def __init__(self, **kwargs):
            assert not any(kwargs.values())

        def build_messages_and_create_chat_completion(self, **kwargs):
            assert backend.LITELLM_SETTINGS.chat_max_tokens == 8192
            return backend.completion(messages=[])

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
    assert len(calls) == 1
    assert agent.snapshot()["accounted_total"] == 43
    assert agent.snapshot()["rdagent_version"] == "0.8.0-test"

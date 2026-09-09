"""Metered RD-Agent proposer for bounded R8 Winner/Stop interactions."""
from __future__ import annotations

from contextlib import ExitStack
import importlib.metadata
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

from backtest.track_d_mechanism_discovery.request_budget import RequestBudgetLedger
from .r6_stability import canonical, digest


SYSTEM = """You propose falsifiable PIT interactions that separate W3 Fast Winner from Stop First.
This is known-history retrospective mechanism research, never untouched OOS or production alpha.
You see only purged historical aggregate summaries and inner-quarter feedback.
Propose at most 2 candidates; return {\"proposals\": []} when no useful idea remains.
Each proposal has exactly name, hypothesis, expression, target, tail, quantile.
target is winner or stop. Use only the extreme tail pairs: tail=low with quantile=0.2, or tail=high with quantile=0.8.
expression is a bounded JSON tree, maximum depth 2, using exactly two distinct available PIT features.
Leaves: {\"op\":\"raw\" or \"train_percentile\", \"feature\": an available feature}.
Binary nodes: {\"op\":\"difference\"/\"product\"/\"minimum\"/\"maximum\", \"left\":node,\"right\":node}.
Do not propose single-feature rules, three-or-more-feature rules, dates, tickers, market fields, future facts,
Python, arbitrary thresholds, changed outcome definitions, inferred earnings trajectories, or renamed duplicates.
Explain the economic mechanism and a falsifying observation. Return only compact JSON.
Never force a proposal merely to improve the reported historical result.
"""


class R8AgentProposer:
    """Official RD-Agent LiteLLM backend with model-aware cache identity and hard metering."""

    def __init__(self, *, ledger_path: Path, cache_dir: Path, model: str,
                 prior_used_floor: int, total_budget: int = 1000, run_cap: int = 80,
                 transport=None):
        if not model:
            raise ValueError("R8 requires an explicit model identifier")
        if not 0 <= prior_used_floor < total_budget:
            raise ValueError("invalid prior provider-usage floor")
        if not 1 <= run_cap <= total_budget-prior_used_floor:
            raise ValueError("invalid R8 run cap")
        self.model = model
        self.prior_used_floor = prior_used_floor
        self.total_budget = total_budget
        self.run_cap = run_cap
        self.ledger = RequestBudgetLedger(ledger_path, total_budget-prior_used_floor)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.start_used = self.ledger.attempts_used
        self.deadline = time.monotonic() + 2 * 3600
        self.transport = transport
        self.provenance = {
            "adapter": "custom_r8_with_official_rdagent_backend",
            "canonical_fin_factor": False,
            "model": model,
            "prior_used_floor": prior_used_floor,
            "provider_account_usage_verified": False,
            "cache_identity_includes_model": True,
            "reasoning_auto_continue": "disabled_full_prompt_retry",
        }

    def snapshot(self) -> dict:
        snap = self.ledger.snapshot()
        return {
            **snap,
            "accounted_total_floor": self.prior_used_floor + snap["attempts_used"],
            "run_attempts": snap["attempts_used"] - self.start_used,
            "run_cap": self.run_cap,
            **self.provenance,
        }

    def _purpose(self, prompt: str) -> str:
        return digest({"model": self.model, "system": SYSTEM, "prompt": prompt})

    def __call__(self, payload: dict) -> dict:
        prompt = canonical(payload)
        purpose = self._purpose(prompt)
        prompt_hash = purpose  # exact model/system/prompt identity, unlike legacy R6 cache keys.
        cache = self.cache_dir / f"{purpose}.json"
        if cache.exists():
            saved = json.loads(cache.read_text())
            if (saved.get("purpose") != purpose or saved.get("model") != self.model
                    or digest(saved.get("response")) != saved.get("response_hash")):
                raise RuntimeError("R8 response cache identity/hash mismatch")
            if not self.ledger.has_success(purpose):
                raise RuntimeError("R8 response cache lacks matching successful ledger record")
            return saved["response"]
        if self.ledger.has_success(purpose):
            raise RuntimeError("successful R8 request has no cache; refusing duplicate spend")

        attempt_number = 0

        def reserve():
            nonlocal attempt_number
            if time.monotonic() >= self.deadline:
                raise RuntimeError("R8 research time ceiling reached")
            if self.ledger.attempts_used-self.start_used >= self.run_cap:
                raise RuntimeError("R8 per-run provider-attempt ceiling reached")
            attempt_number += 1
            self.ledger.reserve_attempt(purpose, prompt_hash, attempt_number)

        try:
            if self.transport is not None:
                reserve()
                text = self.transport(SYSTEM, prompt)
                self.provenance["adapter"] = "test_transport_not_rdagent"
            else:
                text = self._official_call(prompt, reserve)
            if not attempt_number:
                raise RuntimeError("backend returned without a metered completion")
            response = json.loads(text)
            if not isinstance(response, dict) or not isinstance(response.get("proposals"), list):
                raise ValueError("response must contain proposals list")
            response_hash = digest(response)
            cache.write_text(canonical({"purpose": purpose, "model": self.model,
                                        "response_hash": response_hash, "response": response}) + "\n")
            self.ledger.mark_success(purpose, prompt_hash, response_hash)
            return response
        except Exception:
            if attempt_number:
                self.ledger.mark_failure(purpose, prompt_hash, attempt_number,
                                         "backend/JSON failure; raw provider errors omitted")
            raise RuntimeError("R8 Agent request failed; no result published. Check ledger/backend.") from None

    def _official_call(self, prompt: str, reserve):
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
        version = importlib.metadata.version("rdagent")
        from rdagent.oai.backend import litellm as backend_module
        from rdagent.oai.backend.base import LLM_SETTINGS
        from rdagent.log import rdagent_logger

        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("DEEPSEEK_API_BASE") or os.environ.get("OPENAI_API_BASE")
        if not api_key or not api_base:
            raise RuntimeError("configure API key/base in the existing environment")
        original_completion = backend_module.completion

        def counted_completion(*args, **kwargs):
            reserve()  # Every actual provider completion, including backend retries.
            kwargs.update(model=self.model, api_key=api_key, api_base=api_base,
                          timeout=240, max_retries=0, num_retries=0)
            return original_completion(*args, **kwargs)

        class R8LiteLLMBackend(backend_module.LiteLLMAPIBackend):
            def _create_chat_completion_auto_continue(self, messages, response_format=None, **kwargs):
                # DeepSeek reasoning turns cannot safely use RD-Agent 0.8.0's
                # assistant-content-only continuation. Retry the original prompt.
                for key in ("json_mode", "chat_cache_prefix", "seed", "json_target_type",
                            "add_json_in_prompt", "code_block_language", "code_block_fallback"):
                    kwargs.pop(key, None)
                response, finish_reason = self._create_chat_completion_inner_function(
                    messages=messages, response_format=response_format, **kwargs)
                if finish_reason == "length":
                    raise RuntimeError("R8 reasoning completion truncated; retry original prompt")
                return response

        self.provenance.update(rdagent_version=version,
                               backend="rdagent.oai.backend.litellm.LiteLLMAPIBackend")
        with ExitStack() as stack:
            for method in ("info", "warning", "error", "debug", "log_object"):
                if hasattr(rdagent_logger, method):
                    stack.enter_context(patch.object(rdagent_logger, method, lambda *a, **kw: None))
            stack.enter_context(patch.object(backend_module, "completion", counted_completion))
            stack.enter_context(patch.object(LLM_SETTINGS, "max_retry", 3))
            stack.enter_context(patch.object(backend_module.LITELLM_SETTINGS, "chat_model", self.model))
            stack.enter_context(patch.object(backend_module.LITELLM_SETTINGS, "chat_max_tokens", 8192))
            stack.enter_context(patch.object(backend_module.LITELLM_SETTINGS, "chat_stream", True))
            backend = R8LiteLLMBackend(
                use_chat_cache=False, dump_chat_cache=False,
                use_embedding_cache=False, dump_embedding_cache=False)
            return backend.build_messages_and_create_chat_completion(
                user_prompt=prompt, system_prompt=SYSTEM)

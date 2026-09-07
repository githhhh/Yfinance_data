"""Budgeted RD-Agent backend adapter; no generated code is executed.

This is a custom Research/Develop/Evaluate loop using the official RD-Agent LLM
backend, not an invocation of fin_factor/CoSTEER or the canonical blind protocol.
"""
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

SYSTEM = """You propose falsifiable, economically interpretable PIT risk features.
This is known-history retrospective research, never untouched OOS or causal proof.
The target is W3 Stop First, with winner removal cost explicitly constrained.
You see only purged historical aggregates and inner-quarter experiment feedback.
Propose at most 3 candidates; return {"proposals": []} when no useful idea remains.
Each proposal has exactly name, hypothesis, expression, tail, quantile.
expression is a bounded JSON tree (maximum depth 2, leaves at depth <=2).
Leaves: {"op":"raw" or "train_percentile", "feature": an available feature}.
Binary nodes: {"op":"difference"/"product"/"minimum"/"maximum", "left":node,"right":node}.
train_percentile is an empirical CDF fitted using past training data only.
tail is high or low; quantile is one of 0.2,0.4,0.6,0.8, fitted by the evaluator.
Explore interactions, including weak individual features, EPS/structure and volume/structure.
Explain the mechanism and a falsifying observation in hypothesis. Do not simply rename duplicates.
Do not infer earnings surprises or multi-period growth from a single EPS YoY field.
No dates, tickers, execution-time facts, market fields, labels, future information,
Python, arbitrary expressions, custom thresholds, or changed outcome definitions in rules.
The local evaluator selects using equal-snapshot stop-risk contrast and winner cost.
Return only compact JSON. Never force a rule to improve the reported result.
"""


class RDAgentProposer:
    def __init__(self, *, ledger_path: Path, cache_dir: Path, prior_used: int = 42,
                 total_budget: int = 1000, run_cap: int = 120, transport=None):
        if not 0 <= prior_used < total_budget or not 1 <= run_cap <= total_budget-prior_used:
            raise ValueError("invalid remaining request budget")
        self.ledger = RequestBudgetLedger(ledger_path, total_budget-prior_used)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.prior_used, self.total_budget, self.run_cap = prior_used, total_budget, run_cap
        self.start_used = self.ledger.attempts_used
        self.deadline = time.monotonic() + 3600
        self.transport = transport
        self.provenance = {"adapter": "custom_r6_with_official_rdagent_backend",
                           "canonical_fin_factor": False, "prior_used_user_reported": prior_used}

    def snapshot(self):
        return {**self.ledger.snapshot(), "prior_used_user_reported": self.prior_used,
                "accounted_total": self.prior_used+self.ledger.attempts_used,
                "run_attempts": self.ledger.attempts_used-self.start_used,
                "run_cap": self.run_cap,
                "provider_account_usage_verified": False, **self.provenance}

    def __call__(self, payload: dict) -> dict:
        prompt = canonical(payload)
        prompt_hash = self.ledger.prompt_hash(SYSTEM, prompt)
        cache = self.cache_dir / f"{prompt_hash}.json"
        if cache.exists():
            saved = json.loads(cache.read_text())
            if saved["prompt_hash"] != prompt_hash or digest(saved["response"]) != saved["response_hash"]:
                raise RuntimeError("response cache hash mismatch")
            if not self.ledger.has_success(prompt_hash):
                raise RuntimeError("response cache lacks matching successful request ledger")
            return saved["response"]
        if self.ledger.has_success(prompt_hash):
            raise RuntimeError("successful request has no cache; refusing duplicate spend")

        attempt_number = 0
        def reserve():
            nonlocal attempt_number
            if time.monotonic() >= self.deadline:
                raise RuntimeError("R6 research time ceiling reached")
            if self.ledger.attempts_used-self.start_used >= self.run_cap:
                raise RuntimeError("R6 per-run request ceiling reached")
            self.ledger.reserve_attempt(prompt_hash, prompt_hash, attempt_number + 1)
            attempt_number += 1
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
                raise ValueError("response must have proposals list")
            response_hash = digest(response)
            cache.write_text(canonical({"prompt_hash": prompt_hash, "response_hash": response_hash,
                                        "response": response}) + "\n")
            self.ledger.mark_success(prompt_hash, prompt_hash, response_hash)
            return response
        except Exception:
            if attempt_number:
                self.ledger.mark_failure(prompt_hash, prompt_hash, attempt_number,
                                         "backend/JSON failure; raw errors omitted to protect credentials")
            raise RuntimeError("RD-Agent request failed; no result published. Check backend setup and request ledger.") from None

    def _official_call(self, prompt, reserve):
        # Lazy imports: offline tests and --preflight never require credentials.
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
        version = importlib.metadata.version("rdagent")
        from rdagent.oai.backend import litellm as backend_module
        from rdagent.oai.backend.base import LLM_SETTINGS
        from rdagent.log import rdagent_logger

        model = os.environ.get("RD_AGENT_MODEL") or os.environ.get("CHAT_MODEL")
        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("DEEPSEEK_API_BASE") or os.environ.get("OPENAI_API_BASE")
        if not model or not api_key or not api_base:
            raise RuntimeError("configure model, API key and base in the existing environment")
        original_completion = backend_module.completion
        def counted_completion(*args, **kwargs):
            reserve()  # Every actual SDK call, including backend retries/continuations.
            kwargs.update(model=model, api_key=api_key, api_base=api_base,
                          timeout=90, max_retries=0, num_retries=0)
            return original_completion(*args, **kwargs)
        self.provenance.update(rdagent_version=version, model=model,
                               backend="rdagent.oai.backend.litellm.LiteLLMAPIBackend")
        # Keep provider config and training content out of SDK logs; persist only
        # hashed bounded responses and local aggregate experiment traces ourselves.
        with ExitStack() as stack:
            for method in ("info", "warning", "error", "debug", "log_object"):
                if hasattr(rdagent_logger, method):
                    stack.enter_context(patch.object(rdagent_logger, method, lambda *a, **kw: None))
            stack.enter_context(patch.object(backend_module, "completion", counted_completion))
            stack.enter_context(patch.object(LLM_SETTINGS, "max_retry", 1))
            stack.enter_context(patch.object(backend_module.LITELLM_SETTINGS, "chat_model", model))
            stack.enter_context(patch.object(backend_module.LITELLM_SETTINGS, "chat_max_tokens", 3000))
            stack.enter_context(patch.object(backend_module.LITELLM_SETTINGS, "chat_stream", False))
            backend = backend_module.LiteLLMAPIBackend(
                use_chat_cache=False, dump_chat_cache=False,
                use_embedding_cache=False, dump_embedding_cache=False)
            return backend.build_messages_and_create_chat_completion(
                user_prompt=prompt, system_prompt=SYSTEM)

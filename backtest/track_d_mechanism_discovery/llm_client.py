from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .config import ROOT, RAW_LLM_DIR, MAX_TOKENS_PER_CALL, REQUEST_HARD_LIMIT
from .request_budget import RequestBudgetLedger


def load_model_config() -> dict[str, Any]:
    load_dotenv(ROOT / ".env", override=False)
    model = (
        os.environ.get("TRACK_D_RDAGENT_MODEL")
        or os.environ.get("RD_AGENT_MODEL")
        or os.environ.get("CHAT_MODEL")
        or "deepseek/deepseek-v4-pro"
    ).strip()
    api_key = (
        os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()
    api_base = (
        os.environ.get("DEEPSEEK_API_BASE")
        or os.environ.get("OPENAI_API_BASE")
        or ""
    ).strip()
    if not api_key:
        raise RuntimeError("Track D requires DEEPSEEK_API_KEY (or OPENAI_API_KEY) in root .env")
    if model.startswith("deepseek/") and not api_base:
        raise RuntimeError("Track D DeepSeek run requires DEEPSEEK_API_BASE in root .env")

    try:
        global_retry = max(1, int(os.environ.get("MAX_RETRY", "15")))
        track_retry_raw = os.environ.get("TRACK_D_MAX_RETRY", "").strip()
        max_retry = (
            max(1, min(8, int(track_retry_raw)))
            if track_retry_raw
            else min(global_retry, 4)
        )
        retry_wait = max(0.0, float(os.environ.get("RETRY_WAIT_SECONDS", "15")))
        max_tokens = max(1000, int(os.environ.get("TRACK_D_MAX_TOKENS", str(MAX_TOKENS_PER_CALL))))
    except ValueError as exc:
        raise RuntimeError("Invalid Track D retry/token numeric environment configuration") from exc

    return {
        "model": model,
        "api_key": api_key,
        "api_base": api_base,
        "max_retry": max_retry,
        "retry_wait_seconds": retry_wait,
        "max_tokens": max_tokens,
        "reasoning_effort_configured": os.environ.get("REASONING_EFFORT", "").strip(),
        "reasoning_effort_forwarded": False,
    }


def _extract_json(text: str) -> Any:
    raw = str(text or "").strip()
    fence = chr(96) * 3
    if raw.startswith(fence):
        raw = re.sub(r"^" + re.escape(fence) + r"(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*" + re.escape(fence) + r"$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        starts = [i for i in (raw.find("{"), raw.find("[")) if i >= 0]
        if not starts:
            raise
        start = min(starts)
        end_obj = raw.rfind("}")
        end_arr = raw.rfind("]")
        end = max(end_obj, end_arr)
        if end <= start:
            raise
        return json.loads(raw[start : end + 1])


def _safe_filename(purpose_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", purpose_id)[:180]


class DeepSeekResearchClient:
    """Cached, budgeted DeepSeek client used by Track D's RD-Agent research loop."""

    def __init__(self, ledger_path: Path, hard_limit: int = REQUEST_HARD_LIMIT):
        self.cfg = load_model_config()
        self.ledger = RequestBudgetLedger(ledger_path, hard_limit)
        RAW_LLM_DIR.mkdir(parents=True, exist_ok=True)

    def call_json(
        self,
        purpose_id: str,
        system: str,
        prompt: str,
        *,
        temperature: float = 0.65,
    ) -> Any:
        safe = _safe_filename(purpose_id)
        parsed_path = RAW_LLM_DIR / f"{safe}.json"
        raw_path = RAW_LLM_DIR / f"{safe}.txt"
        meta_path = RAW_LLM_DIR / f"{safe}.meta.json"

        prompt_hash = self.ledger.prompt_hash(system, prompt)
        if parsed_path.exists():
            if not meta_path.exists():
                raise RuntimeError(f"Cached response for {purpose_id} has no metadata; stale cache refused.")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("prompt_hash") != prompt_hash:
                raise RuntimeError(
                    f"Cached response prompt hash mismatch for {purpose_id}; "
                    "source/prompt changed and stale cache cannot be reused."
                )
            return json.loads(parsed_path.read_text(encoding="utf-8"))
        if self.ledger.has_success(purpose_id):
            raise RuntimeError(
                f"Budget ledger marks {purpose_id} successful but cache file is missing; "
                "fail closed instead of spending a duplicate request."
            )

        try:
            from litellm import completion
        except Exception as exc:
            raise RuntimeError("LiteLLM is unavailable in the local RD-Agent environment") from exc

        base_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        kwargs: dict[str, Any] = {
            "model": self.cfg["model"],
            "messages": base_messages,
            "api_key": self.cfg["api_key"],
            "temperature": float(temperature),
            "max_tokens": int(self.cfg["max_tokens"]),
            "timeout": 180,
        }
        if self.cfg["api_base"]:
            kwargs["api_base"] = self.cfg["api_base"]

        last_exc: Exception | None = None
        compact_json_retry = False
        for attempt in range(1, int(self.cfg["max_retry"]) + 1):
            self.ledger.reserve_attempt(purpose_id, prompt_hash, attempt)
            try:
                attempt_kwargs = dict(kwargs)
                if compact_json_retry:
                    attempt_kwargs["messages"] = [
                        *base_messages,
                        {
                            "role": "user",
                            "content": (
                                "Retry because the previous answer was empty, truncated, or invalid JSON. "
                                "Return only compact valid JSON. Keep arrays short (normally <=2 items), "
                                "remove prose outside JSON, and prioritize the highest-value tests/claims."
                            ),
                        },
                    ]
                    attempt_kwargs["temperature"] = min(float(temperature), 0.35)
                response = completion(**attempt_kwargs)
                content = response.choices[0].message.content
                if not content:
                    raise RuntimeError("empty model response")
                parsed = _extract_json(str(content))
                raw_path.write_text(str(content), encoding="utf-8")
                parsed_path.write_text(
                    json.dumps(parsed, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                response_hash = hashlib.sha256(str(content).encode("utf-8")).hexdigest()
                meta_path.write_text(json.dumps({
                    "purpose_id": purpose_id,
                    "prompt_hash": prompt_hash,
                    "response_hash": response_hash,
                    "model": self.cfg["model"],
                    "api_base_configured": bool(self.cfg["api_base"]),
                    "max_tokens": int(self.cfg["max_tokens"]),
                    "reasoning_effort_configured": self.cfg["reasoning_effort_configured"],
                    "reasoning_effort_forwarded": False,
                    "attempt": attempt,
                    "compact_json_retry": compact_json_retry,
                }, indent=2), encoding="utf-8")
                self.ledger.mark_success(purpose_id, prompt_hash, response_hash)
                return parsed
            except Exception as exc:
                last_exc = exc
                self.ledger.mark_failure(purpose_id, prompt_hash, attempt, str(exc))
                if isinstance(exc, json.JSONDecodeError) or "empty model response" in str(exc).lower():
                    compact_json_retry = True
                if attempt >= int(self.cfg["max_retry"]):
                    break
                if self.ledger.remaining <= 0:
                    break
                time.sleep(float(self.cfg["retry_wait_seconds"]))

        raise RuntimeError(
            f"Track D DeepSeek request failed for {purpose_id}; "
            f"budget={self.ledger.snapshot()}"
        ) from last_exc

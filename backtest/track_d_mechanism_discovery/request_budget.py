from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any


class RequestBudgetExceeded(RuntimeError):
    pass


class DuplicateResearchRequest(RuntimeError):
    pass


class RequestBudgetLedger:
    """Persistent request-attempt ledger.

    Every provider attempt consumes one unit, including retries. Successful purpose IDs
    are immutable and are expected to be served from cached response files on resume.
    """

    def __init__(self, path: Path, hard_limit: int):
        self.path = path
        self.hard_limit = int(hard_limit)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
            old = int(self.data.get("hard_limit", hard_limit))
            if old != self.hard_limit:
                raise RuntimeError(
                    f"Request budget mismatch: ledger={old}, configured={self.hard_limit}."
                )
        else:
            self.data = {
                "hard_limit": self.hard_limit,
                "attempts_used": 0,
                "successful_calls": 0,
                "failed_attempts": 0,
                "successful_purposes": {},
                "successful_prompt_hashes": {},
                "attempt_log": [],
            }
            self._save()

    @staticmethod
    def prompt_hash(system: str, prompt: str) -> str:
        return hashlib.sha256((system + "\n---\n" + prompt).encode("utf-8")).hexdigest()

    @property
    def attempts_used(self) -> int:
        return int(self.data.get("attempts_used", 0))

    @property
    def remaining(self) -> int:
        return self.hard_limit - self.attempts_used

    def has_success(self, purpose_id: str) -> bool:
        return purpose_id in self.data.get("successful_purposes", {})

    def reserve_attempt(self, purpose_id: str, prompt_hash: str, attempt_no: int) -> None:
        if self.attempts_used >= self.hard_limit:
            raise RequestBudgetExceeded(
                f"Track D request hard limit reached ({self.hard_limit}). "
                "Stop; do not silently increase the budget."
            )

        prior_purpose = self.data.get("successful_prompt_hashes", {}).get(prompt_hash)
        if prior_purpose and prior_purpose != purpose_id:
            raise DuplicateResearchRequest(
                f"Exact research prompt already succeeded under {prior_purpose}; "
                f"refusing duplicate purpose {purpose_id}."
            )

        self.data["attempts_used"] = self.attempts_used + 1
        self.data["attempt_log"].append({
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "purpose_id": purpose_id,
            "prompt_hash": prompt_hash,
            "attempt_no": int(attempt_no),
            "status": "reserved",
        })
        self._save()

    def mark_failure(self, purpose_id: str, prompt_hash: str, attempt_no: int, error: str) -> None:
        self.data["failed_attempts"] = int(self.data.get("failed_attempts", 0)) + 1
        self.data["attempt_log"].append({
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "purpose_id": purpose_id,
            "prompt_hash": prompt_hash,
            "attempt_no": int(attempt_no),
            "status": "failed",
            "error": str(error)[:1000],
        })
        self._save()

    def mark_success(self, purpose_id: str, prompt_hash: str, response_hash: str) -> None:
        self.data["successful_calls"] = int(self.data.get("successful_calls", 0)) + 1
        self.data.setdefault("successful_purposes", {})[purpose_id] = {
            "prompt_hash": prompt_hash,
            "response_hash": response_hash,
        }
        self.data.setdefault("successful_prompt_hashes", {})[prompt_hash] = purpose_id
        self.data["attempt_log"].append({
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "purpose_id": purpose_id,
            "prompt_hash": prompt_hash,
            "status": "success",
            "response_hash": response_hash,
        })
        self._save()

    def snapshot(self) -> dict[str, Any]:
        return {
            "hard_limit": self.hard_limit,
            "attempts_used": self.attempts_used,
            "remaining": self.remaining,
            "successful_calls": int(self.data.get("successful_calls", 0)),
            "failed_attempts": int(self.data.get("failed_attempts", 0)),
        }

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

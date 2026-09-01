from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ..config import FAMILY_BUDGETS, ROOT
from .blind_prompt import generate_blind_discovery_prompt


DISCOVERY_FAMILIES = (
    "industry_breadth",
    "continuous",
    "linear_ranking",
    "portfolio",
    "novel_heuristic",
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_model_name(model: str) -> str:
    return str(model or "").strip()


def _load_rdagent_model_config() -> dict[str, Any]:
    """Load root .env settings used by the local DeepSeek-backed RD-Agent bridge."""
    load_dotenv(ROOT / ".env", override=False)

    model = (
        os.environ.get("TRACK_C_RDAGENT_MODEL")
        or os.environ.get("RD_AGENT_MODEL")
        or os.environ.get("CHAT_MODEL")
        or "deepseek/deepseek-v4-pro"
    ).strip()
    api_key = (
        os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("AZURE_API_KEY")
        or ""
    ).strip()
    api_base = (
        os.environ.get("DEEPSEEK_API_BASE")
        or os.environ.get("OPENAI_API_BASE")
        or ""
    ).strip()

    if not api_key:
        raise RuntimeError(
            "Track C RD-Agent policy discovery requires DEEPSEEK_API_KEY "
            "(or compatible OPENAI_API_KEY) in root .env."
        )
    if model.startswith("deepseek/") and not api_base:
        raise RuntimeError(
            "DeepSeek Track C discovery requires DEEPSEEK_API_BASE in root .env "
            "so the sealed run cannot silently fall back to another endpoint."
        )

    try:
        max_retry = max(1, int(os.environ.get("MAX_RETRY", "15")))
    except ValueError as exc:
        raise RuntimeError("MAX_RETRY must be an integer >= 1") from exc
    try:
        retry_wait_seconds = max(0.0, float(os.environ.get("RETRY_WAIT_SECONDS", "15")))
    except ValueError as exc:
        raise RuntimeError("RETRY_WAIT_SECONDS must be a non-negative number") from exc

    return {
        "model": model,
        "backend": "litellm",
        "api_key": api_key,
        "api_base": api_base,
        "max_retry": max_retry,
        "retry_wait_seconds": retry_wait_seconds,
        # Recorded for provenance only. We intentionally do not forward it to
        # OpenAI-compatible DeepSeek chat/completions because provider support
        # is endpoint-specific.
        "reasoning_effort": os.environ.get("REASONING_EFFORT", "").strip(),
    }


def _outcome_blind_summary(anon_df: pd.DataFrame) -> dict[str, Any]:
    """Create a compact Train-only summary. It never includes ticker/date identities or outcomes."""
    summary: dict[str, Any] = {
        "rows": int(len(anon_df)),
        "snapshots": int(anon_df["snapshot_date"].nunique()) if "snapshot_date" in anon_df.columns else 0,
        "numeric": {},
        "categorical": {},
    }

    for col in anon_df.columns:
        if col in {"code", "snapshot_date"}:
            continue
        s = anon_df[col]
        if pd.api.types.is_numeric_dtype(s):
            # Pandas reports bool as numeric, but NumPy 2.x quantile interpolation
            # cannot subtract boolean values. Normalize every numeric summary
            # input to float64 before quantile calculation.
            if pd.api.types.is_bool_dtype(s):
                vals = s.astype("float64").dropna()
            else:
                vals = pd.to_numeric(s, errors="coerce").astype("float64").dropna()
            if vals.empty:
                continue
            qs = vals.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
            summary["numeric"][col] = {
                "non_null": int(len(vals)),
                "p10": round(float(qs.get(0.1, np.nan)), 6),
                "p25": round(float(qs.get(0.25, np.nan)), 6),
                "p50": round(float(qs.get(0.5, np.nan)), 6),
                "p75": round(float(qs.get(0.75, np.nan)), 6),
                "p90": round(float(qs.get(0.9, np.nan)), 6),
            }
        elif col in {"ibd_candidate_rule", "ibd_entry_status", "industry", "sector"}:
            vc = s.astype(str).value_counts(dropna=False).head(20)
            summary["categorical"][col] = {str(k): int(v) for k, v in vc.items()}

    return summary


def _extract_json_payload(text: str) -> dict[str, Any]:
    raw = text.strip()
    fence = chr(96) * 3
    if raw.startswith(fence):
        raw = re.sub(r"^" + re.escape(fence) + r"(?:json)?\\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\\s*" + re.escape(fence) + r"$", "", raw)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError("RD-Agent model response did not contain a JSON object.")
        obj = json.loads(raw[start : end + 1])
    if not isinstance(obj, dict):
        raise RuntimeError("RD-Agent model response root must be a JSON object.")
    return obj


def _call_litellm(cfg: dict[str, Any], prompt: str) -> str:
    """Call the configured DeepSeek endpoint through LiteLLM with fail-closed retries."""
    try:
        from litellm import completion
    except Exception as exc:
        raise RuntimeError(
            "LiteLLM is unavailable. Use the same Python environment that provides RD-Agent."
        ) from exc

    messages = [
        {
            "role": "system",
            "content": (
                "You are the blind policy-discovery component of RD-Agent. "
                "Return only valid JSON. Never use future returns, stop labels, ticker identity, "
                "calendar identity, or any information outside the supplied PIT-safe summary."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    kwargs: dict[str, Any] = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": 0.8,
        "api_key": cfg["api_key"],
    }
    if cfg.get("api_base"):
        kwargs["api_base"] = cfg["api_base"]

    max_retry = int(cfg["max_retry"])
    retry_wait = float(cfg["retry_wait_seconds"])
    last_exc: Exception | None = None

    for attempt in range(1, max_retry + 1):
        try:
            response = completion(**kwargs)
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("RD-Agent LiteLLM returned an empty response.")
            return str(content)
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retry:
                break
            time.sleep(retry_wait)

    raise RuntimeError(
        f"RD-Agent LiteLLM call failed after {max_retry} attempts "
        f"using model={cfg['model']!r} and the configured DeepSeek api_base."
    ) from last_exc


def run_rdagent_policy_discovery(
    anon_df: pd.DataFrame,
    feature_manifest: dict[str, Any],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate blind Track C policy specs with the RD-Agent-configured DeepSeek/LiteLLM backend.

    This function fails closed. There is no deterministic hard-coded fallback.
    """
    cfg = _load_rdagent_model_config()
    model = cfg["model"]
    summary = _outcome_blind_summary(anon_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, Any]] = []
    family_runs: list[dict[str, Any]] = []

    for family in DISCOVERY_FAMILIES:
        budget = int(FAMILY_BUDGETS[family])
        prompt = generate_blind_discovery_prompt(
            feature_manifest=feature_manifest,
            family=family,
            budget=budget,
            data_summary=summary,
        )
        response_text = _call_litellm(cfg, prompt)
        payload = _extract_json_payload(response_text)
        proposals = payload.get("proposals")
        if not isinstance(proposals, list) or not proposals:
            raise RuntimeError(f"RD-Agent returned no proposals for family={family!r}.")

        raw_path = raw_dir / f"{family}.txt"
        raw_path.write_text(response_text, encoding="utf-8")
        raw_hash = _sha256_text(response_text)

        accepted = 0
        for idx, item in enumerate(proposals[:budget], 1):
            if not isinstance(item, dict):
                continue
            item_family = str(item.get("family") or family).strip()
            if item_family != family:
                continue
            name = re.sub(r"[^a-zA-Z0-9_]+", "_", str(item.get("name") or f"agent_{idx:02d}")).strip("_")
            if not name:
                name = f"agent_{idx:02d}"
            params = item.get("params")
            if not isinstance(params, dict):
                continue

            all_records.append(
                {
                    "family": family,
                    "name": name[:80],
                    "hypothesis": str(item.get("hypothesis") or "").strip()[:1000],
                    "params": params,
                    "source_response_hash": raw_hash,
                    "source_response_path": str(raw_path.relative_to(output_dir.parent)),
                    "source_model": _safe_model_name(model),
                }
            )
            accepted += 1

        if accepted == 0:
            raise RuntimeError(f"RD-Agent produced no schema-valid proposals for family={family!r}.")

        family_runs.append(
            {
                "family": family,
                "budget": budget,
                "raw_response_hash": raw_hash,
                "raw_response_path": str(raw_path.relative_to(output_dir.parent)),
                "accepted_raw_proposals": accepted,
            }
        )

    provenance = {
        "engine": "track_c_rdagent_policy_bridge",
        "backend": cfg["backend"],
        "model": _safe_model_name(model),
        "api_base_configured": bool(cfg.get("api_base")),
        "max_retry": int(cfg["max_retry"]),
        "retry_wait_seconds": float(cfg["retry_wait_seconds"]),
        "reasoning_effort_configured": str(cfg.get("reasoning_effort") or ""),
        "reasoning_effort_forwarded": False,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "train_only": True,
        "outcome_blind": True,
        "ticker_anonymized": True,
        "date_anonymized": True,
        "family_runs": family_runs,
    }
    provenance_path = output_dir / "provenance.json"
    provenance_text = json.dumps(provenance, indent=2, sort_keys=True)
    provenance_path.write_text(provenance_text, encoding="utf-8")
    provenance["provenance_path"] = str(provenance_path.relative_to(output_dir.parent))
    provenance["provenance_hash"] = _sha256_text(provenance_text)

    return all_records, provenance

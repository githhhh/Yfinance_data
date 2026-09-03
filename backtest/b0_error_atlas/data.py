from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    AUDIT_AS_OF_DATE,
    BASE_PRICE_CACHE,
    B0_AUDIT_MANIFEST_SOURCE,
    B0_STATE_SOURCE,
    FEATURE_MANIFEST_SOURCE,
    PANEL_SOURCE,
    YAHOO_SUPPLEMENT,
)


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_prices(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["code", "date", "open", "high", "low", "close", "volume", "source"]
        )
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    if "ticker" in out.columns and "code" not in out.columns:
        out = out.rename(columns={"ticker": "code"})
    required = {"code", "date", "open", "high", "low", "close"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise RuntimeError(f"Price frame missing required columns: {missing}")
    if "volume" not in out.columns:
        out["volume"] = np.nan
    out["code"] = out["code"].astype(str).str.upper().str.strip()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "source" not in out.columns:
        out["source"] = source
    out = out[
        ["code", "date", "open", "high", "low", "close", "volume", "source"]
    ]
    out = out[out["date"] <= pd.Timestamp(AUDIT_AS_OF_DATE)]
    return (
        out.drop_duplicates(["code", "date"], keep="first")
        .sort_values(["code", "date"])
        .reset_index(drop=True)
    )


def load_frozen_prices() -> pd.DataFrame:
    base = _normalize_prices(pd.read_parquet(BASE_PRICE_CACHE), "base_cache")
    supplement = _normalize_prices(pd.read_parquet(YAHOO_SUPPLEMENT), "yahoo_supplement")

    if supplement.empty:
        return base
    if base.empty:
        return supplement

    key = ["code", "date"]
    merged = (
        base.set_index(key)
        .combine_first(supplement.set_index(key))
        .reset_index()
    )
    return _normalize_prices(merged, "merged")


def load_feature_manifest() -> dict[str, Any]:
    return json.loads(FEATURE_MANIFEST_SOURCE.read_text(encoding="utf-8"))


def allowed_raw_features(panel: pd.DataFrame) -> tuple[list[str], list[str]]:
    manifest = load_feature_manifest()
    numeric: list[str] = []
    categorical: list[str] = []
    forbidden_prefixes = ("w1_", "w2_", "w3_", "w4_", "b0_")
    forbidden_exact = {
        "b0_eligible",
        "is_b0",
        "current_b0_raw_rank",
        "current_b0_lane",
        "current_b0_eligible",
        "current_b0_reject_reasons",
        "current_b0_selected",
        "current_b0_pick_order",
    }

    for name, spec in manifest.get("features", {}).items():
        if not spec.get("allowed_for_discovery", False):
            continue
        if name in forbidden_exact or name.startswith(forbidden_prefixes):
            continue
        if name not in panel.columns:
            continue
        dtype = str(spec.get("data_type", "")).lower()
        if dtype in {"float", "int", "bool"}:
            numeric.append(name)
        else:
            categorical.append(name)

    return sorted(set(numeric)), sorted(set(categorical))


def load_analysis_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    panel = pd.read_parquet(PANEL_SOURCE).copy()
    state = pd.read_csv(B0_STATE_SOURCE).copy()
    audit_manifest = json.loads(
        B0_AUDIT_MANIFEST_SOURCE.read_text(encoding="utf-8")
    )

    for frame in [panel, state]:
        frame["snapshot_date"] = frame["snapshot_date"].astype(str)
        frame["code"] = frame["code"].astype(str).str.upper().str.strip()

    state_cols = [
        "snapshot_date",
        "code",
        "current_b0_raw_rank",
        "current_b0_lane",
        "current_b0_eligible",
        "current_b0_reject_reasons",
        "current_b0_selected",
        "current_b0_pick_order",
        "next_open_w4_return_pct",
        "next_open_w4_stop8",
        "next_open_entry_date",
        "next_open_end_date",
        "next_open_price_valid",
        "next_open_invalid_reason",
    ]
    missing_state = sorted(set(state_cols) - set(state.columns))
    if missing_state:
        raise RuntimeError(f"B0 state missing columns: {missing_state}")

    state = state[state_cols]
    merged = panel.merge(
        state,
        on=["snapshot_date", "code"],
        how="left",
        validate="one_to_one",
    )

    if merged["current_b0_eligible"].isna().any():
        bad = merged.loc[
            merged["current_b0_eligible"].isna(), ["snapshot_date", "code"]
        ].head(10)
        raise RuntimeError(
            "B0 state merge incomplete; first missing rows="
            + bad.to_dict(orient="records").__repr__()
        )

    bool_cols = [
        "current_b0_eligible",
        "current_b0_selected",
        "next_open_w4_stop8",
        "next_open_price_valid",
    ]
    for col in bool_cols:
        merged[col] = merged[col].fillna(False).astype(bool)

    numeric_raw, categorical_raw = allowed_raw_features(merged)

    manifest = {
        "source_git_sha": git_sha(),
        "panel_hash": sha256_file(PANEL_SOURCE),
        "feature_manifest_hash": sha256_file(FEATURE_MANIFEST_SOURCE),
        "b0_state_hash": sha256_file(B0_STATE_SOURCE),
        "b0_audit_manifest_hash": sha256_file(B0_AUDIT_MANIFEST_SOURCE),
        "base_price_cache_hash": sha256_file(BASE_PRICE_CACHE),
        "yahoo_supplement_hash": sha256_file(YAHOO_SUPPLEMENT),
        "b0_audit_protocol_version": audit_manifest.get("protocol_version"),
        "b0_audit_source_git_sha": audit_manifest.get("source_git_sha"),
        "rows": int(len(merged)),
        "weeks": int(merged["snapshot_date"].nunique()),
        "min_snapshot": str(merged["snapshot_date"].min()),
        "max_snapshot": str(merged["snapshot_date"].max()),
        "raw_numeric_features": numeric_raw,
        "raw_categorical_features": categorical_raw,
    }
    return (
        merged.sort_values(["snapshot_date", "code"]).reset_index(drop=True),
        manifest,
    )

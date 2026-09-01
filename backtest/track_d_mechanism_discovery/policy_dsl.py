from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from dashboard.skill_industry_eps_known import rank_skill_industry_eps_known, reasoned_item
from backtest.track_c_ranking_discovery.discovery_sandbox.discovery_runner import controlled_eligible
from .config import FEATURE_MANIFEST_PATH, TOP_N


PSEUDO_FEATURES = {
    "b0_lane": "string",
    "b0_raw_rank": "float",
    "b0_evidence_balance": "float",
    "b0_risk_count": "float",
}
NUMERIC_TRANSFORMS = {"identity", "zscore", "rank_pct", "neg_abs"}
CONDITION_OPS = {"gt", "gte", "lt", "lte", "eq", "neq", "in", "is_true", "is_false"}
INDUSTRY_MODES = {"distinct_1", "max_2_per_ind", "unconstrained"}
CAPACITY_MODES = {"fixed", "min_score", "score_gap", "top1_confidence"}


def _load_allowed_types() -> dict[str, str]:
    manifest = json.loads(FEATURE_MANIFEST_PATH.read_text(encoding="utf-8"))
    allowed = {
        name: str(meta.get("data_type", ""))
        for name, meta in manifest["features"].items()
        if meta.get("allowed_for_discovery") is True
    }
    allowed.update(PSEUDO_FEATURES)
    return allowed


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def semantic_spec_hash(spec: dict[str, Any]) -> str:
    core = dict(spec)
    core.pop("policy_id", None)
    core.pop("description", None)
    core.pop("research_origin", None)
    return hashlib.sha256(_canonical(core).encode("utf-8")).hexdigest()


def _safe_policy_id(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "")).strip("_")
    if not text:
        raise ValueError("policy_id must be non-empty")
    return text[:120]


def _bounded_number(value: object, lo: float, hi: float, name: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(x) or not lo <= x <= hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {x}")
    return x


def _validate_condition(cond: dict[str, Any], allowed: dict[str, str]) -> dict[str, Any]:
    if not isinstance(cond, dict):
        raise ValueError("condition must be an object")
    feature = str(cond.get("feature") or "")
    op = str(cond.get("op") or "")
    if feature not in allowed:
        raise ValueError(f"condition feature {feature!r} is not PIT-allowed")
    if op not in CONDITION_OPS:
        raise ValueError(f"unsupported condition op {op!r}")
    out = {"feature": feature, "op": op}
    if op not in {"is_true", "is_false"}:
        if "value" not in cond:
            raise ValueError(f"condition {feature}/{op} requires value")
        value = cond["value"]
        if op == "in":
            if not isinstance(value, list) or not value or len(value) > 20:
                raise ValueError("condition op=in requires 1..20 values")
        out["value"] = value
    return out


def validate_policy_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the safe declarative Track D policy DSL."""
    if not isinstance(spec, dict):
        raise ValueError("policy spec must be a JSON object")
    allowed = _load_allowed_types()

    policy_id = _safe_policy_id(spec.get("policy_id"))
    base = str(spec.get("base") or "zero")
    if base not in {"zero", "b0_rank"}:
        raise ValueError("base must be zero or b0_rank")

    raw_terms = spec.get("terms", [])
    if not isinstance(raw_terms, list) or len(raw_terms) > 16:
        raise ValueError("terms must be an array with <=16 items")
    terms: list[dict[str, Any]] = []

    for idx, term in enumerate(raw_terms):
        if not isinstance(term, dict):
            raise ValueError(f"term[{idx}] must be an object")
        kind = str(term.get("type") or "")
        if kind == "linear":
            feature = str(term.get("feature") or "")
            if feature not in allowed or allowed[feature] not in {"float", "int", "bool"}:
                raise ValueError(f"linear feature {feature!r} is not numeric PIT-allowed")
            transform = str(term.get("transform") or "zscore")
            if transform not in NUMERIC_TRANSFORMS:
                raise ValueError(f"unsupported transform {transform!r}")
            terms.append({
                "type": "linear",
                "feature": feature,
                "transform": transform,
                "weight": _bounded_number(term.get("weight", 1.0), -10.0, 10.0, "weight"),
            })
        elif kind == "threshold":
            conditions = term.get("conditions")
            if not isinstance(conditions, list) or not (1 <= len(conditions) <= 6):
                raise ValueError("threshold.conditions must contain 1..6 conditions")
            terms.append({
                "type": "threshold",
                "logic": "any" if str(term.get("logic") or "all") == "any" else "all",
                "conditions": [_validate_condition(x, allowed) for x in conditions],
                "add": _bounded_number(term.get("add", 0.0), -10.0, 10.0, "threshold.add"),
            })
        elif kind == "interaction":
            left = str(term.get("left") or "")
            right = str(term.get("right") or "")
            if (
                left not in allowed or right not in allowed
                or allowed[left] not in {"float", "int", "bool"}
                or allowed[right] not in {"float", "int", "bool"}
            ):
                raise ValueError("interaction features must both be numeric PIT-allowed")
            transform = str(term.get("transform") or "zscore")
            if transform not in NUMERIC_TRANSFORMS:
                raise ValueError(f"unsupported interaction transform {transform!r}")
            terms.append({
                "type": "interaction",
                "left": left,
                "right": right,
                "transform": transform,
                "weight": _bounded_number(term.get("weight", 1.0), -10.0, 10.0, "interaction.weight"),
            })
        else:
            raise ValueError(f"unsupported term type {kind!r}")

    selector = spec.get("selector") or {}
    if not isinstance(selector, dict):
        raise ValueError("selector must be an object")
    industry_mode = str(selector.get("industry_mode") or "distinct_1")
    if industry_mode not in INDUSTRY_MODES:
        raise ValueError(f"unsupported industry_mode {industry_mode!r}")

    capacity = selector.get("capacity") or {"mode": "fixed", "max_positions": 3}
    if not isinstance(capacity, dict):
        raise ValueError("selector.capacity must be an object")
    mode = str(capacity.get("mode") or "fixed")
    if mode not in CAPACITY_MODES:
        raise ValueError(f"unsupported capacity mode {mode!r}")
    max_positions = int(capacity.get("max_positions", TOP_N))
    if max_positions not in {1, 2, 3}:
        raise ValueError("max_positions must be 1, 2, or 3")
    normalized_capacity: dict[str, Any] = {"mode": mode, "max_positions": max_positions}
    if mode == "min_score":
        normalized_capacity["min_score"] = _bounded_number(
            capacity.get("min_score", 0.0), -20.0, 20.0, "capacity.min_score"
        )
    elif mode in {"score_gap", "top1_confidence"}:
        normalized_capacity["gap"] = _bounded_number(
            capacity.get("gap", 1.0), 0.05, 10.0, "capacity.gap"
        )
        if "min_score" in capacity:
            normalized_capacity["min_score"] = _bounded_number(
                capacity["min_score"], -20.0, 20.0, "capacity.min_score"
            )

    normalized = {
        "policy_id": policy_id,
        "description": str(spec.get("description") or "").strip()[:2000],
        "research_origin": str(spec.get("research_origin") or "").strip()[:300],
        "base": base,
        "terms": terms,
        "selector": {
            "industry_mode": industry_mode,
            "capacity": normalized_capacity,
        },
    }
    normalized["spec_hash"] = semantic_spec_hash(normalized)
    return normalized


def _numeric_transform(series: pd.Series, transform: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if transform == "identity":
        return s.fillna(0.0)
    if transform == "zscore":
        mean = float(s.mean()) if s.notna().any() else 0.0
        std = float(s.std(ddof=0)) if s.notna().sum() > 1 else 0.0
        denom = std if std > 1e-9 else 1.0
        return ((s.fillna(mean) - mean) / denom).clip(-4.0, 4.0)
    if transform == "rank_pct":
        return s.rank(pct=True, method="average").fillna(0.5)
    if transform == "neg_abs":
        return -s.abs().fillna(0.0)
    raise ValueError(transform)


def _condition_mask(df: pd.DataFrame, cond: dict[str, Any]) -> pd.Series:
    s = df[cond["feature"]]
    op = cond["op"]
    if op in {"is_true", "is_false"}:
        def as_bool(v: object) -> bool:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return False
            if isinstance(v, (bool, np.bool_)):
                return bool(v)
            text = str(v).strip().lower()
            return text in {"true", "1", "1.0", "yes"}
        truth = s.map(as_bool)
        return truth if op == "is_true" else ~truth
    value = cond.get("value")
    if op in {"gt", "gte", "lt", "lte"}:
        left = pd.to_numeric(s, errors="coerce")
        right = float(value)
        if op == "gt":
            return left > right
        if op == "gte":
            return left >= right
        if op == "lt":
            return left < right
        return left <= right
    if op in {"eq", "neq", "in"} and pd.api.types.is_numeric_dtype(s):
        left = pd.to_numeric(s, errors="coerce")
        if op == "in":
            values = [float(x) for x in value]
            return left.isin(values)
        right = float(value)
        return (left == right) if op == "eq" else (left != right)
    if op == "eq":
        return s.astype(str) == str(value)
    if op == "neq":
        return s.astype(str) != str(value)
    if op == "in":
        return s.astype(str).isin([str(x) for x in value])
    raise ValueError(op)


def _augment_b0_features(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    df = snapshot_df.copy()
    ranked = rank_skill_industry_eps_known(snapshot_df)
    rank_map = {x.code: float(x.raw_rank) for x in ranked}
    lane_map = {x.code: x.lane for x in ranked}
    ev_map: dict[str, float] = {}
    risk_map: dict[str, float] = {}
    for row_idx, (_, row) in enumerate(snapshot_df.iterrows()):
        item = reasoned_item(row, row_idx)
        evidence = sum(
            x in item.reason_codes
            for x in [
                "near_buy_point", "volume_confirms_breakout", "eps_acceleration_support",
                "weekly_volume_follow_through", "near_52w_high", "dry_pullback",
            ]
        )
        risk = sum(
            x in item.risk_codes
            for x in [
                "non_actionable_radar_only", "freshness_missing", "below_candidate_buy_point",
                "extended_from_buy_point", "entry_volume_missing",
                "entry_volume_below_standard", "pullback_not_dry",
            ]
        )
        code = str(row.get("code", "") or "").strip()
        ev_map[code] = float(evidence - risk)
        risk_map[code] = float(risk)

    codes = df["code"].astype(str)
    df["b0_raw_rank"] = codes.map(rank_map)
    df["b0_lane"] = codes.map(lane_map).fillna("unknown")
    df["b0_evidence_balance"] = codes.map(ev_map).fillna(0.0)
    df["b0_risk_count"] = codes.map(risk_map).fillna(0.0)
    return df


@dataclass
class DSLPolicy:
    spec: dict[str, Any]

    def __post_init__(self) -> None:
        self.spec = validate_policy_spec(self.spec)
        self.policy_id = f"TRACK_D_DSL__{self.spec['policy_id']}"
        self.family = "track_d_dsl"
        self.spec_hash = self.spec["spec_hash"]
        self.fitted_state_hash = "none"

    def score_candidates(self, snapshot_df: pd.DataFrame) -> pd.DataFrame:
        if snapshot_df.empty:
            return pd.DataFrame()
        df = controlled_eligible(_augment_b0_features(snapshot_df))
        if df.empty:
            return df.assign(candidate_score=pd.Series(dtype=float), raw_rank=pd.Series(dtype=float))

        score = pd.Series(0.0, index=df.index, dtype=float)
        if self.spec["base"] == "b0_rank":
            score += _numeric_transform(-pd.to_numeric(df["b0_raw_rank"], errors="coerce"), "zscore")

        for term in self.spec["terms"]:
            if term["type"] == "linear":
                score += term["weight"] * _numeric_transform(df[term["feature"]], term["transform"])
            elif term["type"] == "threshold":
                masks = [_condition_mask(df, cond) for cond in term["conditions"]]
                mask = masks[0]
                for m in masks[1:]:
                    mask = (mask | m) if term["logic"] == "any" else (mask & m)
                score += np.where(mask, term["add"], 0.0)
            elif term["type"] == "interaction":
                transform = term["transform"]
                if transform not in NUMERIC_TRANSFORMS:
                    transform = "zscore"
                left = _numeric_transform(df[term["left"]], transform)
                right = _numeric_transform(df[term["right"]], transform)
                score += term["weight"] * left * right

        df = df.copy()
        df["candidate_score"] = score
        df["raw_rank"] = df["candidate_score"].rank(ascending=False, method="first")
        return df

    def allocate_industries(self, scored_df: pd.DataFrame) -> dict[str, int]:
        if scored_df.empty:
            return {}
        inds = scored_df["industry"].fillna("").astype(str).str.strip().str.lower()
        mode = self.spec["selector"]["industry_mode"]
        quota = 1 if mode == "distinct_1" else (2 if mode == "max_2_per_ind" else TOP_N)
        return {ind: quota for ind in inds.unique() if ind}

    def _capacity_limit(self, ordered: pd.DataFrame) -> int:
        if ordered.empty:
            return 0
        cap = self.spec["selector"]["capacity"]
        max_positions = int(cap["max_positions"])
        mode = cap["mode"]
        scores = ordered["candidate_score"].astype(float).tolist()

        if mode == "fixed":
            return min(max_positions, len(scores))
        if mode == "min_score":
            return min(max_positions, sum(x >= float(cap["min_score"]) for x in scores))
        if mode == "score_gap":
            min_score = float(cap.get("min_score", -1e9))
            if scores[0] < min_score:
                return 0
            limit = 1
            for i in range(1, min(max_positions, len(scores))):
                if scores[i] < min_score:
                    break
                if scores[i - 1] - scores[i] >= float(cap["gap"]):
                    break
                limit += 1
            return limit
        if mode == "top1_confidence":
            if len(scores) == 1:
                return 1 if scores[0] >= float(cap.get("min_score", -1e9)) else 0
            if scores[0] < float(cap.get("min_score", -1e9)):
                return 0
            return 1 if scores[0] - scores[1] >= float(cap["gap"]) else min(max_positions, len(scores))
        raise ValueError(mode)

    def pick_stocks(self, scored_df: pd.DataFrame, industry_quotas: dict[str, int]) -> list[str]:
        if scored_df.empty or not industry_quotas:
            return []
        ordered = scored_df.sort_values(["candidate_score", "code"], ascending=[False, True]).copy()
        cap = self._capacity_limit(ordered)
        if cap <= 0:
            return []

        selected: list[str] = []
        counts: dict[str, int] = {}
        for _, row in ordered.iterrows():
            if len(selected) >= cap:
                break
            ind = str(row["industry"]).strip().lower()
            quota = int(industry_quotas.get(ind, 0))
            if counts.get(ind, 0) >= quota:
                continue
            selected.append(str(row["code"]))
            counts[ind] = counts.get(ind, 0) + 1
        return selected


def deduplicate_policy_specs(specs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, str]] = []
    seen_hashes: dict[str, str] = {}
    seen_ids: dict[str, str] = {}
    for raw in specs:
        spec = validate_policy_spec(raw)
        h = spec["spec_hash"]
        pid = spec["policy_id"]
        if h in seen_hashes:
            dropped.append({"policy_id": pid, "duplicate_of": seen_hashes[h], "spec_hash": h})
            continue
        if pid in seen_ids:
            dropped.append({
                "policy_id": pid,
                "duplicate_of": pid,
                "spec_hash": h,
                "reason": "duplicate_policy_id_with_different_spec",
            })
            continue
        seen_hashes[h] = pid
        seen_ids[pid] = h
        kept.append(spec)
    return kept, dropped

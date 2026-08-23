from __future__ import annotations

import pandas as pd

from backtest.ibd_skill_iteration.core import (
    ReasonedCandidate,
    build_reasoning_skill_picks,
    rank_non_actionable_alpha_radar,
    rank_non_actionable_pullback_scout,
    rank_reasoning_candidates,
    rank_shadow_portfolio_top3,
    rank_signal_shadow_top3,
)
from backtest.ibd_skill_replay.core import compute_path_metrics, to_float


PULLBACK_RULES = {"ceiling_pullback", "pivot", "ma10_touch_confirm", "three_weeks_tight"}


def build_reasoning_pick_metric_rows(
    pool: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    *,
    snapshot_date: str,
    end_date: str,
    version: str = "v1",
) -> list[dict[str, object]]:
    picks = build_reasoning_skill_picks(pool, snapshot_date=snapshot_date, version=version)
    actionable_raw = rank_reasoning_candidates(pool, universe="actionable", version=version)
    non_actionable_alpha = rank_non_actionable_alpha_radar(pool, version=version)
    signal_shadow = rank_signal_shadow_top3(pool, version=version)
    shadow_portfolio = rank_shadow_portfolio_top3(pool, version=version)
    prefix = f"reasoning_{version}"
    lists = {
        f"{prefix}_priority_top3": [item for item in picks if item.final_group == "PRIORITY"][:3],
        f"{prefix}_actionable_raw_top5": actionable_raw[:5],
        f"{prefix}_alpha_radar_top5": [item for item in picks if item.final_group == "ALPHA_RADAR"][:5],
        f"{prefix}_signal_shadow_top3": signal_shadow,
        f"{prefix}_shadow_portfolio_top3": shadow_portfolio,
        f"{prefix}_non_actionable_alpha_radar_top10": non_actionable_alpha[:10],
        f"{prefix}_pullback_radar_top5": [
            item for item in picks if _is_pullback_rule(item.feature_values.get("ibd_candidate_rule"))
        ][:5],
    }
    if version == "v3":
        lists[f"{prefix}_pullback_scout_top10"] = rank_non_actionable_pullback_scout(pool, version=version)[:10]
    rows: list[dict[str, object]] = []
    for list_name, items in lists.items():
        for order, item in enumerate(items, 1):
            rows.append(_metric_row(list_name, order, item, prices, snapshot_date=snapshot_date, end_date=end_date))
    return rows


def find_quality_pullback_candidates(metrics: pd.DataFrame, *, limit: int = 10) -> pd.DataFrame:
    if metrics.empty:
        return metrics.copy()
    frame = metrics.copy()
    rule = frame["ibd_candidate_rule"].map(_is_pullback_rule)
    latest = pd.to_numeric(frame["latest_close_return_pct"], errors="coerce")
    gain = pd.to_numeric(frame["max_gain_pct"], errors="coerce")
    stop = frame["hit_stop_8pct"].astype(bool)
    winners = frame[rule & ~stop & latest.gt(0) & gain.gt(0)].copy()
    return winners.sort_values(
        ["latest_close_return_pct", "max_gain_pct", "max_drawdown_pct", "code"],
        ascending=[False, False, False, True],
    ).head(limit)


def build_non_actionable_hit_summary(skill_metrics: pd.DataFrame, review_oracle: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "skill",
        "entry_status",
        "picks",
        "review_oracle_top3_hits",
        "review_oracle_top5_hits",
        "review_oracle_top3_hit_rate",
        "review_oracle_top5_hit_rate",
        "review_oracle_top3_hit_codes",
        "review_oracle_top5_hit_codes",
    ]
    if skill_metrics.empty:
        return pd.DataFrame(columns=columns)
    non_actionable = skill_metrics[skill_metrics["entry_status"].astype(str).ne("ACTIONABLE")].copy()
    if non_actionable.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for (skill, status), group in non_actionable.groupby(["skill", "entry_status"], sort=True):
        top3_hits = []
        top5_hits = []
        for _, row in group.iterrows():
            oracle = review_oracle[review_oracle["snapshot_date"].eq(row["snapshot_date"])]
            top3 = set(oracle[oracle["oracle_rank"].le(3)]["code"].astype(str))
            top5 = set(oracle[oracle["oracle_rank"].le(5)]["code"].astype(str))
            code = str(row["code"])
            if code in top3:
                top3_hits.append(code)
            if code in top5:
                top5_hits.append(code)
        picks = len(group)
        rows.append(
            {
                "skill": skill,
                "entry_status": status,
                "picks": picks,
                "review_oracle_top3_hits": len(top3_hits),
                "review_oracle_top5_hits": len(top5_hits),
                "review_oracle_top3_hit_rate": len(top3_hits) / picks if picks else 0.0,
                "review_oracle_top5_hit_rate": len(top5_hits) / picks if picks else 0.0,
                "review_oracle_top3_hit_codes": ",".join(top3_hits),
                "review_oracle_top5_hit_codes": ",".join(top5_hits),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _metric_row(
    skill: str,
    order: int,
    item: ReasonedCandidate,
    prices: dict[str, pd.DataFrame],
    *,
    snapshot_date: str,
    end_date: str,
) -> dict[str, object]:
    metrics = compute_path_metrics(
        code=item.code,
        snapshot_date=snapshot_date,
        buy_price=to_float(item.feature_values.get("ibd_candidate_price")),
        snapshot_close=to_float(item.feature_values.get("latest_close")),
        price_bars=prices.get(item.code),
        end_date=end_date,
    )
    return {
        **item.feature_values,
        "snapshot_date": snapshot_date,
        "skill": skill,
        "pick_order": order,
        "code": item.code,
        "entry_status": item.entry_status,
        "lane": item.lane,
        "final_group": item.final_group,
        "reason_codes": ";".join(item.reason_codes),
        "risk_codes": ";".join(item.risk_codes),
        "industry": item.industry,
        "buy_price": metrics.buy_price,
        "latest_close": metrics.latest_close,
        "latest_close_return_pct": metrics.latest_close_return_pct,
        "max_gain_pct": metrics.max_gain_pct,
        "max_gain_date": metrics.max_gain_date,
        "max_drawdown_pct": metrics.max_drawdown_pct,
        "max_drawdown_date": metrics.max_drawdown_date,
        "hit_stop_8pct": metrics.hit_stop_8pct,
        "stop_8pct_date": metrics.stop_8pct_date,
        "path_source": metrics.source,
    }


def _is_pullback_rule(value: object) -> bool:
    return str(value or "").strip() in PULLBACK_RULES

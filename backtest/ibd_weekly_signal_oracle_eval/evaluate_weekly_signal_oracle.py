from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import math
import sys

import pandas as pd

sys.path.insert(0, str(Path.cwd()))

from backtest.ibd_skill_iteration.core import rank_reasoning_candidates
from backtest.ibd_skill_replay.core import compute_path_metrics, to_float
from backtest.ibd_skill_replay.run_ytd_replay import _load_price_cache
from backtest.ibd_weekly_signal_oracle_eval.price_cache import resolve_price_cache
import eps_pit.lookup as eps_lookup


ROOT = Path("backtest/ibd_skill_replay_pools")
OUT = Path("backtest/ibd_weekly_signal_oracle_eval")
END_DATE = "2026-08-14"
VERSION = "v3"


VARIANTS = {
    "v3_core_top3": {"industry_cover": False},
    "skill_industry_eps_known": {"industry_cover": True, "require_eps_known": True},
    "skill_industry_eps_known_no_dry_fail": {
        "industry_cover": True,
        "require_eps_known": True,
        "exclude_pullback_not_dry": True,
        "fill_relaxed": True,
    },
    "eps_pass_only": {"industry_cover": True, "require_eps_pass": True, "fill_relaxed": True},
    "clean_eps_pass_no_dry_no_geom_caution": {
        "industry_cover": True,
        "require_eps_pass": True,
        "exclude_pullback_not_dry": True,
        "exclude_geometry_caution": True,
        "fill_relaxed": True,
    },
    "fresh_or_constructive_eps_pass_clean": {
        "industry_cover": True,
        "require_eps_pass": True,
        "exclude_pullback_not_dry": True,
        "fresh_or_constructive": True,
        "fill_relaxed": True,
    },
    "fresh_demand_eps_pass_clean": {
        "industry_cover": True,
        "require_eps_pass": True,
        "exclude_pullback_not_dry": True,
        "exclude_geometry_caution": True,
        "fresh_demand_only": True,
        "fill_relaxed": True,
    },
    "risk_clean_eps_known": {
        "industry_cover": True,
        "require_eps_known": True,
        "exclude_pullback_not_dry": True,
        "exclude_geometry_caution": True,
        "fill_relaxed": True,
    },
    "research_fresh_demand_proximity_first": {
        "industry_cover": True,
        "research_order": "fresh_proximity",
    },
    "research_pullback_vcp_lane_interleave": {
        "industry_cover": True,
        "research_order": "pullback_interleave",
    },
    "research_proximity_structural_floor_guard": {
        "industry_cover": True,
        "exclude_pullback_not_dry": True,
        "exclude_geometry_caution": True,
        "research_order": "fresh_proximity",
    },
    "research_proximity_eps_known_floor_guard": {
        "industry_cover": True,
        "require_eps_known": True,
        "exclude_pullback_not_dry": True,
        "exclude_geometry_caution": True,
        "research_order": "fresh_proximity",
    },
    "research_proximity_eps_pass_floor_guard": {
        "industry_cover": True,
        "require_eps_pass": True,
        "exclude_pullback_not_dry": True,
        "exclude_geometry_caution": True,
        "research_order": "fresh_proximity",
    },
    "signal_shadow_top3": {
        "industry_cover": True,
        "allow_non_actionable": True,
        "allow_extended_from_buy_point": True,
    },
}


def pool_scope() -> dict[str, object]:
    dates = sorted(p.parent.name for p in ROOT.glob("*/breakout_follow_pool.csv"))
    return {
        "pool_weeks": len(dates),
        "first_week": dates[0] if dates else "n/a",
        "last_week": dates[-1] if dates else "n/a",
    }


@contextmanager
def eps_mode(enabled: bool):
    if enabled:
        yield
        return
    old = eps_lookup.get_signal_eps
    eps_lookup.get_signal_eps = lambda snapshot_date, code: None
    try:
        yield
    finally:
        eps_lookup.get_signal_eps = old


def effective_eps(snapshot: str, code: str, row_val: object, enabled: bool) -> float | None:
    val = to_float(row_val)
    if val is not None:
        return val
    if not enabled:
        return None
    return eps_lookup.get_signal_eps(str(snapshot), str(code))


def signal_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["signal"].astype(str).str.strip().str.lower().isin({"true", "1"})


def item_eps_state(item, row: pd.Series, enabled: bool) -> tuple[str, float | None]:
    snapshot = item.snapshot_date or str(row.get("snapshot_date", ""))
    eps = effective_eps(snapshot, item.code, row.get("eps_yoy_growth"), enabled)
    if eps is None:
        return "missing", None
    if eps >= 25:
        return "pass_25", eps
    return "known_below_25", eps


def item_allowed(item, row: pd.Series, enabled: bool, cfg: dict[str, object], *, relaxed: bool = False) -> bool:
    if item.entry_status != "ACTIONABLE" and not cfg.get("allow_non_actionable"):
        return False
    risks = set(item.risk_codes)
    if "clear_geometry_failure" in risks or "below_candidate_buy_point" in risks:
        return False
    if "extended_from_buy_point" in risks and not cfg.get("allow_extended_from_buy_point"):
        return False
    eps_state, _ = item_eps_state(item, row, enabled)
    risks = set(item.risk_codes)
    reasons = set(item.reason_codes)

    # EPS constraints are hard constraints for variants that test EPS; relaxed fill may only relax
    # lane/cleanliness filters, not EPS availability/pass semantics.
    if cfg.get("require_eps_known") and eps_state == "missing":
        return False
    if cfg.get("require_eps_pass") and eps_state != "pass_25":
        return False
    if relaxed:
        return True

    if cfg.get("exclude_pullback_not_dry") and "pullback_not_dry" in risks:
        return False
    if cfg.get("exclude_geometry_caution") and "geometry_caution_not_failure" in reasons:
        return False
    if cfg.get("fresh_demand_only") and item.lane != "fresh_demand_alpha":
        return False
    if cfg.get("fresh_or_constructive") and item.lane not in {"fresh_demand_alpha", "constructive_pullback"}:
        return False
    return True


def variant_items(items, pool: pd.DataFrame, enabled: bool, cfg: dict[str, object]) -> list:
    by_code = {str(row.get("code")): row for _, row in pool.iterrows()}
    items = _research_ordered_items(items, cfg)
    out = []
    covered = set()
    for item in items:
        row = by_code.get(item.code, pd.Series(dtype=object))
        industry_key = str(item.industry or "").strip().lower()
        if cfg.get("industry_cover") and industry_key and industry_key in covered:
            continue
        if not item_allowed(item, row, enabled, cfg):
            continue
        out.append(item)
        if cfg.get("industry_cover") and industry_key:
            covered.add(industry_key)
        if len(out) == 3:
            return out

    if len(out) < 3 and cfg.get("fill_relaxed"):
        chosen = {item.code for item in out}
        for item in items:
            if item.code in chosen:
                continue
            row = by_code.get(item.code, pd.Series(dtype=object))
            industry_key = str(item.industry or "").strip().lower()
            if cfg.get("industry_cover") and industry_key and industry_key in covered:
                continue
            if not item_allowed(item, row, enabled, cfg, relaxed=True):
                continue
            out.append(item)
            chosen.add(item.code)
            if cfg.get("industry_cover") and industry_key:
                covered.add(industry_key)
            if len(out) == 3:
                return out
    return out


def _research_ordered_items(items: list, cfg: dict[str, object]) -> list:
    order = cfg.get("research_order")
    if order == "fresh_proximity":
        return sorted(items, key=_fresh_proximity_key)
    if order == "pullback_interleave":
        return sorted(items, key=_pullback_interleave_key)
    return items


def _fresh_proximity_key(item) -> tuple:
    cur = to_float(item.feature_values.get("current_vs_ibd_candidate_pct"))
    entry_vol = to_float(item.feature_values.get("ibd_entry_volume_ratio"))
    reasons = set(item.reason_codes)
    risks = set(item.risk_codes)
    evidence_balance = _evidence_balance(item)
    return (
        item.sort_key[0],
        0 if item.entry_status == "ACTIONABLE" else 1,
        0 if item.lane == "fresh_demand_alpha" else 1,
        -evidence_balance,
        _positive_buy_point_distance(cur),
        0 if "geometry_caution_not_failure" not in reasons else 1,
        0 if entry_vol is not None and entry_vol >= 1.5 else 1,
        len(risks),
        item.code,
    )


def _pullback_interleave_key(item) -> tuple:
    cur = to_float(item.feature_values.get("current_vs_ibd_candidate_pct"))
    rule = str(item.feature_values.get("ibd_candidate_rule") or "").strip()
    reasons = set(item.reason_codes)
    risks = set(item.risk_codes)
    is_pullback = rule in {"ceiling_pullback", "pivot", "ma10_touch_confirm", "three_weeks_tight"}
    evidence_balance = _evidence_balance(item)
    return (
        item.sort_key[0],
        0 if item.entry_status == "ACTIONABLE" else 1,
        -evidence_balance,
        0 if is_pullback and "dry_pullback" in reasons else 1,
        0 if item.lane in {"fresh_demand_alpha", "constructive_pullback"} else 1,
        _positive_buy_point_distance(cur),
        len(risks),
        item.code,
    )


def _evidence_balance(item) -> int:
    positive_codes = {
        "near_buy_point",
        "volume_confirms_breakout",
        "eps_acceleration_support",
        "weekly_volume_follow_through",
        "near_52w_high",
        "pullback_structure",
        "dry_pullback",
    }
    negative_codes = {
        "freshness_missing",
        "below_candidate_buy_point",
        "extended_from_buy_point",
        "entry_volume_missing",
        "entry_volume_below_standard",
        "pullback_not_dry",
    }
    return sum(code in item.reason_codes for code in positive_codes) - sum(code in item.risk_codes for code in negative_codes)


def _positive_buy_point_distance(value: float | None) -> float:
    if value is None or value < 0:
        return 999.0
    return float(value)


def fmt_pct(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2f}%"


def build_signal_universe(pool: pd.DataFrame, snapshot: str, prices: dict[str, pd.DataFrame], enabled: bool) -> list[dict]:
    rows = []
    for idx, row in pool[signal_mask(pool)].iterrows():
        code = str(row.get("code", "")).strip()
        metrics = compute_path_metrics(
            code=code,
            snapshot_date=snapshot,
            buy_price=to_float(row.get("ibd_candidate_price")),
            snapshot_close=to_float(row.get("latest_close")),
            price_bars=prices.get(code),
            end_date=END_DATE,
        )
        eps = effective_eps(snapshot, code, row.get("eps_yoy_growth"), enabled)
        rows.append(
            {
                "snapshot_date": snapshot,
                "row_index": idx,
                "code": code,
                "entry_status": str(row.get("ibd_entry_status", "")).strip().upper(),
                "rule": row.get("ibd_candidate_rule"),
                "industry": row.get("industry"),
                "eps_state": "missing" if eps is None else ("pass_25" if eps >= 25 else "known_below_25"),
                "latest_return_pct": metrics.latest_close_return_pct,
                "max_gain_pct": metrics.max_gain_pct,
                "max_drawdown_pct": metrics.max_drawdown_pct,
                "hit_stop_8pct": metrics.hit_stop_8pct,
                "path_source": metrics.source,
            }
        )
    return rows


def add_oracle_ranks(universe: pd.DataFrame) -> pd.DataFrame:
    frame = universe.copy()
    frame["valid_path"] = pd.to_numeric(frame["latest_return_pct"], errors="coerce").notna()
    valid = frame[frame["valid_path"]].copy()
    valid["latest_rank"] = valid.groupby("snapshot_date")["latest_return_pct"].rank(
        method="first", ascending=False
    )
    valid = valid.sort_values(["snapshot_date", "max_gain_pct", "latest_return_pct", "code"], ascending=[True, False, False, True])
    valid["gain_rank"] = valid.groupby("snapshot_date").cumcount() + 1
    valid = valid.sort_values(["snapshot_date", "latest_return_pct", "max_drawdown_pct", "code"], ascending=[True, True, True, True])
    valid["loss_rank"] = valid.groupby("snapshot_date").cumcount() + 1
    counts = valid.groupby("snapshot_date")["code"].transform("count")
    valid["signal_valid_count"] = counts
    return frame.merge(
        valid[["snapshot_date", "code", "latest_rank", "gain_rank", "loss_rank", "signal_valid_count"]],
        on=["snapshot_date", "code"],
        how="left",
    )


def evaluate(
    enabled: bool,
    *,
    price_cache: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    resolved_price_cache = resolve_price_cache(price_cache)
    prices = _load_price_cache(resolved_price_cache)
    universe_rows = []
    pick_rows = []
    with eps_mode(enabled):
        for pool_path in sorted(ROOT.glob("*/breakout_follow_pool.csv")):
            snapshot = pool_path.parent.name
            pool = pd.read_csv(pool_path, encoding="utf-8-sig")
            if pool.empty:
                continue
            universe_rows.extend(build_signal_universe(pool, snapshot, prices, enabled))
            ranked_items = rank_reasoning_candidates(pool, universe="review", version=VERSION)
            for item in ranked_items:
                item.snapshot_date = snapshot
            by_code = {str(row.get("code")): row for _, row in pool.iterrows()}
            for variant, cfg in VARIANTS.items():
                selected = variant_items(ranked_items, pool, enabled, cfg)
                for order, item in enumerate(selected, 1):
                    row = by_code.get(item.code, pd.Series(dtype=object))
                    eps_state, eps = item_eps_state(item, row, enabled)
                    metrics = compute_path_metrics(
                        code=item.code,
                        snapshot_date=snapshot,
                        buy_price=to_float(item.feature_values.get("ibd_candidate_price")),
                        snapshot_close=to_float(item.feature_values.get("latest_close")),
                        price_bars=prices.get(item.code),
                        end_date=END_DATE,
                    )
                    pick_rows.append(
                        {
                            "eps_mode": "with_eps" if enabled else "no_eps",
                            "snapshot_date": snapshot,
                            "variant": variant,
                            "pick_order": order,
                            "code": item.code,
                            "entry_status": item.entry_status,
                            "lane": item.lane,
                            "industry": item.industry,
                            "eps_state": eps_state,
                            "eps_yoy_growth": eps,
                            "reason_codes": ";".join(item.reason_codes),
                            "risk_codes": ";".join(item.risk_codes),
                            "latest_return_pct": metrics.latest_close_return_pct,
                            "max_gain_pct": metrics.max_gain_pct,
                            "max_drawdown_pct": metrics.max_drawdown_pct,
                            "hit_stop_8pct": metrics.hit_stop_8pct,
                            "path_source": metrics.source,
                        }
                    )

    universe = add_oracle_ranks(pd.DataFrame(universe_rows))
    picks = pd.DataFrame(pick_rows)
    if picks.empty:
        return universe, picks, pd.DataFrame(), pd.DataFrame()

    picks = picks.merge(
        universe[["snapshot_date", "code", "latest_rank", "gain_rank", "loss_rank", "signal_valid_count"]],
        on=["snapshot_date", "code"],
        how="left",
    )
    picks["hit_latest_top3"] = picks["latest_rank"].le(3)
    picks["hit_latest_top5"] = picks["latest_rank"].le(5)
    picks["hit_gain_top5"] = picks["gain_rank"].le(5)
    picks["hit_loss_bottom3"] = picks["loss_rank"].le(3)
    picks["hit_loss_bottom5"] = picks["loss_rank"].le(5)

    weekly_rows = []
    for (mode, variant, snapshot), group in picks.groupby(["eps_mode", "variant", "snapshot_date"], sort=True):
        weekly_rows.append(
            {
                "eps_mode": mode,
                "variant": variant,
                "snapshot_date": snapshot,
                "picks": len(group),
                "avg_latest_return_pct": pd.to_numeric(group["latest_return_pct"], errors="coerce").mean(),
                "median_latest_return_pct": pd.to_numeric(group["latest_return_pct"], errors="coerce").median(),
                "avg_max_gain_pct": pd.to_numeric(group["max_gain_pct"], errors="coerce").mean(),
                "worst_latest_return_pct": pd.to_numeric(group["latest_return_pct"], errors="coerce").min(),
                "hit_latest_top3_count": int(group["hit_latest_top3"].sum()),
                "hit_latest_top5_count": int(group["hit_latest_top5"].sum()),
                "hit_gain_top5_count": int(group["hit_gain_top5"].sum()),
                "hit_loss_bottom3_count": int(group["hit_loss_bottom3"].sum()),
                "hit_loss_bottom5_count": int(group["hit_loss_bottom5"].sum()),
                "stop_8pct_count": int(group["hit_stop_8pct"].sum()),
                "codes": ",".join(group.sort_values("pick_order")["code"]),
            }
        )
    weekly = pd.DataFrame(weekly_rows)
    summary_rows = []
    for (mode, variant), group in weekly.groupby(["eps_mode", "variant"], sort=True):
        pick_subset = picks[picks["eps_mode"].eq(mode) & picks["variant"].eq(variant)]
        weeks = group["snapshot_date"].nunique()
        pick_count = int(group["picks"].sum())
        week_top5_hit_rate = float(group["hit_latest_top5_count"].gt(0).mean()) if weeks else 0.0
        week_gain_top5_hit_rate = float(group["hit_gain_top5_count"].gt(0).mean()) if weeks else 0.0
        week_bottom5_hit_rate = float(group["hit_loss_bottom5_count"].gt(0).mean()) if weeks else 0.0
        week_stop_rate = float(group["stop_8pct_count"].gt(0).mean()) if weeks else 0.0
        pick_top5_rate = float(pick_subset["hit_latest_top5"].mean()) if pick_count else 0.0
        pick_bottom5_rate = float(pick_subset["hit_loss_bottom5"].mean()) if pick_count else 0.0
        pick_stop_rate = float(pick_subset["hit_stop_8pct"].mean()) if pick_count else 0.0
        median_week_return = float(group["avg_latest_return_pct"].median()) if weeks else math.nan
        avg_week_return = float(group["avg_latest_return_pct"].mean()) if weeks else math.nan
        median_worst = float(group["worst_latest_return_pct"].median()) if weeks else math.nan
        score = (
            week_top5_hit_rate * 3.0
            + week_gain_top5_hit_rate
            + median_week_return / 100.0
            - week_bottom5_hit_rate * 1.5
            - week_stop_rate
            - pick_bottom5_rate * 0.8
            - pick_stop_rate * 0.5
        )
        summary_rows.append(
            {
                "eps_mode": mode,
                "variant": variant,
                "weeks": weeks,
                "picks": pick_count,
                "median_week_avg_latest_return_pct": median_week_return,
                "avg_week_avg_latest_return_pct": avg_week_return,
                "median_week_worst_pick_return_pct": median_worst,
                "week_latest_top5_hit_rate": week_top5_hit_rate,
                "week_gain_top5_hit_rate": week_gain_top5_hit_rate,
                "pick_latest_top5_rate": pick_top5_rate,
                "week_bottom5_hit_rate": week_bottom5_hit_rate,
                "pick_bottom5_rate": pick_bottom5_rate,
                "week_stop_rate": week_stop_rate,
                "pick_stop_rate": pick_stop_rate,
                "score": score,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["eps_mode", "score"], ascending=[True, False])
    return universe, picks, weekly, summary


def markdown_table(frame: pd.DataFrame) -> list[str]:
    return frame.to_markdown(index=False).splitlines() if not frame.empty else ["No rows."]


def render_mode_report(suffix: str, universe: pd.DataFrame, weekly: pd.DataFrame, summary: pd.DataFrame) -> str:
    label = "含 EPS" if suffix == "with_eps" else "无 EPS"
    scope = pool_scope()
    lines = [
        f"# 按周 Signal Oracle 推荐质量评估 - {label}",
        "",
        f"- Pool 周范围: {scope['first_week']} 至 {scope['last_week']}；pool files: {scope['pool_weeks']}；路径收益截至 {END_DATE}",
        "- Oracle universe: 每周所有 `signal == True` 且路径收益可计算的标的；winner/loser 在每周内排序，不跨周合并。",
        "- Big winner: 当周 latest_return Top5；Opportunity winner: 当周 max_gain Top5；Big loser: 当周 latest_return Bottom5 或命中 -8% stop。",
        "",
        "## Universe 覆盖",
        "",
    ]
    valid = universe[universe["valid_path"]]
    lines.append(f"- Signal rows: {len(universe)}；valid path rows: {len(valid)}；weeks: {valid['snapshot_date'].nunique() if not valid.empty else 0}")
    lines.extend(["", "## Variant 总结", ""])
    show = summary.copy()
    for column in [
        "median_week_avg_latest_return_pct",
        "avg_week_avg_latest_return_pct",
        "median_week_worst_pick_return_pct",
    ]:
        show[column] = show[column].map(fmt_pct)
    for column in [
        "week_latest_top5_hit_rate",
        "week_gain_top5_hit_rate",
        "pick_latest_top5_rate",
        "week_bottom5_hit_rate",
        "pick_bottom5_rate",
        "week_stop_rate",
        "pick_stop_rate",
    ]:
        show[column] = show[column].map(lambda value: "n/a" if pd.isna(value) else f"{float(value) * 100:.1f}%")
    show["score"] = show["score"].map(lambda value: f"{float(value):.3f}")
    lines.extend(markdown_table(show))
    if not summary.empty:
        best = summary.iloc[0]
        best_weekly = weekly[weekly["variant"].eq(best["variant"])].copy()
        lines.extend(["", f"## Best Variant: `{best['variant']}`", ""])
        columns = [
            "snapshot_date",
            "picks",
            "codes",
            "avg_latest_return_pct",
            "worst_latest_return_pct",
            "hit_latest_top5_count",
            "hit_gain_top5_count",
            "hit_loss_bottom5_count",
            "stop_8pct_count",
        ]
        show_weekly = best_weekly[columns].copy()
        for column in ["avg_latest_return_pct", "worst_latest_return_pct"]:
            show_weekly[column] = show_weekly[column].map(fmt_pct)
        lines.extend(markdown_table(show_weekly))
    return "\n".join(lines) + "\n"


def render_run_log(combined: pd.DataFrame, *, price_cache: Path) -> str:
    scope = pool_scope()
    return "\n".join(
        [
            "# Weekly Signal Oracle Evaluation Run Log",
            "",
            "## Purpose",
            "",
            "按周评估 skill 推荐质量。每周先用所有 `signal == True` 标的建立独立 oracle，再评估推荐列表是否命中当周大赢家、避开当周大输家，并比较 EPS-blind 与 EPS-enriched 两种输入模式。",
            "",
            "## Step Logic",
            "",
            f"1. 固定输入：`backtest/ibd_skill_replay_pools/*/breakout_follow_pool.csv` 的 {scope['pool_weeks']} 个成功 replay pool，范围 {scope['first_week']} 至 {scope['last_week']}；不修改 pool。",
            f"2. 固定收益窗口：从每个 `snapshot_date` 的 `ibd_candidate_price` 到 `{END_DATE}`，使用 `{price_cache}` 计算 latest return、max gain、max drawdown、-8% stop。",
            "3. 每周 universe：该周所有 `signal == True` 行；ACTIONABLE 与非 ACTIONABLE 都进入 winner/loser oracle。",
            "4. 每周 winner/loser：latest return Top3/Top5、max gain Top5、latest return Bottom3/Bottom5、以及是否触发 -8% stop。所有排名只在同一周内比较。",
            "5. 推荐生成：对同一 pool 调用 `rank_reasoning_candidates(..., universe='review', version='v3')`，比较现有 ACTIONABLE variants 与 `signal_shadow_top3`（所有 signal，保留 entry_status，最多 3 只的审计层；非正式推荐）。",
            "6. EPS-blind 模式：在内存中关闭 `eps_pit.lookup.get_signal_eps`，所有 CSV 空 EPS 保持 missing。",
            "7. EPS-enriched 模式：允许 `eps_pit.lookup.get_signal_eps(snapshot, code)` 作为 point-in-time 补源，按用户要求先假设其正确。",
            "8. Variant 比较：测试行业覆盖、EPS 已知、EPS>=25、排除 `pullback_not_dry`、排除 `geometry_caution_not_failure`、Fresh Demand/Constructive Pullback 限定，以及 RD candidate 排序假设。",
            "9. 评分函数：`3*周Top5命中率 + 周max-gain Top5命中率 + 周中位平均收益/100 - 1.5*周Bottom5暴露率 - 周stop暴露率 - 0.8*pick Bottom5率 - 0.5*pick stop率`。",
            "10. 规则沉淀：只采用跨周稳定的证据顺序和风险约束；禁止把具体 ticker、日期、收益率、中位数或命中率写成新门槛。",
            "",
            "## Bug Fix During Evaluation",
            "",
            "- 第一版临时脚本的 `fill_relaxed` 会错误放松 EPS 硬约束；已修正为 fallback 只能放松 lane/cleanliness，不能放松 `require_eps_known` 或 `require_eps_pass`。",
            "",
            "## Current Best Rows",
            "",
            *markdown_table(combined.sort_values("score", ascending=False).head(12)),
            "",
        ]
    ) + "\n"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    price_cache = resolve_price_cache(None)
    outputs = []
    for enabled in [False, True]:
        suffix = "with_eps" if enabled else "no_eps"
        universe, picks, weekly, summary = evaluate(enabled, price_cache=price_cache)
        universe.to_csv(OUT / f"{suffix}_signal_universe_oracle.csv", index=False)
        picks.to_csv(OUT / f"{suffix}_variant_picks.csv", index=False)
        weekly.to_csv(OUT / f"{suffix}_weekly_variant_metrics.csv", index=False)
        summary.to_csv(OUT / f"{suffix}_variant_summary.csv", index=False)
        (OUT / f"{suffix}_weekly_report.md").write_text(
            render_mode_report(suffix, universe, weekly, summary),
            encoding="utf-8",
        )
        outputs.append(summary)

    combined = pd.concat(outputs, ignore_index=True, sort=False).sort_values("score", ascending=False)
    combined.to_csv(OUT / "combined_variant_summary.csv", index=False)
    run_log = render_run_log(combined, price_cache=price_cache)
    (OUT / "run_log.md").write_text(run_log, encoding="utf-8")
    (OUT / "combined_weekly_iteration_report.md").write_text(run_log, encoding="utf-8")
    print(f"OUT {OUT}")
    print(combined.head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

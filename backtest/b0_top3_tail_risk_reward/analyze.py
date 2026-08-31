from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE.parent / "b0_top3_quality_audit" / "output"
OUT = HERE / "output"

RANK_DETAIL = UPSTREAM / "b0_rank_position_weekly_detail.csv"
THREE_TIER = UPSTREAM / "three_tier_weekly_comparison.csv"
PATH_DETAIL = UPSTREAM / "b0_path_quality_to_asof.csv"


def _truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().eq("true")


def _tail_stats(values: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().sort_values().to_numpy(dtype=float)
    n = len(x)
    if n == 0:
        return {}
    k10 = max(1, math.ceil(n * 0.10))
    k20 = max(1, math.ceil(n * 0.20))
    c10, t10 = float(x[:k10].mean()), float(x[-k10:].mean())
    c20, t20 = float(x[:k20].mean()), float(x[-k20:].mean())
    return {
        "weeks": n,
        "mean_return_pct": float(x.mean()),
        "median_return_pct": float(np.median(x)),
        "p10_return_pct": float(np.quantile(x, 0.10)),
        "p90_return_pct": float(np.quantile(x, 0.90)),
        "worst_return_pct": float(x[0]),
        "best_return_pct": float(x[-1]),
        "negative_week_rate_pct": float((x < 0).mean() * 100),
        "cvar10_pct": c10,
        "top10_mean_pct": t10,
        "tail_ratio10": float(t10 / abs(c10)) if c10 < 0 else np.nan,
        "cvar20_pct": c20,
        "top20_mean_pct": t20,
        "tail_ratio20": float(t20 / abs(c20)) if c20 < 0 else np.nan,
        "tail_n10": k10,
        "tail_n20": k20,
    }


def run() -> None:
    rank = pd.read_csv(RANK_DETAIL)
    three = pd.read_csv(THREE_TIER)
    path = pd.read_csv(PATH_DETAIL)
    full = rank[_truthy(rank["is_3picks"])].copy()

    summary_rows: list[dict] = []
    internal_rows: list[dict] = []
    weekly_rows: list[dict] = []

    for h in (1, 2, 4):
        common = full[_truthy(full[f"w{h}_common_valid"])].copy()
        dates = set(common["snapshot_date"].astype(str))
        stop = three[
            three["snapshot_date"].astype(str).isin(dates)
            & (pd.to_numeric(three["b0_picks_count"], errors="coerce") == 3)
        ].set_index("snapshot_date")[f"l2_w{h}_ret"]

        local: list[dict] = []
        for _, r in common.iterrows():
            picks = np.array([r[f"r{i}_w{h}_return_pct"] for i in (1, 2, 3)], dtype=float)
            port = float(picks.mean())
            neg, pos = picks[picks < 0], picks[picks > 0]
            worst, best = float(picks.min()), float(picks.max())
            worst_i = int(picks.argmin())
            others = np.delete(picks, worst_i)
            rec = {
                "snapshot_date": str(r["snapshot_date"]),
                "horizon": f"W{h}",
                "rank1_code": r["r1_code"],
                "rank2_code": r["r2_code"],
                "rank3_code": r["r3_code"],
                "rank1_return_pct": float(picks[0]),
                "rank2_return_pct": float(picks[1]),
                "rank3_return_pct": float(picks[2]),
                "raw_portfolio_return_pct": port,
                "stop_capped_portfolio_return_pct": float(stop.get(r["snapshot_date"], np.nan)),
                "worst_pick_return_pct": worst,
                "best_pick_return_pct": best,
                "loss_concentration": float(abs(worst) / np.abs(neg).sum()) if len(neg) else np.nan,
                "gain_concentration": float(best / pos.sum()) if len(pos) else np.nan,
                "one_pick_ruins_portfolio": bool(port < 0 and others.mean() > 0),
            }
            local.append(rec)

        local_df = pd.DataFrame(local)
        raw = _tail_stats(local_df["raw_portfolio_return_pct"])
        stop_stats = _tail_stats(local_df["stop_capped_portfolio_return_pct"])
        summary_rows += [
            {"scope": "full_top3_common_support", "horizon": f"W{h}", "return_mode": "raw", **raw},
            {"scope": "full_top3_common_support", "horizon": f"W{h}", "return_mode": "stop_capped", **stop_stats},
        ]

        k = raw["tail_n10"]
        ordered = local_df.sort_values("raw_portfolio_return_pct")
        left_dates = set(ordered.head(k)["snapshot_date"])
        right_dates = set(ordered.tail(k)["snapshot_date"])
        local_df["is_left10_raw"] = local_df["snapshot_date"].isin(left_dates)
        local_df["is_right10_raw"] = local_df["snapshot_date"].isin(right_dates)
        weekly_rows.extend(local_df.to_dict("records"))

        neg_weeks = local_df[local_df["raw_portfolio_return_pct"] < 0]
        pos_weeks = local_df[local_df["raw_portfolio_return_pct"] > 0]
        internal_rows.append({
            "horizon": f"W{h}",
            "weeks": len(local_df),
            "negative_weeks": len(neg_weeks),
            "positive_weeks": len(pos_weeks),
            "one_pick_ruins_weeks": int(local_df["one_pick_ruins_portfolio"].sum()),
            "one_pick_ruins_rate_all_pct": float(local_df["one_pick_ruins_portfolio"].mean() * 100),
            "one_pick_ruins_rate_negative_pct": float(local_df["one_pick_ruins_portfolio"].sum() / len(neg_weeks) * 100) if len(neg_weeks) else np.nan,
            "mean_loss_concentration_negative_weeks": float(neg_weeks["loss_concentration"].mean()),
            "mean_gain_concentration_positive_weeks": float(pos_weeks["gain_concentration"].mean()),
            "left10_mean_worst_pick_loss_concentration": float(local_df[local_df["is_left10_raw"]]["loss_concentration"].mean()),
            "right10_mean_best_pick_gain_concentration": float(local_df[local_df["is_right10_raw"]]["gain_concentration"].mean()),
        })

    keys = {
        (str(r["snapshot_date"]), str(r[f"r{i}_code"]))
        for _, r in full.iterrows()
        for i in (1, 2, 3)
    }
    selected = path[
        path.apply(lambda r: (str(r["snapshot_date"]), str(r["code"])) in keys, axis=1)
    ].copy()

    def rate(col: str) -> tuple[int, float]:
        n = int(_truthy(selected[col]).sum())
        return n, float(n / len(selected) * 100)

    sb_n, sb_r = rate("stop8_before_profit20")
    pb_n, pb_r = rate("profit20_before_stop8")
    se_n, se_r = rate("stop_8_hit_ever")
    gs_n, gs_r = rate("gap_stop")
    p20_n, p20_r = rate("profit20_hit")
    path_summary = pd.DataFrame([{
        "scope": "full_top3_25_weeks_75_picks",
        "picks": len(selected),
        "stop8_before_profit20_count": sb_n,
        "stop8_before_profit20_rate_pct": sb_r,
        "profit20_before_stop8_count": pb_n,
        "profit20_before_stop8_rate_pct": pb_r,
        "stop8_ever_count": se_n,
        "stop8_ever_rate_pct": se_r,
        "gap_stop_count": gs_n,
        "gap_stop_rate_pct": gs_r,
        "profit20_hit_count": p20_n,
        "profit20_hit_rate_pct": p20_r,
        "path_odds_profit20_before_stop_vs_stop_before_profit20": pb_n / sb_n if sb_n else np.nan,
    }])

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(OUT / "tail_summary.csv", index=False)
    pd.DataFrame(internal_rows).to_csv(OUT / "internal_concentration_summary.csv", index=False)
    pd.DataFrame(weekly_rows).to_csv(OUT / "weekly_tail_detail.csv", index=False)
    path_summary.to_csv(OUT / "path_summary.csv", index=False)


if __name__ == "__main__":
    run()

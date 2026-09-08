"""Frozen-family terminal-return triage, not strategy search or portfolio P&L."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from .r6_stability import LABELS, numeric, purged_before
from .stop_risk_validation_r5 import build_risk_family_rule, risk_family_mask

FAMILIES = ("deep_pullback", "extended_vs_candidate", "deep_pullback_and_extended", "high_pct_above_ceiling")
FEATURES = ("pullback_pct", "current_vs_ibd_candidate_pct", "pct_above_ceiling")
HORIZONS = ("w1", "w2", "w3", "w4")
PANELS = ("all_entries", "nonoverlap_w4")
MIN_TRAIN_QUARTERS = 6
BLOCK_WEEKS = 8
BOOTSTRAP_DRAWS = 2000
SEED = 42


def finite_mean(values) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def safe_ratio(a, b) -> float | None:
    return float(a / b) if b else None


def baseline_admission(frame: pd.DataFrame) -> np.ndarray:
    """Shared earliest-entry schedule: a ticker stays reserved through W4 close.

    No policy gets an extra admission after a veto. This isolates the accounting
    comparison; it is not a capital-constrained, policy-dependent execution engine.
    """
    admitted = np.zeros(len(frame), dtype=bool)
    busy: dict[str, pd.Timestamp] = {}
    ordered = frame.assign(_position=np.arange(len(frame))).sort_values(
        ["entry_date", "snapshot_date", "code"], kind="stable")
    for row in ordered.to_dict("records"):
        if row["code"] not in busy or row["entry_date"] > busy[row["code"]]:
            admitted[row["_position"]] = True
            busy[row["code"]] = row["exit_date_w4"]
    return admitted


def freeze_families(frame: pd.DataFrame, calendar: list[str]):
    frozen, flags = [], {}
    quarters = frame.snapshot_date.dt.to_period("Q").astype(str)
    for quarter in calendar[MIN_TRAIN_QUARTERS:]:
        past = purged_before(frame, pd.Period(quarter, freq="Q").start_time)
        test = frame.loc[quarters == quarter]
        surface = past[["snapshot_date", "code", "exit_date_w3", *FEATURES]].to_csv(index=False)
        for family in FAMILIES:
            rule = build_risk_family_rule(past, family)
            fields = [c[0] for c in rule["conditions"]] if rule else []
            known = np.logical_and.reduce([np.isfinite(numeric(test[f])) for f in fields]) if fields else np.zeros(len(test), bool)
            mask = risk_family_mask(test, rule) if rule else np.zeros(len(test), bool)
            frozen.append(dict(quarter=quarter, policy=family, rule=rule,
                status="FROZEN" if rule else "NO_TRAIN_FEATURE_SUPPORT", training_rows=len(past),
                training_surface_sha256=hashlib.sha256(surface.encode()).hexdigest()))
            flags[(quarter, family)] = (mask, known)
    return frozen, flags


def account_week(frame: pd.DataFrame, flagged: np.ndarray, horizon: str,
                 cost_bps: float, known: np.ndarray | None = None):
    """Equal initial candidate slots; removed slots earn zero, never reinvested.

    Random is the exact expectation over uniformly removing exactly m of n
    candidates that week. This avoids simulation error and never shrinks N.
    Return fields are decimal terminal mark-to-market facts, not stop executions.
    """
    flagged = np.asarray(flagged, bool)
    known = np.ones(len(frame), bool) if known is None else np.asarray(known, bool)
    if len(flagged) != len(frame) or len(known) != len(frame) or (flagged & ~known).any():
        raise ValueError("invalid flag/feature-support mask")
    if not np.isfinite(cost_bps) or cost_bps < 0 or horizon not in HORIZONS:
        raise ValueError("invalid cost or horizon")
    n, m = len(frame), int(flagged.sum())
    if not n:
        raise ValueError("account_week requires a nonempty snapshot")
    returns = numeric(frame[f"return_{horizon}"])
    if not np.isfinite(returns).all() or (returns < -1).any():
        raise ValueError("terminal returns must be finite decimal returns >= -1")
    cost = cost_bps / 10000
    net = returns - cost
    retained = ~flagged
    baseline = float(net.mean())
    veto = float(net[retained].sum() / n)
    random = float((n-m) / n * baseline)
    valid_path = numeric(frame.ambiguous_3w) == 0
    stops = numeric(frame.stop_first_3w) == 1
    winners = numeric(frame.fast_winner_3w) == 1
    a, b = flagged & valid_path, retained & known & valid_path
    stop_lift = float(stops[a].mean() - stops[b].mean()) if a.any() and b.any() else None
    result: dict[str, Any] = dict(n=n, flagged_n=m, retained_n=n-m, known_n=int(known.sum()),
        unknown_n=int((~known).sum()), coverage=m/n, known_coverage=float(known.mean()),
        baseline_gross=float(returns.mean()), baseline_net=baseline,
        veto_cash_net=veto, random_cash_net=random, cash_delta=veto-baseline,
        cash_delta_gross=float(-returns[flagged].sum()/n), incremental_vs_random=veto-random,
        avoided_gross_loss=float(-np.minimum(returns[flagged], 0).sum()/n),
        foregone_gross_gain=float(np.maximum(returns[flagged], 0).sum()/n), saved_cost=m/n*cost,
        flagged_mean=finite_mean(returns[flagged]), retained_mean=finite_mean(returns[retained]),
        known_retained_mean=finite_mean(returns[retained & known]),
        known_matched_return_spread=(float(returns[retained & known].mean()-returns[flagged].mean())
                                    if flagged.any() and (retained & known).any() else None),
        stop_lift=stop_lift, matched=int(a.any() and b.any()),
        stop_capture=safe_ratio((stops & a).sum(), (stops & valid_path).sum()),
        winner_loss=safe_ratio((winners & a).sum(), (winners & valid_path).sum()),
        break_even_cost_bps=float(returns[flagged].mean()*10000) if m else None)
    parts = []
    for label in LABELS:
        group = numeric(frame[label]) == 1
        removed = group & flagged
        parts.append(dict(path_label=label, baseline_n=int(group.sum()), flagged_n=int(removed.sum()),
            avoided_gross_loss=float(-np.minimum(returns[removed], 0).sum()/n),
            foregone_gross_gain=float(np.maximum(returns[removed], 0).sum()/n),
            saved_cost=float(removed.sum()/n*cost),
            cash_delta_contribution=float(-net[removed].sum()/n)))
    return result, parts


def stability(weekly: pd.DataFrame) -> dict:
    """Descriptive time-block intervals; not post-search significance tests.

    Missing snapshot weeks remain NaN on the W-FRI period grid, not invented
    zero-return observations. A block spans eight calendar weeks, not eight rows.
    """
    keys = ("incremental_mean", "incremental_median", "positive_week_fraction",
            "block_ci_low", "block_ci_high", "leave_one_quarter_out_min",
            "leave_one_quarter_out_max", "without_best_week", "without_worst_week")
    out = {k: None for k in keys}
    out.update(calendar_quarters_observed=0, independent_confirmation=False)
    if weekly.empty:
        return out
    weekly = weekly.sort_values("snapshot_date").copy()
    values = weekly.incremental_vs_random.to_numpy(float)
    dates = pd.to_datetime(weekly.snapshot_date)
    out.update(incremental_mean=float(values.mean()), incremental_median=float(np.median(values)),
        positive_week_fraction=float((values > 0).mean()),
        calendar_quarters_observed=int(weekly.quarter.nunique()),
        without_best_week=float(np.delete(values, np.argmax(values)).mean()) if len(values) > 1 else None,
        without_worst_week=float(np.delete(values, np.argmin(values)).mean()) if len(values) > 1 else None)
    leave = [g.incremental_vs_random.mean() for q in weekly.quarter.unique()
             if not (g := weekly.loc[weekly.quarter != q]).empty]
    if leave:
        out.update(leave_one_quarter_out_min=float(min(leave)), leave_one_quarter_out_max=float(max(leave)))
    weeks = dates.dt.to_period("W-FRI")
    grid = pd.period_range(weeks.min(), weeks.max(), freq="W-FRI")
    if weeks.duplicated().any():
        raise ValueError("multiple snapshots in one weekly calendar period")
    series = pd.Series(values, index=weeks).reindex(grid).to_numpy()
    if len(series) >= 2*BLOCK_WEEKS:
        rng = np.random.default_rng(SEED)
        count = int(np.ceil(len(series)/BLOCK_WEEKS))
        starts = rng.integers(0, len(series)-BLOCK_WEEKS+1, size=(BOOTSTRAP_DRAWS, count))
        positions = (starts[:, :, None]+np.arange(BLOCK_WEEKS)).reshape(BOOTSTRAP_DRAWS, -1)[:, :len(series)]
        sampled = series[positions]
        supported = np.isfinite(sampled).any(axis=1)
        means = np.nanmean(sampled[supported], axis=1)
        if len(means):
            low, high = np.quantile(means, [.025, .975])
            out.update(block_ci_low=float(low), block_ci_high=float(high))
    return out


def group_outcomes(frame: pd.DataFrame, flagged: np.ndarray, horizon: str, known: np.ndarray):
    result = []
    for name, mask in (("baseline", np.ones(len(frame), bool)), ("flagged", flagged),
                       ("retained", ~flagged), ("unknown_feature", ~known)):
        group = frame.loc[mask]
        values = numeric(group[f"return_{horizon}"])
        row = dict(group=name, n=len(group), unique_tickers=int(group.code.nunique()),
            mean=finite_mean(values), median=float(np.median(values)) if len(values) else None,
            p10=float(np.quantile(values, .1)) if len(values) else None,
            p90=float(np.quantile(values, .9)) if len(values) else None,
            positive_return_rate=float((values > 0).mean()) if len(values) else None)
        for label in LABELS:
            row[label + "_n"] = int(group[label].sum())
        for path_metric in ("mae_3w", "mfe_3w", "mae_4w", "mfe_4w"):
            row[path_metric + "_median"] = (float(group[path_metric].median())
                if path_metric in group and group[path_metric].notna().any() else None)
        result.append(row)
    return result


def ticker_sensitivity(frame: pd.DataFrame, horizon: str) -> list[dict]:
    """Exact leave-one-issuer-out change to the equal-week random-relative mean.

    Thresholds/admission stay fixed. Each ticker occurs at most once per snapshot;
    removing an issuer may remove a singleton week, whose denominator is updated.
    This is a concentration diagnostic, never a ticker exclusion search.
    """
    if frame.empty:
        return []
    changes, weekly_values = [], []
    for _, group in frame.groupby("snapshot_date", sort=True):
        returns = numeric(group[f"return_{horizon}"])
        flags = group._flagged.to_numpy(bool)
        n, m = len(group), int(flags.sum())
        total, retained = returns.sum(), returns[~flags].sum()
        base = retained/n - (n-m)*total/n**2
        weekly_values.append(base)
        for code, ret, flag in zip(group.code, returns, flags):
            after = ((retained-(not flag)*ret)/(n-1) -
                     (n-1-m+int(flag))*(total-ret)/(n-1)**2) if n > 1 else 0.
            changes.append(dict(code=code, difference=after-base, removed_week=int(n == 1)))
    grouped = pd.DataFrame(changes).groupby("code", sort=True).agg(
        difference=("difference", "sum"), removed_weeks=("removed_week", "sum"), affected_weeks=("code", "size"))
    return [dict(code=code, affected_weeks=int(row.affected_weeks),
                 incremental_without_ticker=safe_ratio(sum(weekly_values)+row.difference, len(weekly_values)-row.removed_weeks))
            for code, row in grouped.iterrows()]


def evaluate(frame: pd.DataFrame, calendar: list[str], frozen: list[dict], flags: dict, cost_bps: float):
    """One complete fixed matrix: six policies, two admission panels, four horizons."""
    all_quarters = calendar[MIN_TRAIN_QUARTERS:]
    test = frame.loc[frame.snapshot_date.dt.to_period("Q").astype(str).isin(all_quarters)].copy()
    admission = pd.Series(baseline_admission(test), index=test.index)
    weekly, labels, quarters, groups, events, cohort_parts = [], [], [], [], [], []
    for item in frozen:
        quarter, policy = item["quarter"], item["policy"]
        quarter_frame = test.loc[test.snapshot_date.dt.to_period("Q").astype(str) == quarter]
        flagged, known = flags[(quarter, policy)]
        indexed_flags = pd.Series(flagged, index=quarter_frame.index)
        indexed_known = pd.Series(known, index=quarter_frame.index)
        for panel in PANELS:
            panel_frame = quarter_frame if panel == "all_entries" else quarter_frame.loc[admission.loc[quarter_frame.index]]
            pmask, pknown = indexed_flags.loc[panel_frame.index].to_numpy(), indexed_known.loc[panel_frame.index].to_numpy()
            if item["rule"] is not None:
                cohort = panel_frame.assign(_flagged=pmask, _known=pknown, _policy=policy, _panel=panel)
                observable = cohort.groupby("snapshot_date")._known.transform("any")
                cohort_parts.append(cohort.loc[observable])
            for ix, row in quarter_frame.iterrows():
                if panel == "all_entries":
                    events.append(dict(quarter=quarter, policy=policy, snapshot_date=row.snapshot_date,
                        code=row.code, flagged=bool(indexed_flags[ix]), feature_known=bool(indexed_known[ix]),
                        admitted_nonoverlap_w4=bool(admission[ix])))
            for horizon in HORIZONS:
                context = dict(policy=policy, panel=panel, quarter=quarter, horizon=horizon,
                    role="DIAGNOSTIC_ONLY" if horizon == "w3" else "PRIMARY")
                subset_rows = []
                for snapshot, snapshot_frame in panel_frame.groupby("snapshot_date", sort=True):
                    m, parts = account_week(snapshot_frame, indexed_flags.loc[snapshot_frame.index].to_numpy(),
                        horizon, cost_bps, indexed_known.loc[snapshot_frame.index].to_numpy())
                    row = dict(**context, snapshot_date=snapshot, rule_available=item["rule"] is not None, **m)
                    weekly.append(row)
                    subset_rows.append(row)
                    labels.extend(dict(**context, snapshot_date=snapshot, **part) for part in parts)
                groups.extend(dict(**context, **g) for g in group_outcomes(panel_frame, pmask, horizon, pknown))
                matched = sum(r["matched"] for r in subset_rows)
                flag_n, retain_n = int(pmask.sum()), int((~pmask & pknown).sum())
                status = ("EMPTY_TEST_QUARTER" if not len(panel_frame) else
                          "NO_TRAIN_FEATURE_SUPPORT" if item["rule"] is None else
                          "TEST_FEATURE_UNAVAILABLE" if not pknown.any() else
                          "NO_FLAGS" if flag_n == 0 else
                          "INSUFFICIENT_SUPPORT" if min(flag_n, retain_n) < 10 or matched < 3 else "SUPPORTED")
                qrow = dict(**context, status=status, n=len(panel_frame), flagged_n=flag_n,
                    unknown_n=int((~pknown).sum()), snapshot_weeks=len(subset_rows), matched_weeks=matched)
                evidence_rows = [r for r in subset_rows if r["rule_available"] and r["known_n"] > 0]
                qrow["evidence_weeks"] = len(evidence_rows)
                qrow["operational_cash_delta"] = finite_mean([r["cash_delta"] for r in subset_rows])
                for metric in ("cash_delta", "cash_delta_gross", "incremental_vs_random", "stop_lift",
                               "avoided_gross_loss", "foregone_gross_gain", "saved_cost", "coverage"):
                    qrow[metric] = finite_mean([r[metric] if r[metric] is not None else np.nan for r in evidence_rows])
                quarters.append(qrow)
    weekly_df, quarter_df = pd.DataFrame(weekly), pd.DataFrame(quarters)
    cohorts = pd.concat(cohort_parts, ignore_index=True) if cohort_parts else pd.DataFrame()
    summaries, decisions, concentration = [], [], []
    for (policy, panel, horizon), q in quarter_df.groupby(["policy", "panel", "horizon"], sort=True):
        w = weekly_df.loc[(weekly_df.policy == policy) & (weekly_df.panel == panel) & (weekly_df.horizon == horizon)]
        # No-feature/no-rule quarters remain explicit, never treated as evidence of zero alpha.
        available = w.loc[w.rule_available & (w.known_n > 0)]
        supported_q = q.loc[q.status == "SUPPORTED"]
        row = dict(policy=policy, panel=panel, horizon=horizon,
            role="DIAGNOSTIC_ONLY" if horizon == "w3" else "PRIMARY",
            total_quarters=len(q), supported_quarters=len(supported_q),
            empty_quarters=int((q.status == "EMPTY_TEST_QUARTER").sum()),
            no_flag_quarters=int((q.status == "NO_FLAGS").sum()),
            available_weeks=len(available), observed_weeks=len(w),
            rule_available_weeks=int(w.rule_available.sum()),
            wholly_unknown_weeks=int((w.known_n == 0).sum()),
            positive_supported_quarters=int((supported_q.incremental_vs_random > 0).sum()),
            stop_positive_supported_quarters=int((supported_q.stop_lift > 0).sum()),
            **stability(available))
        for metric in ("baseline_net", "veto_cash_net", "random_cash_net", "cash_delta", "cash_delta_gross",
                       "avoided_gross_loss", "foregone_gross_gain", "saved_cost", "coverage", "known_coverage",
                       "known_matched_return_spread", "stop_lift"):
            row[metric] = finite_mean(available[metric])
        row["break_even_cost_bps"] = (-row["cash_delta_gross"]/row["coverage"]*10000
            if row["coverage"] and row["cash_delta_gross"] is not None else None)
        cohort = (cohorts.loc[(cohorts._policy == policy) & (cohorts._panel == panel)]
                  if not cohorts.empty else cohorts)
        ticker_rows = ticker_sensitivity(cohort, horizon)
        concentration.extend(dict(policy=policy, panel=panel, horizon=horizon, **r) for r in ticker_rows)
        ticker_values = [r["incremental_without_ticker"] for r in ticker_rows if r["incremental_without_ticker"] is not None]
        row["leave_one_ticker_out_min"] = min(ticker_values) if ticker_values else None
        row["leave_one_ticker_out_max"] = max(ticker_values) if ticker_values else None
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    for policy, group in summary.loc[(summary.panel == "nonoverlap_w4") & (summary.role == "PRIMARY")].groupby("policy", sort=True):
        enough = len(group) == 3 and (group.supported_quarters >= 3).all()
        positive = enough and (group.incremental_mean > 0).all() and (group.cash_delta > 0).all()
        stable = (positive and (group.leave_one_quarter_out_min > 0).all()
                  and (group.without_best_week > 0).all() and (group.leave_one_ticker_out_min > 0).all())
        verdict = ("INSUFFICIENT_EVIDENCE" if not enough else
                   "HISTORICAL_ECONOMIC_DIRECTION" if stable else
                   "MIXED_ECONOMIC_DIRECTION" if (group.incremental_mean > 0).any() else
                   "ECONOMIC_VALUE_NOT_DEMONSTRATED")
        risk = summary.loc[(summary.policy == policy) & (summary.panel == "nonoverlap_w4") & (summary.horizon == "w3")].iloc[0]
        risk_direction = ("INSUFFICIENT_EVIDENCE" if risk.supported_quarters < 3 else
            "DIRECTIONALLY_ELEVATED" if risk.stop_lift > 0 and risk.stop_positive_supported_quarters/risk.supported_quarters >= 2/3 else "MIXED_OR_NOT_ELEVATED")
        decisions.append(dict(policy=policy, verdict=verdict, risk_direction=risk_direction,
            risk_only_without_consistent_economic_direction=risk_direction == "DIRECTIONALLY_ELEVATED" and not positive,
            primary_horizons_positive=int((group.incremental_mean > 0).sum()),
            primary_horizons_cash_positive=int((group.cash_delta > 0).sum()),
            production_change=False, prospective_confirmation=False,
            realized_stop_pnl="NOT_IDENTIFIABLE_FROM_TERMINAL_RETURN_CSV"))
    return dict(weekly_economics=weekly_df, label_contributions=pd.DataFrame(labels),
        quarterly_summary=quarter_df, group_outcomes=pd.DataFrame(groups),
        event_flags=pd.DataFrame(events), economic_summary=summary, decision_matrix=pd.DataFrame(decisions),
        ticker_concentration=pd.DataFrame(concentration))

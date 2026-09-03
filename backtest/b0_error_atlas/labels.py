from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import (
    CLEAN_SELECTED_THRESHOLD_PCT,
    CLEAN_WINNER_THRESHOLD_PCT,
    PROFIT_THRESHOLD_PCT,
    STOP_THRESHOLD_PCT,
    TERMINAL_LOSER_THRESHOLD_PCT,
)


def _first_hit_date(
    frame: pd.DataFrame,
    *,
    column: str,
    threshold: float,
    direction: str,
) -> pd.Timestamp | None:
    values = pd.to_numeric(frame[column], errors="coerce")
    if direction == "le":
        mask = values <= threshold
    elif direction == "ge":
        mask = values >= threshold
    else:
        raise RuntimeError(f"Unknown hit direction: {direction}")
    if not bool(mask.any()):
        return None
    return pd.Timestamp(frame.loc[mask, "date"].iloc[0])


def add_path_labels(panel: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    grouped = {
        str(code): g.sort_values("date").copy()
        for code, g in prices.groupby("code", sort=False)
    }

    rows: list[dict[str, Any]] = []
    parity_mismatches: list[tuple[str, str, bool, bool]] = []

    for idx, row in out.iterrows():
        valid = bool(row.get("next_open_price_valid", False))
        record: dict[str, Any] = {
            "_idx": idx,
            "path_valid": False,
            "path_mae_pct": np.nan,
            "path_mfe_pct": np.nan,
            "path_stop8_hit": np.nan,
            "path_profit20_hit": np.nan,
            "path_first_stop_date": "",
            "path_first_profit20_date": "",
            "path_order": "INVALID",
            "stop8_before_profit20_strict": np.nan,
            "clean_big_winner": False,
            "rebound_big_winner": False,
            "terminal_loser": False,
            "strict_path_failure": np.nan,
        }
        if not valid:
            rows.append(record)
            continue

        code = str(row["code"])
        entry_date = pd.Timestamp(str(row["next_open_entry_date"]))
        end_date = pd.Timestamp(str(row["next_open_end_date"]))
        g = grouped.get(code)
        if g is None or g.empty:
            rows.append(record)
            continue

        path = g[(g["date"] >= entry_date) & (g["date"] <= end_date)].copy()
        if path.empty:
            rows.append(record)
            continue

        entry_rows = path[path["date"] == entry_date]
        if entry_rows.empty:
            rows.append(record)
            continue
        entry_open = pd.to_numeric(entry_rows["open"], errors="coerce").iloc[0]
        if pd.isna(entry_open) or float(entry_open) <= 0:
            rows.append(record)
            continue
        entry_open = float(entry_open)

        stop_px = entry_open * (1.0 + STOP_THRESHOLD_PCT / 100.0)
        profit_px = entry_open * (1.0 + PROFIT_THRESHOLD_PCT / 100.0)

        stop_date = _first_hit_date(
            path,
            column="low",
            threshold=stop_px,
            direction="le",
        )
        profit_date = _first_hit_date(
            path,
            column="high",
            threshold=profit_px,
            direction="ge",
        )
        stop_hit = stop_date is not None
        profit_hit = profit_date is not None

        if stop_hit and profit_hit:
            if stop_date < profit_date:
                order = "STOP_FIRST"
                strict_before: bool | float = True
            elif profit_date < stop_date:
                order = "PROFIT_FIRST"
                strict_before = False
            else:
                order = "SAME_DAY_AMBIGUOUS"
                strict_before = np.nan
        elif stop_hit:
            order = "STOP_ONLY"
            strict_before = True
        elif profit_hit:
            order = "PROFIT_ONLY"
            strict_before = False
        else:
            order = "NEITHER"
            strict_before = False

        lows = pd.to_numeric(path["low"], errors="coerce").dropna()
        highs = pd.to_numeric(path["high"], errors="coerce").dropna()
        mae = (
            np.nan
            if lows.empty
            else float((lows.min() / entry_open - 1.0) * 100.0)
        )
        mfe = (
            np.nan
            if highs.empty
            else float((highs.max() / entry_open - 1.0) * 100.0)
        )

        terminal = pd.to_numeric(
            pd.Series([row.get("next_open_w4_return_pct")]),
            errors="coerce",
        ).iloc[0]
        terminal_valid = pd.notna(terminal)
        clean_big = bool(
            terminal_valid
            and float(terminal) >= CLEAN_WINNER_THRESHOLD_PCT
            and not stop_hit
        )
        rebound_big = bool(
            terminal_valid
            and float(terminal) >= CLEAN_WINNER_THRESHOLD_PCT
            and stop_hit
        )
        terminal_loser = bool(
            terminal_valid and float(terminal) <= TERMINAL_LOSER_THRESHOLD_PCT
        )
        strict_failure: bool | float
        if order == "SAME_DAY_AMBIGUOUS":
            strict_failure = np.nan
        else:
            strict_failure = bool(strict_before or terminal_loser)

        audit_stop = bool(row.get("next_open_w4_stop8", False))
        if audit_stop != stop_hit:
            parity_mismatches.append(
                (str(row["snapshot_date"]), code, audit_stop, stop_hit)
            )

        record.update({
            "path_valid": True,
            "path_mae_pct": mae,
            "path_mfe_pct": mfe,
            "path_stop8_hit": stop_hit,
            "path_profit20_hit": profit_hit,
            "path_first_stop_date": "" if stop_date is None else str(stop_date.date()),
            "path_first_profit20_date": (
                "" if profit_date is None else str(profit_date.date())
            ),
            "path_order": order,
            "stop8_before_profit20_strict": strict_before,
            "clean_big_winner": clean_big,
            "rebound_big_winner": rebound_big,
            "terminal_loser": terminal_loser,
            "strict_path_failure": strict_failure,
        })
        rows.append(record)

    if parity_mismatches:
        raise RuntimeError(
            "Path Stop8 parity mismatch against frozen audit state; first="
            f"{parity_mismatches[:10]}"
        )

    labels = pd.DataFrame(rows).set_index("_idx").sort_index()
    for col in labels.columns:
        out[col] = labels[col]
    return out


def task_frames(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build frozen binary tasks without forcing ambiguous middle outcomes."""
    valid = panel[
        (panel["path_valid"] == True)
        & panel["strict_path_failure"].notna()
    ].copy()

    tasks: dict[str, pd.DataFrame] = {}

    # Gate false-negative recovery: among hard-gate rejects, distinguish
    # clean +20% winners from clear path failures.
    gate = valid[valid["current_b0_eligible"] == False].copy()
    gate = gate[
        gate["clean_big_winner"]
        | gate["strict_path_failure"].astype(bool)
    ].copy()
    gate["target"] = gate["clean_big_winner"].astype(int)
    tasks["gate_recovery_clean20_vs_fail"] = gate

    # Eligible-but-unselected misses: isolates selector/capacity rather than gate.
    selector = valid[
        (valid["current_b0_eligible"] == True)
        & (valid["current_b0_selected"] == False)
    ].copy()
    selector = selector[
        selector["clean_big_winner"]
        | selector["strict_path_failure"].astype(bool)
    ].copy()
    selector["target"] = selector["clean_big_winner"].astype(int)
    tasks["selector_recovery_clean20_vs_fail"] = selector

    # All names B0 did not select, regardless of whether gate or Top3 was the cause.
    unselected = valid[valid["current_b0_selected"] == False].copy()
    unselected = unselected[
        unselected["clean_big_winner"]
        | unselected["strict_path_failure"].astype(bool)
    ].copy()
    unselected["target"] = unselected["clean_big_winner"].astype(int)
    tasks["all_unselected_recovery_clean20_vs_fail"] = unselected

    # FP veto: among selected names, distinguish path failures from clean >=8% wins.
    selected = valid[valid["current_b0_selected"] == True].copy()
    clean_selected = (
        pd.to_numeric(selected["next_open_w4_return_pct"], errors="coerce")
        >= CLEAN_SELECTED_THRESHOLD_PCT
    ) & (~selected["path_stop8_hit"].astype(bool))
    fail_selected = selected["strict_path_failure"].astype(bool)
    veto = selected[clean_selected | fail_selected].copy()
    veto["target"] = veto["strict_path_failure"].astype(int)
    tasks["selected_veto_fail_vs_clean8"] = veto

    return tasks


def label_summary(panel: pd.DataFrame, tasks: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    valid = panel[panel["path_valid"] == True].copy()
    for name, frame in tasks.items():
        positives = int(frame["target"].sum()) if not frame.empty else 0
        rows.append({
            "task": name,
            "rows": int(len(frame)),
            "weeks": int(frame["snapshot_date"].nunique()) if not frame.empty else 0,
            "positive_rows": positives,
            "negative_rows": int(len(frame) - positives),
            "positive_rate": None if frame.empty else float(frame["target"].mean()),
        })

    rows.extend([
        {
            "task": "descriptive_gate_clean_big_winners",
            "rows": int(
                (
                    (valid["current_b0_eligible"] == False)
                    & valid["clean_big_winner"].astype(bool)
                ).sum()
            ),
            "weeks": int(
                valid.loc[
                    (valid["current_b0_eligible"] == False)
                    & valid["clean_big_winner"].astype(bool),
                    "snapshot_date",
                ].nunique()
            ),
            "positive_rows": np.nan,
            "negative_rows": np.nan,
            "positive_rate": np.nan,
        },
        {
            "task": "descriptive_gate_rebound_big_winners",
            "rows": int(
                (
                    (valid["current_b0_eligible"] == False)
                    & valid["rebound_big_winner"].astype(bool)
                ).sum()
            ),
            "weeks": int(
                valid.loc[
                    (valid["current_b0_eligible"] == False)
                    & valid["rebound_big_winner"].astype(bool),
                    "snapshot_date",
                ].nunique()
            ),
            "positive_rows": np.nan,
            "negative_rows": np.nan,
            "positive_rate": np.nan,
        },
        {
            "task": "descriptive_selected_path_failures",
            "rows": int(
                (
                    valid["current_b0_selected"].astype(bool)
                    & valid["strict_path_failure"].astype(bool)
                ).sum()
            ),
            "weeks": int(
                valid.loc[
                    valid["current_b0_selected"].astype(bool)
                    & valid["strict_path_failure"].astype(bool),
                    "snapshot_date",
                ].nunique()
            ),
            "positive_rows": np.nan,
            "negative_rows": np.nan,
            "positive_rate": np.nan,
        },
    ])
    return pd.DataFrame(rows)

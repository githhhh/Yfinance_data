"""Causal, strategy-blind discovery dataset and data-profile construction.

The discovery surface intentionally contains only anonymous feature columns plus
causal outcomes. Existing selector/ranking artifacts are rejected before any
agent-facing files are written.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

MAX_RESEARCH_SECONDS = 3600
TRADING_HORIZONS = {"4w": 20, "8w": 40, "12w": 60}

# These tokens identify either prior-decision artifacts or post-entry leakage.
# Feature semantics are also hidden from the agent by X### aliases.
LEAK_TOKENS = (
    "rank",
    "score",
    "top",
    "select",
    "preferred",
    "priority",
    "setup",
    "rule",
    "strict",
    "gate",
    "missed",
    "winner",
    "loser",
    "future",
    "forward",
    "outcome",
    "label",
    "target",
    "stop",
    "mfe",
    "mae",
)
IDENTIFIER_COLUMNS = {
    "code",
    "ticker",
    "symbol",
    "snapshot_date",
    "asof",
    "signal_date",
    "entry_date",
    "entry_price",
}
AGENT_ALLOWED_NON_FEATURE_COLUMNS = {
    "sample_id",
    "period_month",
    "period_quarter",
    "Y_label",
    "Y_stopped_out",
    "Y_4w_return",
    "Y_8w_return",
    "Y_12w_return",
    "Y_4w_excess",
    "Y_8w_excess",
    "Y_12w_excess",
    "Y_mae_12w",
    "Y_mfe_12w",
}


@dataclass(frozen=True)
class OutcomeConfig:
    stop_loss: float = -0.08
    winner_gain: float = 0.20
    minimum_sessions: int = 60


def _normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(df.columns):
        raise ValueError(f"price frame missing columns: {sorted(required - set(df.columns))}")
    out = df.loc[:, ["Open", "High", "Low", "Close"]].copy()
    idx = pd.DatetimeIndex(pd.to_datetime(df.index, errors="coerce"))
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    out["date"] = idx.normalize()
    for col in ["Open", "High", "Low", "Close"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["date", "Open", "High", "Low", "Close"]).sort_values("date").reset_index(drop=True)


def load_price_pickle(path: Path) -> dict[str, pd.DataFrame]:
    """Load the repository's dict[ticker, DataFrame] daily-price pickle shape."""
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    out: dict[str, pd.DataFrame] = {}
    for code, value in raw.items():
        if isinstance(value, dict) and {"index", "columns", "data"}.issubset(value):
            value = pd.DataFrame(index=value["index"], columns=value["columns"], data=value["data"])
        if isinstance(value, pd.DataFrame):
            out[str(code)] = _normalize_price_frame(value)
    return out


def load_replay_candidates(replay_root: Path, *, signal_only: bool = True) -> pd.DataFrame:
    """Load every replay pool row; never apply a downstream rank/selection filter."""
    frames: list[pd.DataFrame] = []
    for path in sorted(replay_root.glob("*/breakout_follow_pool.csv")):
        frame = pd.read_csv(path, dtype={"code": str}, encoding="utf-8-sig")
        if "snapshot_date" not in frame.columns:
            frame["snapshot_date"] = path.parent.name
        frame["_source_pool"] = str(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no replay pools under {replay_root}")
    out = pd.concat(frames, ignore_index=True, sort=False)
    if signal_only and "signal" in out.columns:
        signal = out["signal"].astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes"})
        out = out.loc[signal].copy()
    if "code" not in out.columns or "snapshot_date" not in out.columns:
        raise ValueError("replay candidates require code and snapshot_date")
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["snapshot_date", "code"])
    return out.drop_duplicates(subset=["snapshot_date", "code"], keep="last").reset_index(drop=True)


def _first_event_date(window: pd.DataFrame, *, threshold: float, entry_price: float, side: str) -> pd.Timestamp | None:
    if side == "up":
        hits = window.loc[window["High"] >= entry_price * (1.0 + threshold), "date"]
    elif side == "down":
        hits = window.loc[window["Low"] <= entry_price * (1.0 + threshold), "date"]
    else:
        raise ValueError(side)
    return None if hits.empty else pd.Timestamp(hits.iloc[0])


def _benchmark_return(spy: pd.DataFrame, entry_date: pd.Timestamp, exit_date: pd.Timestamp) -> float | None:
    entry_rows = spy.loc[spy["date"] == entry_date]
    exit_rows = spy.loc[spy["date"] <= exit_date]
    if entry_rows.empty or exit_rows.empty:
        return None
    entry = float(entry_rows.iloc[0]["Open"])
    exit_close = float(exit_rows.iloc[-1]["Close"])
    return exit_close / entry - 1.0 if entry > 0 else None


def evaluate_candidate_path(
    prices: pd.DataFrame,
    signal_date: str | pd.Timestamp,
    *,
    spy_prices: pd.DataFrame | None = None,
    config: OutcomeConfig = OutcomeConfig(),
) -> dict[str, Any]:
    """Label one candidate using only an executable post-signal entry.

    Entry is the next trading session's Open. A later rally after the stop was
    hit is never a clean winner. If stop and target are both touched on the same
    daily bar, daily OHLC cannot identify order, so the sample is ambiguous.
    """
    px = prices if "date" in prices.columns else _normalize_price_frame(prices)
    sig = pd.Timestamp(signal_date).tz_localize(None).normalize()
    post = px.loc[px["date"] > sig].reset_index(drop=True)
    if post.empty:
        return {"label": "censored", "reason": "no_executable_entry"}
    entry_date = pd.Timestamp(post.iloc[0]["date"])
    entry_price = float(post.iloc[0]["Open"])
    window = post.iloc[: config.minimum_sessions].copy()
    if len(window) < config.minimum_sessions or entry_price <= 0:
        return {
            "label": "censored",
            "reason": "insufficient_future_sessions",
            "entry_date": entry_date,
            "entry_price": entry_price,
        }

    target_date = _first_event_date(window, threshold=config.winner_gain, entry_price=entry_price, side="up")
    stop_date = _first_event_date(window, threshold=config.stop_loss, entry_price=entry_price, side="down")
    if target_date is not None and stop_date is not None and target_date == stop_date:
        label = "ambiguous_path"
    elif target_date is not None and (stop_date is None or target_date < stop_date):
        label = "clean_winner"
    elif target_date is not None and stop_date is not None and stop_date < target_date:
        label = "stop_out_then_winner"
    else:
        label = "loser"

    result: dict[str, Any] = {
        "label": label,
        "reason": "",
        "entry_date": entry_date,
        "entry_price": entry_price,
        "target_date": target_date,
        "stop_date": stop_date,
        "stopped_out": stop_date is not None,
        "mae_12w": float(window["Low"].min() / entry_price - 1.0),
        "mfe_12w": float(window["High"].max() / entry_price - 1.0),
    }
    spy = None
    if spy_prices is not None:
        spy = spy_prices if "date" in spy_prices.columns else _normalize_price_frame(spy_prices)
    for name, sessions in TRADING_HORIZONS.items():
        row = window.iloc[sessions - 1]
        exit_date = pd.Timestamp(row["date"])
        stock_return = float(row["Close"] / entry_price - 1.0)
        result[f"return_{name}"] = stock_return
        result[f"exit_date_{name}"] = exit_date
        benchmark = _benchmark_return(spy, entry_date, exit_date) if spy is not None else None
        result[f"benchmark_return_{name}"] = benchmark
        result[f"excess_{name}"] = None if benchmark is None else stock_return - benchmark
    return result


def _bool_like(series: pd.Series) -> bool:
    values = series.dropna().astype(str).str.strip().str.lower()
    return not values.empty and set(values.unique()).issubset({"1", "0", "true", "false", "t", "f", "yes", "no"})


def _feature_is_allowed(name: str, series: pd.Series) -> bool:
    low = name.lower()
    if name in IDENTIFIER_COLUMNS or name == "_source_pool":
        return False
    if any(token in low for token in LEAK_TOKENS):
        return False
    return pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series) or _bool_like(series)


def _blind_feature_value(value: Any, series: pd.Series) -> Any:
    if _bool_like(series):
        if pd.isna(value):
            return pd.NA
        normalized = str(value).strip().lower()
        return 1 if normalized in {"1", "true", "t", "yes"} else 0
    return value


def assert_agent_surface_is_blind(columns: Iterable[str]) -> None:
    for name in columns:
        if name in AGENT_ALLOWED_NON_FEATURE_COLUMNS or name.startswith("X"):
            continue
        raise ValueError(f"unexpected agent-facing column: {name}")


def build_blind_dataset(
    candidates: pd.DataFrame,
    price_map: Mapping[str, pd.DataFrame],
    spy_prices: pd.DataFrame,
    *,
    config: OutcomeConfig = OutcomeConfig(),
) -> tuple[pd.DataFrame, dict[str, str], pd.DataFrame]:
    """Return agent-facing anonymous data, private feature map, and reviewer rows."""
    feature_names = [name for name in candidates.columns if _feature_is_allowed(name, candidates[name])]
    feature_names = sorted(feature_names)
    aliases = {name: f"X{i:03d}" for i, name in enumerate(feature_names, start=1)}
    agent_rows: list[dict[str, Any]] = []
    reviewer_rows: list[dict[str, Any]] = []

    for _, row in candidates.reset_index(drop=True).iterrows():
        code = str(row["code"])
        prices = price_map.get(code)
        if prices is None or prices.empty:
            continue
        outcome = evaluate_candidate_path(prices, row["snapshot_date"], spy_prices=spy_prices, config=config)
        if outcome.get("label") == "censored":
            continue
        entry_date = pd.Timestamp(outcome["entry_date"])
        sample_id = hashlib.sha256(f"{code}|{row['snapshot_date']}".encode()).hexdigest()[:16]
        agent: dict[str, Any] = {
            "sample_id": sample_id,
            "period_month": entry_date.strftime("%Y-%m"),
            "period_quarter": f"{entry_date.year}Q{entry_date.quarter}",
            "Y_label": outcome["label"],
            "Y_stopped_out": int(bool(outcome["stopped_out"])),
            "Y_4w_return": outcome["return_4w"],
            "Y_8w_return": outcome["return_8w"],
            "Y_12w_return": outcome["return_12w"],
            "Y_4w_excess": outcome["excess_4w"],
            "Y_8w_excess": outcome["excess_8w"],
            "Y_12w_excess": outcome["excess_12w"],
            "Y_mae_12w": outcome["mae_12w"],
            "Y_mfe_12w": outcome["mfe_12w"],
        }
        for original, alias in aliases.items():
            agent[alias] = _blind_feature_value(row[original], candidates[original])
        agent_rows.append(agent)
        reviewer = {
            "sample_id": sample_id,
            "code": code,
            "signal_date": pd.Timestamp(row["snapshot_date"]),
            **outcome,
        }
        reviewer_rows.append(reviewer)

    agent_df = pd.DataFrame(agent_rows)
    assert_agent_surface_is_blind(agent_df.columns)
    return agent_df, {alias: original for original, alias in aliases.items()}, pd.DataFrame(reviewer_rows)


def _period_key(ts: pd.Series, granularity: str) -> pd.Series:
    if granularity == "month":
        return ts.dt.strftime("%Y-%m")
    if granularity == "quarter":
        return ts.dt.to_period("Q").astype(str)
    raise ValueError(granularity)


def _max_drawdown(close: pd.Series) -> float | None:
    values = pd.to_numeric(close, errors="coerce").dropna()
    if values.empty:
        return None
    running_peak = values.cummax()
    return float((values / running_peak - 1.0).min())


def build_market_context(reviewer_rows: pd.DataFrame, spy_prices: pd.DataFrame) -> pd.DataFrame:
    """Build contemporaneous month/quarter SPY context, not a global average."""
    if reviewer_rows.empty:
        return pd.DataFrame()
    spy = spy_prices if "date" in spy_prices.columns else _normalize_price_frame(spy_prices)
    entries = pd.to_datetime(reviewer_rows["entry_date"], errors="coerce").dropna()
    rows: list[dict[str, Any]] = []
    for granularity in ("month", "quarter"):
        periods = sorted(_period_key(entries.to_frame("d")["d"], granularity).unique())
        spy_period = _period_key(spy["date"], granularity)
        for period in periods:
            cur = spy.loc[spy_period == period]
            if cur.empty:
                continue
            rows.append(
                {
                    "granularity": granularity,
                    "period": period,
                    "spy_period_return": float(cur.iloc[-1]["Close"] / cur.iloc[0]["Open"] - 1.0),
                    "spy_period_max_drawdown": _max_drawdown(cur["Close"]),
                    "spy_sessions": int(len(cur)),
                }
            )
    return pd.DataFrame(rows)


def build_feature_dossier(agent_df: pd.DataFrame, market_context: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create month/quarter distribution profiles. No global averages are emitted."""
    if agent_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    feature_cols = [c for c in agent_df.columns if c.startswith("X")]
    summary_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    context_lookup = {
        (str(row.granularity), str(row.period)): row
        for row in market_context.itertuples(index=False)
    }
    for granularity, period_col in (("month", "period_month"), ("quarter", "period_quarter")):
        for period, period_df in agent_df.groupby(period_col, dropna=False):
            context = context_lookup.get((granularity, str(period)))
            total = int(len(period_df))
            label_counts = period_df["Y_label"].value_counts(dropna=False).to_dict()
            base = {
                "granularity": granularity,
                "period": str(period),
                "sample_n": total,
                "clean_winner_n": int(label_counts.get("clean_winner", 0)),
                "stop_out_then_winner_n": int(label_counts.get("stop_out_then_winner", 0)),
                "loser_n": int(label_counts.get("loser", 0)),
                "ambiguous_path_n": int(label_counts.get("ambiguous_path", 0)),
                "clean_winner_rate": (int(label_counts.get("clean_winner", 0)) / total) if total else None,
                "spy_period_return": getattr(context, "spy_period_return", None) if context is not None else None,
                "spy_period_max_drawdown": getattr(context, "spy_period_max_drawdown", None) if context is not None else None,
            }
            for target in ("Y_4w_excess", "Y_8w_excess", "Y_12w_excess", "Y_mae_12w", "Y_mfe_12w"):
                numeric = pd.to_numeric(period_df[target], errors="coerce").dropna()
                base[f"{target}_p25"] = numeric.quantile(0.25) if not numeric.empty else None
                base[f"{target}_p50"] = numeric.quantile(0.50) if not numeric.empty else None
                base[f"{target}_p75"] = numeric.quantile(0.75) if not numeric.empty else None
            summary_rows.append(base)

            for label, label_df in period_df.groupby("Y_label", dropna=False):
                for feature in feature_cols:
                    numeric = pd.to_numeric(label_df[feature], errors="coerce").dropna()
                    feature_rows.append(
                        {
                            "granularity": granularity,
                            "period": str(period),
                            "label": str(label),
                            "feature": feature,
                            "sample_n": int(len(label_df)),
                            "observed_n": int(len(numeric)),
                            "missing_n": int(len(label_df) - len(numeric)),
                            "p10": numeric.quantile(0.10) if not numeric.empty else None,
                            "p25": numeric.quantile(0.25) if not numeric.empty else None,
                            "p50": numeric.quantile(0.50) if not numeric.empty else None,
                            "p75": numeric.quantile(0.75) if not numeric.empty else None,
                            "p90": numeric.quantile(0.90) if not numeric.empty else None,
                            "spy_period_return": getattr(context, "spy_period_return", None) if context is not None else None,
                        }
                    )
    return pd.DataFrame(summary_rows), pd.DataFrame(feature_rows)


def chronological_holdout(
    agent_df: pd.DataFrame, *, holdout_quarters: int = 4
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Seal the most recent quarters so discovery cannot tune on them."""
    if holdout_quarters < 1:
        raise ValueError("holdout_quarters must be >= 1")
    quarters = sorted(agent_df["period_quarter"].dropna().astype(str).unique())
    if len(quarters) <= holdout_quarters:
        raise ValueError("not enough quarters to create chronological holdout")
    sealed = quarters[-holdout_quarters:]
    mask = agent_df["period_quarter"].astype(str).isin(sealed)
    discovery = agent_df.loc[~mask].reset_index(drop=True)
    holdout = agent_df.loc[mask].reset_index(drop=True)
    if discovery.empty or holdout.empty:
        raise ValueError("chronological split produced an empty partition")
    return discovery, holdout, sealed


def write_agent_workspace(
    agent_df: pd.DataFrame, output_root: Path, market_context: pd.DataFrame | None = None
) -> Path:
    """Write only blind artifacts to the directory passed to the research agent."""
    workspace = output_root / "agent_workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    assert_agent_surface_is_blind(agent_df.columns)
    agent_df.to_csv(workspace / "samples.csv", index=False)
    if market_context is not None and not market_context.empty:
        public_market = market_context.rename(
            columns={
                "granularity": "period_type",
                "spy_period_return": "M_market_return",
                "spy_period_max_drawdown": "M_market_drawdown",
                "spy_sessions": "M_sessions",
            }
        )
        public_market.to_csv(workspace / "market_context.csv", index=False)
    prompt = """You are given time-indexed samples with anonymous feature columns X### and causal outcome columns Y_*.
Discover compact, stable rules that distinguish clean_winner from loser across months and quarters.
Treat stop_out_then_winner and ambiguous_path as non-winning outcomes for rule discovery.
Use distributional evidence and temporal stability; do not optimize a global average.
Use market_context.csv to check whether findings persist across different broad-market conditions.
Prefer rules whose direction and usefulness persist across multiple periods and market regimes.
Do not infer feature semantics. Limit interactions and thresholds to those supported by repeated period evidence.
Write rule.json with the proposed frozen rule, feature aliases, thresholds, period evidence, failure cases, and holdout plan.
"""
    (workspace / "prompt.md").write_text(prompt, encoding="utf-8")
    return workspace


def run_research_command(
    command: list[str], workspace: Path, *, timeout_seconds: int = MAX_RESEARCH_SECONDS
) -> subprocess.CompletedProcess[str]:
    """Run research from a temporary copy containing only agent-facing files."""
    effective_timeout = min(max(1, int(timeout_seconds)), MAX_RESEARCH_SECONDS)
    with tempfile.TemporaryDirectory(prefix="blind_rule_agent_") as tmp:
        isolated = Path(tmp) / "workspace"
        shutil.copytree(workspace, isolated)
        env = os.environ.copy()
        env.update({"HOME": str(Path(tmp) / "home"), "PYTHONNOUSERSITE": "1"})
        for name in ("PWD", "OLDPWD", "PYTHONPATH"):
            env.pop(name, None)
        completed = subprocess.run(
            command,
            cwd=isolated,
            text=True,
            capture_output=True,
            timeout=effective_timeout,
            check=False,
            env=env,
        )
        for name in ("rule.json", "research_notes.md"):
            produced = isolated / name
            if produced.exists():
                shutil.copy2(produced, workspace / name)
        return completed


def freeze_rule_artifact(rule_path: Path, output_root: Path) -> dict[str, Any]:
    """Freeze the blind rule before any comparator-specific analysis is allowed."""
    raw = rule_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    frozen_dir = output_root / "frozen"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    frozen_rule = frozen_dir / "rule.json"
    frozen_rule.write_bytes(raw)
    manifest = {
        "sha256": digest,
        "source": str(rule_path),
        "frozen_rule": str(frozen_rule),
        "comparison_allowed_only_after_freeze": True,
    }
    (frozen_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest

"""Run-scoped BreakoutFollow Pool contract consumed by ``quant_trade``.

The review projection itself lives in ``dashboard.services`` so Dashboard and
the compatibility API use the same current-left business rules.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from hashlib import sha256
from pathlib import Path

import pandas as pd

from dashboard.services.bf_midweek_review import build_midweek_review_for_snapshots
from eps_pit import EPSResolveMode, EPSStatus, SignalEPSLookup, enrich_pool_with_signal_eps


DATA_ROOT = str(Path(__file__).resolve().parents[1])
BREAKOUT_FOLLOW_POOL_PATH = os.path.join(DATA_ROOT, "us", "breakout_follow_pool.csv")
BREAKOUT_FOLLOW_POOL_MIDWEEK_PATH = os.path.join(
    DATA_ROOT, "us", "breakout_follow_pool_midweek.csv"
)

EPS_PUBLICATION_COLUMNS = (
    "eps_yoy_growth",
    "eps_yoy_growth_source",
    "eps_yoy_growth_status",
    "eps_yoy_growth_missing_reason",
    "eps_growth_type",
)


def _is_truthy(value) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, bool):
        return value
    try:
        if int(value) == 1:
            return True
    except Exception:
        pass
    return str(value).strip().lower() in {"true", "1", "1.0"}


def _pool_codes(pool: pd.DataFrame) -> frozenset[str]:
    if "code" not in pool.columns:
        raise ValueError("BF Pool 缺少字段: ['code']")
    normalized = [
        str(value).strip()
        for value in pool["code"].dropna()
        if str(value).strip() and str(value).strip().lower() != "nan"
    ]
    if len(normalized) != len(set(normalized)):
        raise ValueError("BF Pool code 重复")
    return frozenset(normalized)


def _snapshot_digest(path: str) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _pool_snapshot_date(path: str) -> str:
    if not os.path.exists(path):
        return ""
    try:
        pool = pd.read_csv(path, usecols=["snapshot_date"], encoding="utf-8-sig")
    except Exception:
        return ""
    if "snapshot_date" not in pool.columns:
        return ""
    dates = pool["snapshot_date"].dropna().astype(str).str.strip().str[:10]
    dates = dates[dates.ne("")]
    return str(dates.max()) if not dates.empty else ""


def _latest_pool_path_by_snapshot_date() -> tuple[str, str]:
    candidates = [
        (_pool_snapshot_date(BREAKOUT_FOLLOW_POOL_PATH), BREAKOUT_FOLLOW_POOL_PATH),
        (_pool_snapshot_date(BREAKOUT_FOLLOW_POOL_MIDWEEK_PATH), BREAKOUT_FOLLOW_POOL_MIDWEEK_PATH),
    ]
    candidates = [(snapshot_date, path) for snapshot_date, path in candidates if snapshot_date]
    if not candidates:
        raise FileNotFoundError("没有可用的 BF Pool snapshot_date")
    snapshot_date, path = max(candidates, key=lambda item: item[0])
    return path, snapshot_date


def _load_complete_actionable_codes(pool: pd.DataFrame) -> list[str]:
    required = {"code", "signal", "ibd_entry_valid", "ibd_entry_status"}
    missing = required.difference(pool.columns)
    if missing:
        raise ValueError(f"BF Pool 缺少字段: {sorted(missing)}")
    signal_mask = pool["signal"].map(_is_truthy)
    if pool.loc[signal_mask, "ibd_entry_valid"].isna().any():
        raise ValueError("BF Pool IBD enrichment 未完成")
    actionable = pool[signal_mask & pool["ibd_entry_status"].eq("ACTIONABLE")]
    codes = actionable["code"].dropna().astype(str).str.strip().tolist()
    codes.reverse()
    return codes


def _load_midweek_actionable_codes(midweek: pd.DataFrame) -> list[str]:
    complete = pd.DataFrame()
    if os.path.exists(BREAKOUT_FOLLOW_POOL_PATH):
        try:
            complete = pd.read_csv(
                BREAKOUT_FOLLOW_POOL_PATH,
                dtype={"code": str},
                encoding="utf-8-sig",
            )
        except Exception as exc:
            logging.warning("BF complete Pool 不可用，本轮周中结果禁用 Carry: %s", exc)
    review = build_midweek_review_for_snapshots(midweek, complete)
    codes = list(review.actionable_codes)
    codes.reverse()
    return codes


def _signal_eps_missing_count(pool: pd.DataFrame) -> int:
    if "signal" not in pool.columns:
        return 0
    signal_mask = pool["signal"].map(_is_truthy)
    if "eps_yoy_growth" not in pool.columns:
        return int(signal_mask.sum())
    return int(pool.loc[signal_mask, "eps_yoy_growth"].isna().sum())


def _signal_eps_missing_codes(pool: pd.DataFrame) -> list[str]:
    if "signal" not in pool.columns or "code" not in pool.columns:
        return []
    signal_mask = pool["signal"].map(_is_truthy)
    if "eps_yoy_growth" in pool.columns:
        signal_mask = signal_mask & pool["eps_yoy_growth"].isna()
    missing = pool.loc[signal_mask, "code"]
    codes = {
        str(value).strip()
        for value in missing.dropna()
        if str(value).strip() and str(value).strip().lower() != "nan"
    }
    return sorted(codes)


def _signal_eps_provider_error_codes(pool: pd.DataFrame) -> list[str]:
    required = {"signal", "code", "eps_yoy_growth", "eps_yoy_growth_status"}
    if not required.issubset(pool.columns):
        return []
    signal_mask = pool["signal"].map(_is_truthy)
    error_mask = pool["eps_yoy_growth_status"].astype(str).eq(EPSStatus.PROVIDER_ERROR.value)
    missing_mask = pool["eps_yoy_growth"].isna()
    values = pool.loc[signal_mask & error_mask & missing_mask, "code"]
    return sorted({str(value).strip() for value in values.dropna() if str(value).strip()})


def _validate_eps_publication_contract(pool: pd.DataFrame) -> None:
    """Require complete EPS PIT metadata on every published signal row."""
    if "signal" not in pool.columns:
        return

    missing_columns = [
        column for column in EPS_PUBLICATION_COLUMNS if column not in pool.columns
    ]
    if missing_columns:
        raise ValueError(f"BF Pool EPS PIT publication 字段不完整: {missing_columns}")

    signal_mask = pool["signal"].map(_is_truthy)
    if not signal_mask.any():
        return

    signal_rows = pool.loc[signal_mask]
    raw_eps = signal_rows["eps_yoy_growth"]
    numeric_eps = pd.to_numeric(raw_eps, errors="coerce").replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )
    invalid_eps = (
        raw_eps.notna()
        & raw_eps.astype(str).str.strip().ne("")
        & numeric_eps.isna()
    )
    if invalid_eps.any():
        codes = ", ".join(signal_rows.loc[invalid_eps, "code"].astype(str).tolist())
        raise ValueError(f"BF Pool EPS PIT publication EPS 非有限数值: {codes}")

    status = signal_rows["eps_yoy_growth_status"].astype("string").str.strip().str.lower()
    source = signal_rows["eps_yoy_growth_source"].astype("string").str.strip()
    reason = signal_rows["eps_yoy_growth_missing_reason"].astype("string").str.strip()

    resolved_mask = numeric_eps.notna()
    unresolved_mask = ~resolved_mask

    bad_resolved = resolved_mask & (
        status.ne(EPSStatus.RESOLVED.value)
        | status.isna()
        | source.isna()
        | source.eq("")
    )
    if bad_resolved.any():
        codes = ", ".join(signal_rows.loc[bad_resolved, "code"].astype(str).tolist())
        raise ValueError(f"BF Pool EPS PIT publication resolved 元数据不完整: {codes}")

    bad_resolved_reason = resolved_mask & reason.notna() & reason.ne("")
    if bad_resolved_reason.any():
        codes = ", ".join(
            signal_rows.loc[bad_resolved_reason, "code"].astype(str).tolist()
        )
        raise ValueError(
            f"BF Pool EPS PIT publication resolved 行不应有 missing_reason: {codes}"
        )

    bad_unresolved = unresolved_mask & (
        status.ne(EPSStatus.EXPECTED_UNAVAILABLE.value)
        | status.isna()
        | reason.isna()
        | reason.eq("")
    )
    if bad_unresolved.any():
        codes = ", ".join(signal_rows.loc[bad_unresolved, "code"].astype(str).tolist())
        raise ValueError(f"BF Pool EPS PIT publication unresolved 元数据不完整: {codes}")


def _clean_log_value(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _signal_eps_status_items(pool: pd.DataFrame, status: EPSStatus) -> list[str]:
    required = {"signal", "code", "eps_yoy_growth_status"}
    if not required.issubset(pool.columns):
        return []
    signal_mask = pool["signal"].map(_is_truthy)
    status_mask = pool["eps_yoy_growth_status"].astype(str).eq(status.value)
    rows = pool.loc[signal_mask & status_mask].copy()
    items: list[str] = []
    for _, row in rows.sort_values("code").iterrows():
        code = _clean_log_value(row.get("code"))
        if not code:
            continue
        reason = _clean_log_value(row.get("eps_yoy_growth_missing_reason"))
        if reason:
            items.append(f"{code}({reason})")
        else:
            items.append(code)
    return items


def _format_signal_eps_status_items(items: list[str]) -> str:
    return ", ".join(items) if items else "none"


def _log_signal_eps_pit_summary(pool: pd.DataFrame) -> None:
    resolved = _signal_eps_status_items(pool, EPSStatus.RESOLVED)
    expected_unavailable = _signal_eps_status_items(pool, EPSStatus.EXPECTED_UNAVAILABLE)
    provider_error = _signal_eps_status_items(pool, EPSStatus.PROVIDER_ERROR)
    logging.info(
        "BF Pool signal EPS PIT summary: resolved=%s [%s]; "
        "expected_unavailable=%s [%s]; provider_error=%s [%s]",
        len(resolved),
        _format_signal_eps_status_items(resolved),
        len(expected_unavailable),
        _format_signal_eps_status_items(expected_unavailable),
        len(provider_error),
        _format_signal_eps_status_items(provider_error),
    )


def _enrich_signal_eps(pool: pd.DataFrame) -> pd.DataFrame:
    before = _signal_eps_missing_count(pool)
    enriched = enrich_pool_with_signal_eps(
        pool,
        refresh_missing=before > 0,
        mode=EPSResolveMode.LIVE,
        csv_path=_pit_store_path(),
    )
    after = _signal_eps_missing_count(enriched)
    repaired = before - after
    if repaired:
        logging.info("BF Pool signal EPS supplemented: %s repaired, %s unresolved", repaired, after)
    elif before:
        logging.warning("BF Pool signal EPS still missing after supplement: %s unresolved", after)

    unresolved_codes = _signal_eps_missing_codes(enriched)
    if unresolved_codes:
        logging.warning("BF Pool signal EPS unresolved codes: %s", ", ".join(unresolved_codes))

    _log_signal_eps_pit_summary(enriched)

    provider_error_codes = _signal_eps_provider_error_codes(enriched)
    if provider_error_codes:
        # Technical provider failures are different from a legitimate lack of
        # comparable EPS. Publishing here would silently make downstream Top3
        # selection exclude otherwise-valid candidates, so fail this run.
        codes = ", ".join(provider_error_codes)
        logging.error("BF Pool signal EPS provider failure: %s", codes)
        raise RuntimeError(f"BF Pool signal EPS provider failure: {codes}")
    return enriched


def supplement_latest_pool_signal_eps() -> dict[str, object]:
    """Refresh missing signal EPS in the latest pool selected by snapshot_date."""
    path, snapshot_date = _latest_pool_path_by_snapshot_date()
    pool = pd.read_csv(path, dtype={"code": str}, encoding="utf-8-sig")
    before = _signal_eps_missing_count(pool)
    enriched = _enrich_signal_eps(pool)
    _validate_eps_publication_contract(enriched)
    after = _signal_eps_missing_count(enriched)
    repaired = before - after
    if not enriched.equals(pool):
        enriched.to_csv(path, index=False, encoding="utf-8-sig")
    return {
        "path": path,
        "snapshot_date": snapshot_date,
        "before_missing": before,
        "after_missing": after,
        "repaired": repaired,
        "unresolved_codes": _signal_eps_missing_codes(enriched),
    }


class BreakoutFollowPoolRun:
    """Run-scoped access to the weekend or midweek BreakoutFollow Pool."""

    def __init__(self, *, _midweek: bool):
        self._midweek = _midweek
        self._published_digest: str | None = None

    @classmethod
    def weekend(cls) -> "BreakoutFollowPoolRun":
        return cls(_midweek=False)

    @classmethod
    def midweek(cls) -> "BreakoutFollowPoolRun":
        return cls(_midweek=True)

    @property
    def name(self) -> str:
        return "midweek" if self._midweek else "weekend"

    @property
    def path(self) -> str:
        return BREAKOUT_FOLLOW_POOL_MIDWEEK_PATH if self._midweek else BREAKOUT_FOLLOW_POOL_PATH

    def save_snapshot(self, pool: pd.DataFrame) -> None:
        # Weekend and midweek intentionally share this exact publication path:
        # structure -> LIVE EPS PIT enrichment -> publication invariant -> CSV.
        _pool_codes(pool)
        pool = _enrich_signal_eps(pool)
        _validate_eps_publication_contract(pool)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        pool.to_csv(self.path, index=False, encoding="utf-8-sig")
        self._published_digest = _snapshot_digest(self.path)

    def ensure_current_snapshot(self) -> pd.DataFrame:
        if self._published_digest is None:
            raise RuntimeError(f"BF {self.name} Pool 本轮快照尚未成功写入")
        try:
            current_digest = _snapshot_digest(self.path)
        except OSError as exc:
            raise ValueError(f"BF {self.name} Pool 与本轮快照不一致") from exc
        if current_digest != self._published_digest:
            raise ValueError(f"BF {self.name} Pool 与本轮快照不一致")
        pool = pd.read_csv(self.path, dtype={"code": str}, encoding="utf-8-sig")
        _validate_eps_publication_contract(pool)
        return pool

    def load_actionable_codes(self) -> list[str]:
        pool = self.ensure_current_snapshot()
        if self._midweek:
            return _load_midweek_actionable_codes(pool)
        return _load_complete_actionable_codes(pool)

    def commit(self) -> None:
        self.ensure_current_snapshot()
        _commit_pool(self.path)


def _pit_store_path() -> str:
    path = SignalEPSLookup.DEFAULT_CSV_PATH
    return path if os.path.isabs(path) else os.path.join(DATA_ROOT, path)


def _commit_pool(pool_path: str) -> None:
    """Commit only a pool that satisfies the current EPS publication contract."""
    if os.path.exists(pool_path):
        pool = pd.read_csv(pool_path, dtype={"code": str}, encoding="utf-8-sig")
        _validate_eps_publication_contract(pool)

    try:
        managed_paths = [pool_path]
        pit_path = _pit_store_path()
        if os.path.exists(pit_path):
            managed_paths.append(pit_path)

        subprocess.run(["git", "add", *managed_paths], cwd=DATA_ROOT, check=True)
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--", *managed_paths],
            cwd=DATA_ROOT,
        )
        if staged.returncode == 0:
            return
        if staged.returncode != 1:
            raise subprocess.CalledProcessError(staged.returncode, staged.args)

        subprocess.run(
            ["git", "commit", "-m", "Update breakout follow pool"],
            cwd=DATA_ROOT,
            check=True,
        )
        for attempt in range(1, 4):
            try:
                subprocess.run(["git", "push"], cwd=DATA_ROOT, check=True)
                logging.info("Yfinance_data仓库已更新: %s", os.path.basename(pool_path))
                break
            except subprocess.CalledProcessError:
                if attempt == 3:
                    raise
                time.sleep(5)
    except subprocess.CalledProcessError as exc:
        logging.error("Git操作失败: %s", exc)
    except Exception as exc:
        logging.error("检查并提交文件时出错: %s", exc)


__all__ = [
    "BREAKOUT_FOLLOW_POOL_MIDWEEK_PATH",
    "BREAKOUT_FOLLOW_POOL_PATH",
    "BreakoutFollowPoolRun",
    "supplement_latest_pool_signal_eps",
]

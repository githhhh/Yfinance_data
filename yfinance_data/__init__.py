"""Run-scoped BreakoutFollow Pool contract consumed by ``quant_trade``.

The review projection itself lives in ``dashboard.services`` so Dashboard and
the compatibility API use the same current-left business rules.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from hashlib import sha256
from pathlib import Path

import pandas as pd

from pool_exit_watch_store import (COLUMNS, WATCH_COMPLETE, WATCH_MIDWEEK, pool_identity,
    published_complete_pools, official_watch_for_week, watch_was_ever_published,
    watch_frame, parse_watch_csv, published_watch_state, inherit_download_holds,
    active_watch_codes)
from bf_snapshot import (
    BreakoutFollowPoolKind,
    classify_breakout_follow_pool,
    complete_snapshot_week,
    complete_target_week,
    is_valid_complete_baseline,
    monday_of_week,
)
from dashboard.services.bf_midweek_review import build_midweek_review_for_snapshots
from eps_pit import EPSResolveMode, EPSStatus, SignalEPSLookup, enrich_pool_with_signal_eps, get_signal_eps
from .pool_industry import enrich_pool_with_industry, load_industry_lookup


DATA_ROOT = str(Path(__file__).resolve().parents[1])
BREAKOUT_FOLLOW_POOL_PATH = os.path.join(DATA_ROOT, "us", "breakout_follow_pool.csv")
BREAKOUT_FOLLOW_POOL_MIDWEEK_PATH = os.path.join(
    DATA_ROOT, "us", "breakout_follow_pool_midweek.csv"
)
IBD_DOUBLE_BOTTOM_SNAPSHOT_PATH = os.path.join(
    DATA_ROOT, "us", "ibd_double_bottom_snapshot.csv"
)
IBD_DOUBLE_BOTTOM_SNAPSHOT_MIDWEEK_PATH = os.path.join(
    DATA_ROOT, "us", "ibd_double_bottom_snapshot_midweek.csv"
)

EPS_PUBLICATION_COLUMNS = (
    "eps_yoy_growth",
    "eps_yoy_growth_source",
    "eps_yoy_growth_status",
    "eps_yoy_growth_missing_reason",
    "eps_growth_type",
)
POOL_PRESENTATION_COLUMNS = (*EPS_PUBLICATION_COLUMNS, "sector", "industry")


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


def _ensure_pool_industry_columns(pool: pd.DataFrame) -> pd.DataFrame:
    result = pool.copy()
    for column in ("sector", "industry"):
        if column not in result.columns:
            result[column] = pd.NA
    return result


def _reject_empty_pool_overwrite(path: str, candidate_codes: frozenset[str], *, name: str) -> None:
    if candidate_codes:
        return

    # The previous strategy-side guard always inspected the complete Pool,
    # including during a midweek run. Keep that protection in the authority:
    # an empty current-left snapshot must not clear a non-empty complete Pool
    # merely because this is the first midweek publication for the week.
    protected_paths = [path]
    if path != BREAKOUT_FOLLOW_POOL_PATH:
        protected_paths.append(BREAKOUT_FOLLOW_POOL_PATH)

    for protected_path in dict.fromkeys(protected_paths):
        if not os.path.exists(protected_path):
            continue
        try:
            existing = pd.read_csv(protected_path, dtype={"code": str}, keep_default_na=False, na_values=[""], encoding="utf-8-sig")
            existing_codes = _pool_codes(existing)
        except Exception as exc:
            raise RuntimeError(f"BF {name} Pool 候选为空且无法验证现有快照") from exc
        if existing_codes:
            raise RuntimeError(
                f"BF {name} Pool 保存拦截: 已发布 Pool 包含 {len(existing_codes)} 条记录，"
                "本轮候选为空"
            )


def _snapshot_digest(path: str) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _write_pool_snapshot_atomically(pool: pd.DataFrame, path: str) -> None:
    """Replace a published Pool only after its complete CSV is written nearby."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, pending_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".pending",
        dir=directory,
    )
    os.close(fd)
    try:
        pool.to_csv(pending_path, index=False, encoding="utf-8-sig")
        os.replace(pending_path, path)
    except Exception:
        if os.path.exists(pending_path):
            os.unlink(pending_path)
        raise


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
                dtype={"code": str}, keep_default_na=False, na_values=[""],
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
    pool = pd.read_csv(path, dtype={"code": str}, keep_default_na=False, na_values=[""], encoding="utf-8-sig")
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


class IbdDoubleBottomSnapshotRun:
    """Run-scoped publication contract for Double Bottom review snapshots."""

    REQUIRED_COLUMNS = {
        "code",
        "snapshot_date",
        "detection_path",
        "signal_type",
        "selection_eligible",
    }

    def __init__(self, *, _midweek: bool):
        self._midweek = _midweek
        self._published_digest: str | None = None

    @classmethod
    def weekend(cls) -> "IbdDoubleBottomSnapshotRun":
        return cls(_midweek=False)

    @classmethod
    def complete(cls) -> "IbdDoubleBottomSnapshotRun":
        """Backward-compatible alias for the weekend snapshot."""
        return cls.weekend()

    @classmethod
    def midweek(cls) -> "IbdDoubleBottomSnapshotRun":
        return cls(_midweek=True)

    @property
    def name(self) -> str:
        # Preserve the pre-existing public name while exposing weekend() as
        # the canonical constructor for the complete snapshot.
        return "midweek" if self._midweek else "complete"

    @property
    def path(self) -> str:
        if self._midweek:
            return IBD_DOUBLE_BOTTOM_SNAPSHOT_MIDWEEK_PATH
        return IBD_DOUBLE_BOTTOM_SNAPSHOT_PATH

    def _validate(self, snapshot: pd.DataFrame) -> None:
        missing = self.REQUIRED_COLUMNS.difference(snapshot.columns)
        if missing:
            raise ValueError(
                f"IBD Double Bottom {self.name} snapshot 缺少字段: {sorted(missing)}"
            )

        codes = [
            str(value).strip()
            for value in snapshot["code"].dropna()
            if str(value).strip() and str(value).strip().lower() != "nan"
        ]
        if len(codes) != len(set(codes)):
            raise ValueError(f"IBD Double Bottom {self.name} snapshot code 重复")

        if snapshot.empty:
            return

        dates = (
            snapshot["snapshot_date"]
            .dropna()
            .astype(str)
            .str.strip()
            .str[:10]
        )
        dates = dates[dates.ne("")]
        if len(dates) != len(snapshot) or dates.nunique() != 1:
            raise ValueError(
                f"IBD Double Bottom {self.name} snapshot snapshot_date 不一致"
            )

    def save_snapshot(
        self,
        snapshot: pd.DataFrame,
        *,
        snapshot_date: str,
    ) -> None:
        snapshot_date = str(snapshot_date).strip()[:10]
        try:
            normalized_date = pd.Timestamp(snapshot_date).strftime("%Y-%m-%d")
        except Exception as exc:
            raise ValueError(
                f"IBD Double Bottom {self.name} snapshot_date 非法: {snapshot_date}"
            ) from exc
        if normalized_date != snapshot_date:
            raise ValueError(
                f"IBD Double Bottom {self.name} snapshot_date 非法: {snapshot_date}"
            )

        published = snapshot.copy(deep=True)
        published["snapshot_date"] = snapshot_date
        self._validate(published)
        _write_pool_snapshot_atomically(published, self.path)
        self._published_digest = _snapshot_digest(self.path)

    def ensure_current_snapshot(self) -> pd.DataFrame:
        if self._published_digest is None:
            raise RuntimeError(
                f"IBD Double Bottom {self.name} snapshot 本轮尚未成功写入"
            )
        try:
            current_digest = _snapshot_digest(self.path)
        except OSError as exc:
            raise ValueError(
                f"IBD Double Bottom {self.name} snapshot 与本轮结果不一致"
            ) from exc
        if current_digest != self._published_digest:
            raise ValueError(
                f"IBD Double Bottom {self.name} snapshot 与本轮结果不一致"
            )
        snapshot = pd.read_csv(
            self.path,
            dtype={"code": str}, keep_default_na=False, na_values=[""],
            encoding="utf-8-sig",
        )
        self._validate(snapshot)
        return snapshot

    def commit(self) -> None:
        self.ensure_current_snapshot()
        _commit_managed_csv(
            self.path,
            message="Update IBD double bottom snapshot",
        )


class BreakoutFollowPoolRun:
    """Run-scoped access to the weekend or midweek BreakoutFollow Pool."""

    def __init__(self, *, _midweek: bool):
        self._midweek = _midweek
        self._published_digest: str | None = None
        self._watch_companion: dict | None = None

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
        # structure -> LIVE EPS PIT + industry -> publication invariant -> CSV.
        candidate_codes = _pool_codes(pool)
        _reject_empty_pool_overwrite(self.path, candidate_codes, name=self.name)
        pool = _enrich_signal_eps(pool)
        pool = enrich_pool_with_industry(pool)
        pool = _ensure_pool_industry_columns(pool)
        _validate_eps_publication_contract(pool)
        _write_pool_snapshot_atomically(pool, self.path)
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
        pool = pd.read_csv(self.path, dtype={"code": str}, keep_default_na=False, na_values=[""], encoding="utf-8-sig")
        _validate_eps_publication_contract(pool)
        return pool

    def load_actionable_codes(self) -> list[str]:
        pool = self.ensure_current_snapshot()
        if self._midweek:
            return _load_midweek_actionable_codes(pool)
        return _load_complete_actionable_codes(pool)

    def project_published_display_fields(self, preview: pd.DataFrame) -> pd.DataFrame:
        """Return a preview with display fields copied from this published snapshot."""
        preview_codes = _pool_codes(preview)
        published = self.ensure_current_snapshot()
        published_codes = _pool_codes(published)
        if preview_codes != published_codes:
            raise ValueError(f"BF {self.name} Preview code 集合与本轮权威 Pool 不一致")

        result = preview.copy()
        published_by_code = published.set_index("code")
        for column in POOL_PRESENTATION_COLUMNS:
            result[column] = (
                result["code"].map(published_by_code[column])
                if column in published_by_code.columns
                else pd.NA
            )
        return result

    def register_companion(self, *, target_path: str, staged_path: str,
                           source_pool_date: str, source_pool_codes_hash: str) -> None:
        """Bind one prepared CSV to this run's published Pool."""
        date, digest = pool_identity(self.ensure_current_snapshot())
        if (date, digest) != (source_pool_date, source_pool_codes_hash):
            raise ValueError("Watch/Pool source identity mismatch")
        target = WATCH_MIDWEEK if self._midweek else WATCH_COMPLETE
        if Path(target_path).resolve() != (Path(DATA_ROOT) / target).resolve():
            raise ValueError("invalid Watch companion target")
        if self._watch_companion is not None:
            raise ValueError("duplicate companion")
        source = Path(staged_path)
        if not source.is_file() or source.suffix != ".pending":
            raise ValueError("missing staged companion")
        # Validate the *same* schema, phases and event JSON that readers use.
        # Column-only validation could publish a pair that no downloader or
        # dashboard could subsequently parse, despite matching Pool identity.
        # The authority owns manual fields. Merge them once, BEFORE binding
        # the existing staged-file digest; the Watch engine remains unaware.
        version = None
        draft = pd.read_csv(source, dtype={"code": str},
                            keep_default_na=False, encoding="utf-8-sig")
        if self._midweek:
            # Provisional projection never publishes a manual control value.
            draft["download_hold_reason"] = ""
        else:
            published, version, _, _ = published_watch_state(DATA_ROOT)
            draft = inherit_download_holds(draft, published)
        draft.to_csv(source, index=False, encoding="utf-8-sig")
        rows = parse_watch_csv(source)
        if not self._midweek and any(r["phase"] == "PROVISIONAL_OUT" for r in rows):
            raise ValueError("complete Watch cannot contain provisional rows")
        if any(r["snapshot_date"] != date for r in rows):
            raise ValueError("Watch rows reference another Pool")
        self._watch_companion = dict(target=str(target_path), staged=str(staged_path),
                                      date=date, digest=digest, watch_version=version,
                                      sha=sha256(source.read_bytes()).hexdigest())

    def commit(self) -> None:
        pool = self.ensure_current_snapshot()
        target = WATCH_MIDWEEK if self._midweek else WATCH_COMPLETE
        art = self._watch_companion
        if art is None:
            # Deleting the local Watch file does not disable a feature that was
            # already published. Never silently revert to Pool-only commits.
            if ((Path(DATA_ROOT) / target).exists() or
                ((Path(DATA_ROOT) / ".git").exists() and
                 watch_was_ever_published(DATA_ROOT))):
                raise RuntimeError("published Watch exists; BF run requires companion")
            _commit_pool(self.path)
            return
        if pool_identity(pool) != (art["date"], art["digest"]):
            raise RuntimeError("BF Pool changed after companion was attached")
        staged, published = Path(art["staged"]), Path(art["target"])

        def prepare_locked_watch() -> None:
            # Same managed publisher lock as manual Watch-only edits. Never
            # overwrite the official file before validating its Git vintage.
            if art["watch_version"] is not None:
                _, current_version, _, _ = published_watch_state(DATA_ROOT)
                if current_version != art["watch_version"]:
                    raise RuntimeError(
                        "published complete Watch changed since preparation; "
                        "discard this run and rerun after checking external side effects"
                    )
            if not staged.is_file():
                raise RuntimeError("staged Watch missing")
            if sha256(staged.read_bytes()).hexdigest() != art["sha"]:
                raise RuntimeError("staged Watch mutated")
            os.replace(staged, published)
            if not published.is_file() or sha256(published.read_bytes()).hexdigest() != art["sha"]:
                raise RuntimeError("Watch companion missing or corrupt")

        _commit_pool(self.path, extra_paths=[str(published)],
                     before_stage=prepare_locked_watch)
        self._watch_companion = None


def _pit_store_path() -> str:
    path = SignalEPSLookup.DEFAULT_CSV_PATH
    return path if os.path.isabs(path) else os.path.join(DATA_ROOT, path)


def _commit_managed_csv(paths: str | list[str], *, message: str,
                        before_stage=None) -> None:
    """Publish only owned paths under a per-checkout lock, fail closed on drift.

    A Git commit cannot be pushed in isolation: push includes all its ancestors.
    Check every unpushed commit, not just the final diff, to prevent a foreign
    committed file from hitchhiking with a Pool/Watch publication.
    """
    import fcntl

    root = Path(DATA_ROOT).resolve()
    requested = [paths] if isinstance(paths, str) else list(paths)
    owned = set()
    for raw in requested:
        absolute = Path(raw).resolve()
        if not absolute.is_relative_to(root):
            raise ValueError(f"managed Git path outside data repository: {raw}")
        owned.add(absolute.relative_to(root).as_posix())
    if not owned:
        raise ValueError("no managed Git files supplied")

    def git(*args, check=True, capture_output=False):
        return subprocess.run(
            ["git", *args], cwd=root, check=check, text=True,
            capture_output=capture_output,
        )

    def output(*args):
        return git(*args, capture_output=True).stdout.strip()

    def is_owned(path: str) -> bool:
        return any(path == item or path.startswith(item.rstrip("/") + "/")
                   for item in owned)

    def assert_outgoing_owned() -> None:
        # A pending legacy Pool-only commit can otherwise hitchhike as an
        # owned ancestor of a new Pool+Watch commit. Path ownership alone does
        # not prove the required paired historical publication invariant.
        requires_pairs = (WATCH_COMPLETE in owned and
                          "us/breakout_follow_pool.csv" in owned and
                          watch_was_ever_published(root))
        commits = output("log", "--format=%H", "origin/main..HEAD").splitlines()
        for commit_sha in commits:
            parents = output("show", "-s", "--format=%P", commit_sha).split()
            if len(parents) != 1:
                raise RuntimeError("managed publication refuses unpushed merge commits")
            changed = output("diff-tree", "--no-commit-id", "--name-only", "-r",
                             commit_sha).splitlines()
            foreign = [p for p in changed if not is_owned(p)]
            if foreign:
                raise RuntimeError(
                    f"managed publication cannot push foreign committed files: {foreign[:8]}"
                )
            if requires_pairs and "us/breakout_follow_pool.csv" in changed:
                import io
                snapshot_bytes = subprocess.check_output(
                    ["git", "show", f"{commit_sha}:us/breakout_follow_pool.csv"],
                    cwd=root)
                source_date, codes_digest = pool_identity(
                    pd.read_csv(io.BytesIO(snapshot_bytes), dtype={"code": str}, keep_default_na=False, na_values=[""]))
                message_body = output("show", "-s", "--format=%B", commit_sha)
                if (f"Watch-Source-Date: {source_date}" not in message_body or
                    f"Watch-Codes-SHA256: {codes_digest}" not in message_body):
                    raise RuntimeError(
                        "managed publication refuses unpaired historical Pool ancestor"
                    )

    git_path = output("rev-parse", "--git-path", "bf-pool-watch-publish.lock")
    lock_path = Path(git_path)
    if not lock_path.is_absolute():
        lock_path = root / lock_path
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        try:
            # Fresh remote ref: a stale local origin/main can hide a concurrent
            # external writer. A non-FF push is never forced or rebased blindly.
            git("fetch", "--no-tags", "origin", "main")
            git("merge-base", "--is-ancestor", "origin/main", "HEAD")
            assert_outgoing_owned()
            # This callback runs exactly once under the existing Git lock,
            # after remote refresh and before any official file replacement.
            commit_message = message
            if before_stage is not None:
                prepared_message = before_stage()
                if prepared_message is not None:
                    commit_message = prepared_message

            protected = ("us/breakout_follow_pool.csv",
                         "us/breakout_follow_pool_midweek.csv",
                         WATCH_COMPLETE, WATCH_MIDWEEK)
            if not any(is_owned(p) for p in protected):
                clean = git("diff", "--quiet", "HEAD", "--", *protected, check=False)
                if clean.returncode != 0:
                    raise RuntimeError("another writer left dirty Pool/Watch files")

            pending = output("ls-files", "--",
                             "us/*.pending", "us/.*.pending").splitlines()
            if pending:
                raise RuntimeError(f"tracked Watch pending files: {pending[:8]}")

            # Some optional screener outputs may never have been created.
            # 'git add -A -- missing-untracked-file' is fatal (pathspec does
            # not match), while a *tracked* missing file must still be staged
            # as a deletion. Preserve both cases without broadening ownership.
            staged_paths = [
                path for path in sorted(owned)
                if (root / path).exists() or output("ls-files", "--", path)
            ]
            if staged_paths:
                git("add", "-A", "--", *staged_paths)
                staged = git("diff", "--cached", "--quiet", "--", *staged_paths,
                             check=False)
                if staged.returncode not in (0, 1):
                    raise subprocess.CalledProcessError(staged.returncode, staged.args)
                if staged.returncode == 1:
                    git("commit", "--only", "-m", commit_message, "--", *staged_paths)

            assert_outgoing_owned()
            if output("rev-parse", "HEAD") == output("rev-parse", "origin/main"):
                return
            git("push", "origin", "HEAD:main")
            logging.info("Yfinance_data managed publication completed: %s",
                         ", ".join(sorted(owned)))
        finally:
            fcntl.flock(lock_handle, fcntl.LOCK_UN)


def _commit_pool(pool_path: str, *, extra_paths: list[str] | None = None,
                 before_stage=None) -> None:
    """Commit only a pool that satisfies the current EPS publication contract."""
    if os.path.exists(pool_path):
        pool = pd.read_csv(pool_path, dtype={"code": str}, keep_default_na=False, na_values=[""], encoding="utf-8-sig")
        _validate_eps_publication_contract(pool)

    managed_paths = [pool_path, *(extra_paths or [])]
    pit_path = _pit_store_path()
    if os.path.exists(pit_path):
        managed_paths.append(pit_path)

    message = "Update breakout follow pool"
    if extra_paths:
        source, digest = pool_identity(pool)
        message += f"\n\nWatch-Source-Date: {source}\nWatch-Codes-SHA256: {digest}"
    if before_stage is None:
        _commit_managed_csv(managed_paths, message=message)
    else:
        _commit_managed_csv(managed_paths, message=message,
                            before_stage=before_stage)


def get_download_hold(code: str) -> dict:
    """Query the current published COMPLETE Watch identity and manual reason."""
    published, _, _, _ = published_watch_state(DATA_ROOT)
    row = next((r for r in published if r["code"] == code), None)
    if row is None:
        raise ValueError(f"no official Watch record for {code}")
    return {key: row.get(key, "") for key in (
        "code", "watch_cycle_id", "phase", "download_hold_reason")}


def set_download_hold(code: str, watch_cycle_id: str, reason: str) -> None:
    """Set or clear exactly one manual field in the published active Cycle."""
    from pool_exit_watch_store import ACTIVE_PHASES
    if not code or not watch_cycle_id or not isinstance(reason, str):
        raise ValueError("code, expected watch_cycle_id and string reason are required")
    if len(reason) > 160 or reason != reason.strip() or any(ord(c) < 32 for c in reason):
        raise ValueError("invalid hold reason")
    root = Path(DATA_ROOT)
    target = root / WATCH_COMPLETE

    def prepare_locked_update() -> str:
        # Must use paired authority and verify that local Pool/Watch still
        # exactly match the published version. Never modify dirty snapshots.
        active_watch_codes(root, required=True)
        official, _, source, digest = published_watch_state(root)
        frame = pd.read_csv(target, dtype={"code": str},
                            keep_default_na=False, encoding="utf-8-sig")
        parse_watch_csv(target)
        matched = [i for i, row in frame.iterrows() if
                   row["code"] == code and row["watch_cycle_id"] == watch_cycle_id]
        if len(matched) != 1:
            raise ValueError("official Watch Cycle identity is absent or stale")
        idx = matched[0]
        if frame.at[idx, "phase"] not in ACTIVE_PHASES:
            raise ValueError("only an active official Watch Cycle may be held")
        if not any(r["code"] == code and r["watch_cycle_id"] == watch_cycle_id
                   and r["phase"] in ACTIVE_PHASES for r in official):
            raise ValueError("official Watch Cycle identity changed")
        if frame.at[idx, "download_hold_reason"] == reason:
            return (f"Update BF Watch download hold for {code}\n\n"
                    f"Watch-Source-Date: {source}\nWatch-Codes-SHA256: {digest}")
        frame.at[idx, "download_hold_reason"] = reason
        from pool_exit_watch_store import parse_watch_frame
        parse_watch_frame(frame)
        fd, temp_path = tempfile.mkstemp(prefix=".watch-hold.", suffix=".pending",
                                          dir=str(target.parent))
        os.close(fd)
        try:
            frame.to_csv(temp_path, index=False, encoding="utf-8-sig")
            os.replace(temp_path, target)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        return (f"Update BF Watch download hold for {code}\n\n"
                f"Watch-Source-Date: {source}\nWatch-Codes-SHA256: {digest}")

    # Watch-only owner: never call _commit_pool, which also owns Pool/EPS PIT.
    _commit_managed_csv([str(target)], message="Update BF Watch download hold",
                        before_stage=prepare_locked_update)


__all__ = [
    "BREAKOUT_FOLLOW_POOL_MIDWEEK_PATH",
    "BREAKOUT_FOLLOW_POOL_PATH",
    "IBD_DOUBLE_BOTTOM_SNAPSHOT_MIDWEEK_PATH",
    "IBD_DOUBLE_BOTTOM_SNAPSHOT_PATH",
    "BreakoutFollowPoolKind",
    "BreakoutFollowPoolRun",
    "IbdDoubleBottomSnapshotRun",
    "classify_breakout_follow_pool",
    "complete_snapshot_week",
    "complete_target_week",
    "is_valid_complete_baseline",
    "get_signal_eps",
    "load_industry_lookup",
    "monday_of_week",
    "supplement_latest_pool_signal_eps",
]


def bootstrap_bf_exit_watch(rows: list[dict], recent: dict) -> None:
    """Explicit first publication before next data download."""
    root = Path(DATA_ROOT)
    target = root / WATCH_COMPLETE
    if target.exists() or watch_was_ever_published(root):
        raise RuntimeError("Watch already enabled or needs recovery")
    latest = published_complete_pools(root, limit=2)
    if len(latest) != 2 or latest[0]["ref"] != recent["ref"]:
        raise RuntimeError("bootstrap Pool refs changed")
    when, digest = pool_identity(pd.DataFrame(recent["rows"]))
    official_when, official_digest = pool_identity(pd.DataFrame(latest[0]["rows"]))
    if (when, digest) != (official_when, official_digest):
        raise RuntimeError("bootstrap Pool identity differs from official history")
    # The first Watch has no carried cycles. Its members must be exactly the
    # prior-complete Pool exits, including all former signal=False members.
    old_codes = {str(r["code"]) for r in latest[1]["rows"]}
    recent_codes = {str(r["code"]) for r in latest[0]["rows"]}
    expected = old_codes - recent_codes
    if {str(row.get("code")) for row in rows} != expected:
        raise ValueError("bootstrap Watch members differ from complete Pool exit diff")
    frame = watch_frame(rows, snapshot_date=when, pool_codes_hash=digest)
    fd, pending = tempfile.mkstemp(prefix=".bf_pool_exit_watch.",
                                   suffix=".pending", dir=str(target.parent))
    os.close(fd)
    try:
        frame.to_csv(pending, index=False, encoding="utf-8-sig")
        # Bootstrap bypasses register_companion: apply the exact same official
        # row/event contract *before* any local or remote publication.
        validated_rows = parse_watch_csv(pending)
        if any(r["phase"] == "PROVISIONAL_OUT" for r in validated_rows):
            raise ValueError("bootstrap cannot publish provisional Watch rows")
        os.replace(pending, target)
        _commit_managed_csv(
            [str(target)],
            message=(f"Initialize BF Pool Exit Watch\n\nWatch-Source-Date: {when}"
                     f"\nWatch-Codes-SHA256: {digest}"),
        )
    finally:
        if os.path.exists(pending):
            os.unlink(pending)

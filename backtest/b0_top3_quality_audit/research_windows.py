"""Frozen calendar windows for B0 historical research."""

TRAIN_START = "2025-10-10"
TRAIN_END = "2026-05-22"
CONTAMINATED_VALIDATION_START = "2026-05-29"
CONTAMINATED_VALIDATION_END = "2026-08-07"
PRE_FREEZE_REPLAY_START = "2026-08-14"
PRE_FREEZE_REPLAY_END = "2026-08-21"
FORWARD_SHADOW_START = "2026-08-28"


def in_closed_window(snapshot_date: object, start: str, end: str) -> bool:
    value = str(snapshot_date)[:10]
    return start <= value <= end


def train_dates(snapshot_dates) -> set[str]:
    return {
        str(d)
        for d in snapshot_dates
        if in_closed_window(d, TRAIN_START, TRAIN_END)
    }


def contaminated_validation_dates(snapshot_dates) -> set[str]:
    return {
        str(d)
        for d in snapshot_dates
        if in_closed_window(
            d,
            CONTAMINATED_VALIDATION_START,
            CONTAMINATED_VALIDATION_END,
        )
    }

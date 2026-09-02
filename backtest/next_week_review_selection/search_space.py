from __future__ import annotations

from itertools import product

from .selectors import ReviewRule, SUPPORT_KEYS, rule_complexity


EVIDENCE_PROFILES = {
    "ALL": SUPPORT_KEYS,
    "NO_ENTRY_VOL": tuple(k for k in SUPPORT_KEYS if k != "entry_volume_confirmed"),
    "NO_WEEKLY_VOL": tuple(k for k in SUPPORT_KEYS if k != "weekly_volume_follow_through"),
    "NO_EPS": tuple(k for k in SUPPORT_KEYS if k != "eps_support"),
    "NO_52W": tuple(k for k in SUPPORT_KEYS if k != "near_52w_high"),
    "NO_DRY": tuple(k for k in SUPPORT_KEYS if k != "dry_pullback"),
}


def generate_candidate_rules() -> list[ReviewRule]:
    rules: list[ReviewRule] = []
    for near, statuses, min_support, exclude_geometry, profile in product(
        (3.0, 5.0, 7.0),
        (("UNCONFIRMED",), ("UNCONFIRMED", "BELOW_TRIGGER")),
        (1, 2),
        (True, False),
        tuple(EVIDENCE_PROFILES),
    ):
        status_tag = "U" if statuses == ("UNCONFIRMED",) else "UB"
        geom_tag = "GX" if exclude_geometry else "GA"
        name = f"R_NEAR{near:g}_{status_tag}_S{min_support}_{geom_tag}_{profile}"
        rules.append(
            ReviewRule(
                name=name,
                near_below_pct=near,
                supplemental_statuses=statuses,
                min_support_count=min_support,
                exclude_clear_geometry_failure=exclude_geometry,
                enabled_supports=EVIDENCE_PROFILES[profile],
            )
        )
    return sorted(rules, key=lambda rule: (rule_complexity(rule), rule.name))

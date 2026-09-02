from __future__ import annotations

from itertools import product

from .selectors import EVIDENCE_FAMILIES, ReviewRule, rule_complexity


def generate_core_rules() -> list[ReviewRule]:
    """Stage 1: small structural grid only (24 rules)."""
    rules: list[ReviewRule] = []
    for near, statuses, min_evidence, exclude_geometry in product(
        (3.0, 5.0, 7.0),
        (("UNCONFIRMED",), ("UNCONFIRMED", "BELOW_TRIGGER")),
        (1, 2),
        (False, True),
    ):
        status_tag = "U" if statuses == ("UNCONFIRMED",) else "UB"
        geom_tag = "GA" if not exclude_geometry else "GX"
        name = f"CORE_NEAR{near:g}_{status_tag}_E{min_evidence}_{geom_tag}"
        rules.append(
            ReviewRule(
                name=name,
                near_below_pct=near,
                supplemental_statuses=statuses,
                min_evidence_families=min_evidence,
                exclude_clear_geometry_failure=exclude_geometry,
                enabled_evidence_families=EVIDENCE_FAMILIES,
            )
        )
    return sorted(rules, key=lambda rule: (rule_complexity(rule), rule.name))


def generate_evidence_ablations(core_rule: ReviewRule) -> list[ReviewRule]:
    """Stage 2: only around structural finalists, one family removed at a time."""
    rules = [
        ReviewRule(
            name=f"{core_rule.name}_ALL",
            near_below_pct=core_rule.near_below_pct,
            supplemental_statuses=core_rule.supplemental_statuses,
            min_evidence_families=core_rule.min_evidence_families,
            exclude_clear_geometry_failure=core_rule.exclude_clear_geometry_failure,
            enabled_evidence_families=EVIDENCE_FAMILIES,
        )
    ]
    for family in EVIDENCE_FAMILIES:
        enabled = tuple(item for item in EVIDENCE_FAMILIES if item != family)
        rules.append(
            ReviewRule(
                name=f"{core_rule.name}_NO_{family.upper()}",
                near_below_pct=core_rule.near_below_pct,
                supplemental_statuses=core_rule.supplemental_statuses,
                min_evidence_families=core_rule.min_evidence_families,
                exclude_clear_geometry_failure=core_rule.exclude_clear_geometry_failure,
                enabled_evidence_families=enabled,
            )
        )
    return sorted(rules, key=lambda rule: (rule_complexity(rule), rule.name))

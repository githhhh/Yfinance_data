from __future__ import annotations


def hypothesis_space() -> list[dict[str, object]]:
    """Candidate factor/rule hypotheses RD-Agent may explore before any skill absorption."""
    return [
        {
            "name": "fresh_demand_proximity_first",
            "question": "After entry volume is clearly sufficient, should buy-point proximity outrank raw volume magnitude?",
            "features": [
                "current_vs_ibd_candidate_pct",
                "ibd_entry_volume_ratio_pass",
                "geometry_caution_not_failure",
                "weekly_volume_follow_through",
                "eps_state",
            ],
            "official_skill_absorption": "candidate_only",
        },
        {
            "name": "pullback_vcp_lane_interleave",
            "question": "Should high-quality pullback/VCP lanes interleave with Fresh Demand instead of always ranking behind it?",
            "features": [
                "ibd_candidate_rule",
                "pullback_pct",
                "pullback_duration_weeks",
                "pullback_v_is_dry",
                "dist_to_52w_high_pct",
                "volume_ratio",
            ],
            "official_skill_absorption": "candidate_only",
        },
        {
            "name": "displacement_quality_proxy",
            "question": "Can breakout geometry proxy displacement without adding chart-derived subjective labels?",
            "features": [
                "ibd_entry_close_position",
                "ibd_entry_breakout_range_ratio",
                "ibd_entry_close_vs_trigger_pct",
                "ibd_entry_volume_ratio_pass",
            ],
            "official_skill_absorption": "audit_only",
        },
        {
            "name": "shadow_status_recovery",
            "question": "Which UNCONFIRMED and EXTENDED signal states deserve audit-only elevation into Signal Shadow?",
            "features": [
                "ibd_entry_status",
                "current_vs_ibd_candidate_pct",
                "ibd_candidate_rule",
                "volume_ratio",
                "eps_state",
            ],
            "official_skill_absorption": "audit_only",
        },
    ]

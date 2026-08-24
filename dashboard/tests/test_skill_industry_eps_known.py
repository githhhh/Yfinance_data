import pandas as pd

from dashboard.skill_industry_eps_known import select_skill_industry_eps_known


def _row(
    code,
    *,
    industry,
    eps,
    current_pct,
    entry_volume,
    close_position,
    range_ratio,
    weekly_volume,
    dist_52w,
    rule="ceiling",
    dry=None,
):
    return {
        "snapshot_date": "2026-08-14",
        "code": code,
        "signal": True,
        "ibd_candidate_rule": rule,
        "ibd_entry_status": "ACTIONABLE",
        "industry": industry,
        "eps_yoy_growth": eps,
        "current_vs_ibd_candidate_pct": current_pct,
        "ibd_entry_volume_ratio": entry_volume,
        "ibd_entry_close_position": close_position,
        "ibd_entry_breakout_range_ratio": range_ratio,
        "volume_ratio": weekly_volume,
        "dist_to_52w_high_pct": dist_52w,
        "pullback_v_is_dry": dry,
    }


def test_skill_industry_eps_known_selects_20260814_expected_order():
    pool = pd.DataFrame(
        [
            _row(
                "ABNB",
                industry="Hotels/Resorts/Cruise lines",
                eps=33.060748,
                current_pct=3.13,
                entry_volume=2.502220,
                close_position=0.849299,
                range_ratio=0.726636,
                weekly_volume=1.603177,
                dist_52w=-1.635314,
                rule="ma10_touch_confirm",
            ),
            _row(
                "HTFL",
                industry="Packaged Software",
                eps=-65.185857,
                current_pct=3.70,
                entry_volume=6.516050,
                close_position=0.869065,
                range_ratio=0.215827,
                weekly_volume=1.758315,
                dist_52w=-2.139535,
            ),
            _row(
                "MTW",
                industry="Trucks/Construction/Farm Machinery",
                eps=827.684964,
                current_pct=4.39,
                entry_volume=1.572212,
                close_position=0.884211,
                range_ratio=0.368421,
                weekly_volume=1.946689,
                dist_52w=-3.092784,
            ),
            _row(
                "UMH",
                industry="Real Estate Investment Trusts",
                eps=72.575251,
                current_pct=1.73,
                entry_volume=1.649871,
                close_position=0.833333,
                range_ratio=0.500000,
                weekly_volume=1.388225,
                dist_52w=-1.260504,
            ),
        ]
    )

    selected = select_skill_industry_eps_known(pool)

    assert [item.code for item in selected] == ["UMH", "MTW", "HTFL"]


def test_skill_industry_eps_known_requires_eps_known_and_industry_cover():
    pool = pd.DataFrame(
        [
            _row(
                "AAA",
                industry="Same Industry",
                eps=30.0,
                current_pct=1.0,
                entry_volume=2.0,
                close_position=0.90,
                range_ratio=0.70,
                weekly_volume=1.5,
                dist_52w=-1.0,
            ),
            _row(
                "AAB",
                industry="Same Industry",
                eps=80.0,
                current_pct=1.0,
                entry_volume=3.0,
                close_position=0.90,
                range_ratio=0.70,
                weekly_volume=1.5,
                dist_52w=-1.0,
            ),
            _row(
                "AAC",
                industry="Other Industry",
                eps=pd.NA,
                current_pct=1.0,
                entry_volume=4.0,
                close_position=0.90,
                range_ratio=0.70,
                weekly_volume=1.5,
                dist_52w=-1.0,
            ),
            _row(
                "AAD",
                industry="Third Industry",
                eps=-10.0,
                current_pct=1.0,
                entry_volume=1.6,
                close_position=0.90,
                range_ratio=0.70,
                weekly_volume=1.5,
                dist_52w=-1.0,
            ),
        ]
    )

    selected = select_skill_industry_eps_known(pool)

    assert [item.code for item in selected] == ["AAB", "AAD"]

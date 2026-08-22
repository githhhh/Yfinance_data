import pandas as pd

from eps_pit.lookup import enrich_pool_with_signal_eps


def test_signal_eps_enrichment_uses_pit_then_stage2_for_signal_rows(tmp_path):
    pit_path = tmp_path / "signal_eps_pit.csv"
    stage2_path = tmp_path / "stage2_whitelist.csv"

    pd.DataFrame(
        [
            {
                "snapshot_date": "2026-08-14",
                "code": "PIT",
                "eps_yoy_growth": 31.5,
            },
        ]
    ).to_csv(pit_path, index=False)
    pd.DataFrame(
        [
            {"code": "STAGE", "eps_yoy_growth": 42.0},
            {"code": "QUIET", "eps_yoy_growth": 88.0},
        ]
    ).to_csv(stage2_path, index=False)

    pool = pd.DataFrame(
        [
            {"snapshot_date": "2026-08-14", "code": "PIT", "signal": True, "eps_yoy_growth": pd.NA},
            {"snapshot_date": "2026-08-14", "code": "STAGE", "signal": True, "eps_yoy_growth": pd.NA},
            {"snapshot_date": "2026-08-14", "code": "QUIET", "signal": False, "eps_yoy_growth": pd.NA},
            {"snapshot_date": "2026-08-14", "code": "MISS", "signal": True, "eps_yoy_growth": pd.NA},
        ]
    )

    enriched = enrich_pool_with_signal_eps(
        pool,
        csv_path=str(pit_path),
        stage2_path=str(stage2_path),
    )

    assert enriched.loc[0, "eps_yoy_growth"] == 31.5
    assert enriched.loc[0, "eps_yoy_growth_repair_method"] == "pit_signal_supplement"
    assert enriched.loc[1, "eps_yoy_growth"] == 42.0
    assert enriched.loc[1, "eps_yoy_growth_repair_method"] == "stage2_current_snapshot"
    assert pd.isna(enriched.loc[2, "eps_yoy_growth"])
    assert pd.isna(enriched.loc[3, "eps_yoy_growth"])

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest.rd_agent_research_bench.hypotheses import hypothesis_space
from backtest.rd_agent_research_bench.research import (
    DEFAULT_ORACLE_DIR,
    absorption_candidate_matrix,
    imax_rank_audit,
    intuitive_variant_summary,
    load_oracle_tables,
    pair_outcome_audit,
    render_markdown_report,
    rule_status_coverage,
    summarize_variant_quality,
    stop_loss_capital_backtest,
)


DEFAULT_OUTPUT_DIR = Path("backtest/rd_agent_research_bench/output")
DEFAULT_VARIANTS = [
    "v3_core_top3",
    "skill_industry_eps_known",
    "clean_eps_pass_no_dry_no_geom_caution",
    "signal_shadow_top3",
    "research_fresh_demand_proximity_first",
    "research_pullback_vcp_lane_interleave",
    "research_proximity_structural_floor_guard",
    "research_proximity_eps_known_floor_guard",
    "research_proximity_eps_pass_floor_guard",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build auditable RD-Agent research bench reports from weekly signal oracle outputs.")
    parser.add_argument("--oracle-dir", default=str(DEFAULT_ORACLE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    args = parser.parse_args(argv)

    oracle_dir = Path(args.oracle_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]

    quality_rows = []
    coverage_rows = []
    imax_rows = []
    pair_rows = []
    intuitive_rows = []
    stop_loss_rows = []
    for eps_mode in ["no_eps", "with_eps"]:
        universe, weekly, picks = load_oracle_tables(eps_mode, oracle_dir=oracle_dir)
        for variant in variants:
            quality_rows.append(summarize_variant_quality(eps_mode, variant, universe, weekly, picks))
            intuitive_rows.append(intuitive_variant_summary(weekly, eps_mode=eps_mode, variant=variant))
            stop_loss_rows.append(stop_loss_capital_backtest(picks, eps_mode=eps_mode, variant=variant))
            variant_picks = picks[picks["variant"].astype(str).eq(variant)].copy()
            coverage = rule_status_coverage(universe, variant_picks)
            coverage.insert(0, "variant", variant)
            coverage.insert(0, "eps_mode", eps_mode)
            coverage_rows.extend(coverage.to_dict("records"))
        imax = imax_rank_audit(universe, picks)
        imax.insert(0, "eps_mode", eps_mode)
        imax_rows.extend(imax.to_dict("records"))
        pair = pair_outcome_audit(universe, picks, snapshot_date="2026-07-24", codes=("BLFS", "IMAX"))
        if not pair.empty:
            pair.insert(0, "eps_mode", eps_mode)
            pair_rows.extend(pair.to_dict("records"))

    quality = pd.DataFrame(quality_rows)
    coverage = pd.DataFrame(coverage_rows)
    imax = pd.DataFrame(imax_rows)
    pair_audit = pd.DataFrame(pair_rows)
    intuitive = pd.DataFrame(intuitive_rows)
    stop_loss = pd.DataFrame(stop_loss_rows)
    absorption_parts = []
    with_eps_absorption = absorption_candidate_matrix(
        quality,
        eps_mode="with_eps",
        baseline_variant="skill_industry_eps_known",
    )
    if not with_eps_absorption.empty:
        absorption_parts.append(with_eps_absorption)
    no_eps_absorption = absorption_candidate_matrix(
        quality,
        eps_mode="no_eps",
        baseline_variant="v3_core_top3",
    )
    if not no_eps_absorption.empty:
        absorption_parts.append(no_eps_absorption)
    absorption = pd.concat(absorption_parts, ignore_index=True, sort=False) if absorption_parts else pd.DataFrame()
    hypotheses = pd.DataFrame(hypothesis_space())

    quality.to_csv(output_dir / "variant_quality_summary.csv", index=False)
    coverage.to_csv(output_dir / "rule_status_coverage.csv", index=False)
    imax.to_csv(output_dir / "imax_rank_audit.csv", index=False)
    pair_audit.to_csv(output_dir / "pair_outcome_audit.csv", index=False)
    intuitive.to_csv(output_dir / "intuitive_variant_summary.csv", index=False)
    stop_loss.to_csv(output_dir / "stop_loss_capital_replay.csv", index=False)
    absorption.to_csv(output_dir / "candidate_absorption_matrix.csv", index=False)
    hypotheses.to_csv(output_dir / "rd_agent_hypothesis_space.csv", index=False)
    (output_dir / "rd_agent_hypothesis_space.json").write_text(
        json.dumps(hypothesis_space(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = render_markdown_report(
        quality,
        imax,
        coverage,
        pair_audit=pair_audit,
        intuitive_summary=intuitive,
        absorption_matrix=absorption,
        stop_loss_backtest=stop_loss,
    )
    (output_dir / "research_bench_report.md").write_text(report, encoding="utf-8")
    manifest = {
        "oracle_dir": str(oracle_dir),
        "output_dir": str(output_dir),
        "variants": variants,
        "outputs": {
            "quality": str(output_dir / "variant_quality_summary.csv"),
            "coverage": str(output_dir / "rule_status_coverage.csv"),
            "imax": str(output_dir / "imax_rank_audit.csv"),
            "pair_audit": str(output_dir / "pair_outcome_audit.csv"),
            "intuitive": str(output_dir / "intuitive_variant_summary.csv"),
            "stop_loss": str(output_dir / "stop_loss_capital_replay.csv"),
            "absorption": str(output_dir / "candidate_absorption_matrix.csv"),
            "hypotheses_csv": str(output_dir / "rd_agent_hypothesis_space.csv"),
            "hypotheses_json": str(output_dir / "rd_agent_hypothesis_space.json"),
            "report": str(output_dir / "research_bench_report.md"),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

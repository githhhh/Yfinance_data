from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest.rd_agent_research_bench.hypotheses import hypothesis_space
from backtest.rd_agent_research_bench.research import (
    DEFAULT_ORACLE_DIR,
    imax_rank_audit,
    load_oracle_tables,
    render_markdown_report,
    rule_status_coverage,
    summarize_variant_quality,
)


DEFAULT_OUTPUT_DIR = Path("backtest/rd_agent_research_bench/output")
DEFAULT_VARIANTS = [
    "v3_core_top3",
    "skill_industry_eps_known",
    "clean_eps_pass_no_dry_no_geom_caution",
    "signal_shadow_top3",
    "research_fresh_demand_proximity_first",
    "research_pullback_vcp_lane_interleave",
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
    for eps_mode in ["no_eps", "with_eps"]:
        universe, weekly, picks = load_oracle_tables(eps_mode, oracle_dir=oracle_dir)
        for variant in variants:
            quality_rows.append(summarize_variant_quality(eps_mode, variant, universe, weekly, picks))
            variant_picks = picks[picks["variant"].astype(str).eq(variant)].copy()
            coverage = rule_status_coverage(universe, variant_picks)
            coverage.insert(0, "variant", variant)
            coverage.insert(0, "eps_mode", eps_mode)
            coverage_rows.extend(coverage.to_dict("records"))
        imax = imax_rank_audit(universe, picks)
        imax.insert(0, "eps_mode", eps_mode)
        imax_rows.extend(imax.to_dict("records"))

    quality = pd.DataFrame(quality_rows)
    coverage = pd.DataFrame(coverage_rows)
    imax = pd.DataFrame(imax_rows)
    hypotheses = pd.DataFrame(hypothesis_space())

    quality.to_csv(output_dir / "variant_quality_summary.csv", index=False)
    coverage.to_csv(output_dir / "rule_status_coverage.csv", index=False)
    imax.to_csv(output_dir / "imax_rank_audit.csv", index=False)
    hypotheses.to_csv(output_dir / "rd_agent_hypothesis_space.csv", index=False)
    (output_dir / "rd_agent_hypothesis_space.json").write_text(
        json.dumps(hypothesis_space(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = render_markdown_report(quality, imax, coverage)
    (output_dir / "research_bench_report.md").write_text(report, encoding="utf-8")
    manifest = {
        "oracle_dir": str(oracle_dir),
        "output_dir": str(output_dir),
        "variants": variants,
        "outputs": {
            "quality": str(output_dir / "variant_quality_summary.csv"),
            "coverage": str(output_dir / "rule_status_coverage.csv"),
            "imax": str(output_dir / "imax_rank_audit.csv"),
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

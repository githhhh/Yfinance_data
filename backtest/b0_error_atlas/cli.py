from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import OUT, PROTOCOL_VERSION
from .data import sha256_file
from .report import write_report
from .study import run_study

REPORT = OUT / "B0_ERROR_ATLAS_REPORT.md"
MANIFEST = OUT / "run_manifest.json"
ANALYSIS_FRAME = OUT / "error_atlas_rows.parquet"

CSV_OUTPUTS = {
    "label_summary": "label_summary.csv",
    "task_support_by_quarter": "task_support_by_quarter.csv",
    "numeric_feature_separation": "numeric_feature_separation.csv",
    "numeric_quarter_stability": "numeric_quarter_stability.csv",
    "categorical_feature_separation": "categorical_feature_separation.csv",
    "categorical_level_rates": "categorical_level_rates.csv",
    "model_fold_results": "model_fold_results.csv",
    "model_summary": "model_summary.csv",
    "pair_interactions": "pair_interactions.csv",
    "tree_permutation_importance": "tree_permutation_importance.csv",
    "high_corr_pairs": "high_corr_pairs.csv",
    "missed_winner_examples": "missed_winner_examples.csv",
    "bad_selected_examples": "bad_selected_examples.csv",
    "reject_reason_outcomes": "reject_reason_outcomes.csv",
}


def _clear_output() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in OUT.iterdir():
        if path.is_file():
            path.unlink()


def _write_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def materialize() -> None:
    _clear_output()
    result = run_study()
    outputs = result["outputs"]
    manifest = dict(result["manifest"])

    analysis = outputs["analysis_frame"]
    analysis.to_parquet(ANALYSIS_FRAME, index=False, engine="pyarrow")

    artifacts: list[Path] = [ANALYSIS_FRAME]
    for key, filename in CSV_OUTPUTS.items():
        path = OUT / filename
        _write_frame(outputs[key], path)
        artifacts.append(path)

    write_report(REPORT, outputs=outputs, manifest=manifest)
    artifacts.append(REPORT)

    manifest.update({
        "protocol_version": PROTOCOL_VERSION,
        "artifact_semantics": {
            "error_atlas_rows.parquet": (
                "Full frozen panel plus current B0 state, PIT-derived features, "
                "path labels, and next-open W4 outcomes."
            ),
            "numeric_feature_separation.csv": (
                "Full-history descriptive univariate separation only; not OOS."
            ),
            "model_fold_results.csv": (
                "Expanding-quarter chronological evaluation; no random row split."
            ),
            "pair_interactions.csv": (
                "Exploratory pair hypotheses; pair candidates preselected using "
                "full-history univariate diagnostics."
            ),
        },
        "production_change": False,
        "decision_policy": (
            "No automatic PASS/FAIL threshold. Interpret label support, chronological "
            "AUC stability, pair synergy, and feature redundancy jointly."
        ),
    })
    manifest["artifacts"] = {
        path.name: sha256_file(path)
        for path in artifacts
        if path.exists()
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    labels = outputs["label_summary"]
    models = outputs["model_summary"]
    print("=== B0 Error Atlas v1 ===")
    print(f"source={manifest['source_git_sha']}")
    print(f"rows={manifest['rows']} weeks={manifest['weeks']}")
    for _, row in labels.iterrows():
        if pd.isna(row.get("positive_rows")):
            continue
        print(
            f"task={row['task']} rows={int(row['rows'])} "
            f"pos={int(row['positive_rows'])} neg={int(row['negative_rows'])}"
        )
    if not models.empty:
        best = models.sort_values("mean_roc_auc", ascending=False).iloc[0]
        print(
            "best_chronological_model="
            f"{best['task']} {best['feature_set']} {best['model']} "
            f"mean_auc={best['mean_roc_auc']:.4f} "
            f"min_auc={best['min_roc_auc']:.4f}"
        )
    print(f"report={REPORT}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B0 Error Atlas / FN Recovery / FP Veto research"
    )
    parser.add_argument("command", choices=["materialize"])
    args = parser.parse_args()
    if args.command == "materialize":
        materialize()


if __name__ == "__main__":
    main()

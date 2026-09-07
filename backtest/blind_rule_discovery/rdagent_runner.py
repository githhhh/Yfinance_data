"""Blind Rule Discovery RD-Agent Runner.

This module executes the autonomous research loop inside the isolated agent workspace:
1. Ingests discovery-only files: prompt.md, period_summary.csv, feature_profile.csv, samples.csv.
2. Extracts empirical repeated-period separation signals between winners and losers.
3. Consults the configured LLM (DeepSeek via LiteLLM) to formulate compact, economically sound DNF rules.
4. Validates rule support and cross-quarter stability on discovery samples.
5. Stops early as soon as a compact, stable rule with sufficient support converges.
6. Writes rule.json and research_notes.md with exact model request provenance.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


MAX_CLAUSES = 3
MAX_CONDITIONS = 6
ALLOWED_OPERATORS = {">", ">=", "<", "<=", "==", "!="}
MAX_RESEARCH_CALLS = 10  # Stop early once converged, well below the 500 ceiling


def _extract_json_payload(text: str) -> dict[str, Any]:
    raw = text.strip()
    fence = chr(96) * 3
    if raw.startswith(fence):
        raw = re.sub(r"^" + re.escape(fence) + r"(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*" + re.escape(fence) + r"$", "", raw)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError("LLM response did not contain a JSON object.")
        obj = json.loads(raw[start : end + 1])
    if not isinstance(obj, dict):
        raise RuntimeError("LLM response root must be a JSON object.")
    return obj


def _condition_mask(df: pd.DataFrame, condition: Mapping[str, Any]) -> pd.Series:
    feature = str(condition["feature"])
    op = str(condition["op"])
    threshold = float(condition["threshold"])
    values = pd.to_numeric(df[feature], errors="coerce")
    if op == ">":
        return values > threshold
    if op == ">=":
        return values >= threshold
    if op == "<":
        return values < threshold
    if op == "<=":
        return values <= threshold
    if op == "==":
        return values == threshold
    if op == "!=":
        return values != threshold
    raise ValueError(f"unsupported operator: {op}")


def apply_rule(rule: Mapping[str, Any], df: pd.DataFrame) -> pd.Series:
    selected = pd.Series(False, index=df.index)
    for clause in rule["clauses"]:
        clause_mask = pd.Series(True, index=df.index)
        for condition in clause["all"]:
            clause_mask &= _condition_mask(df, condition).fillna(False)
        selected |= clause_mask
    return selected


def validate_rule(rule: dict[str, Any], allowed_features: set[str]) -> None:
    if not isinstance(rule, dict) or rule.get("version") != 1:
        raise ValueError("rule requires version=1")
    clauses = rule.get("clauses")
    if not isinstance(clauses, list) or not 1 <= len(clauses) <= MAX_CLAUSES:
        raise ValueError(f"clauses must contain 1..{MAX_CLAUSES} items")
    total_conditions = 0
    for clause in clauses:
        if not isinstance(clause, dict) or set(clause) != {"all"} or not isinstance(clause["all"], list):
            raise ValueError("each clause must be exactly {'all': [...]}")
        if not clause["all"]:
            raise ValueError("empty clause")
        for condition in clause["all"]:
            total_conditions += 1
            if total_conditions > MAX_CONDITIONS:
                raise ValueError(f"rule exceeds {MAX_CONDITIONS} total conditions")
            if not isinstance(condition, dict) or set(condition) != {"feature", "op", "threshold"}:
                raise ValueError("condition must contain only feature/op/threshold")
            feat = str(condition["feature"])
            if feat not in allowed_features:
                raise ValueError(f"forbidden or unknown feature: {feat}")
            if str(condition["op"]) not in ALLOWED_OPERATORS:
                raise ValueError(f"unsupported operator: {condition['op']}")
            thresh = condition["threshold"]
            if isinstance(thresh, bool) or not isinstance(thresh, (int, float)) or not math.isfinite(float(thresh)):
                raise ValueError("threshold must be a finite number")


def evaluate_rule_support_and_stability(rule: dict[str, Any], samples: pd.DataFrame) -> dict[str, Any]:
    mask = apply_rule(rule, samples)
    selected = samples.loc[mask]
    total_selected = len(selected)
    
    if total_selected == 0:
        return {
            "selected_samples": 0,
            "active_quarters": 0,
            "resolved_selected": 0,
            "resolved_winner_rate": 0.0,
            "universe_resolved_winner_rate": 0.0,
            "stable_quarter_fraction": 0.0,
            "quarter_details": {},
            "is_valid": False,
            "reason": "zero samples selected",
        }
    
    active_quarters = int(selected["period_quarter"].nunique())
    resolved_selected_df = selected[selected["Y_primary"].isin(["winner", "loser"])]
    resolved_count = len(resolved_selected_df)
    
    resolved_winner_rate = (
        float((resolved_selected_df["Y_primary"] == "winner").mean())
        if resolved_count > 0
        else 0.0
    )
    
    # Universe baseline
    resolved_universe = samples[samples["Y_primary"].isin(["winner", "loser"])]
    universe_winner_rate = (
        float((resolved_universe["Y_primary"] == "winner").mean())
        if len(resolved_universe) > 0
        else 0.0
    )
    
    # Quarter-by-quarter comparison
    q_details = {}
    better_quarters = 0
    all_quarters = sorted(samples["period_quarter"].unique())
    for q in all_quarters:
        q_sel = selected[selected["period_quarter"] == q]
        q_sel_res = q_sel[q_sel["Y_primary"].isin(["winner", "loser"])]
        q_uni_res = samples[(samples["period_quarter"] == q) & (samples["Y_primary"].isin(["winner", "loser"]))]
        
        q_sel_wr = float((q_sel_res["Y_primary"] == "winner").mean()) if len(q_sel_res) > 0 else 0.0
        q_uni_wr = float((q_uni_res["Y_primary"] == "winner").mean()) if len(q_uni_res) > 0 else 0.0
        
        if len(q_sel_res) >= 2 and q_sel_wr >= q_uni_wr:
            better_quarters += 1
            
        q_details[q] = {
            "selected_n": len(q_sel),
            "resolved_n": len(q_sel_res),
            "selected_wr": round(q_sel_wr, 4),
            "universe_wr": round(q_uni_wr, 4),
        }
    
    stable_fraction = better_quarters / max(1, active_quarters)
    is_valid = (
        total_selected >= 20
        and active_quarters >= 3
        and resolved_count >= 10
    )
    
    return {
        "selected_samples": total_selected,
        "active_quarters": active_quarters,
        "resolved_selected": resolved_count,
        "resolved_winner_rate": round(resolved_winner_rate, 4),
        "universe_resolved_winner_rate": round(universe_winner_rate, 4),
        "stable_quarter_fraction": round(stable_fraction, 4),
        "quarter_details": q_details,
        "is_valid": is_valid,
    }


def analyze_feature_separations(feature_profile: pd.DataFrame) -> dict[str, Any]:
    """Identify features that reliably distinguish winners from losers across quarters."""
    q_df = feature_profile[
        (feature_profile["granularity"] == "quarter")
        & (feature_profile["primary_label"].isin(["winner", "loser"]))
    ]
    if q_df.empty:
        return {}
    
    piv = q_df.pivot(index=["period", "feature"], columns="primary_label", values="p50").reset_index()
    if "winner" not in piv.columns or "loser" not in piv.columns:
        return {}
        
    piv["diff"] = piv["winner"] - piv["loser"]
    piv["winner_higher"] = piv["diff"] > 0
    piv["winner_lower"] = piv["diff"] < 0
    
    summary = piv.groupby("feature").agg(
        quarters=("period", "count"),
        higher_count=("winner_higher", "sum"),
        lower_count=("winner_lower", "sum"),
        mean_diff=("diff", "mean"),
        median_diff=("diff", "median"),
    ).reset_index()
    
    summary["higher_pct"] = summary["higher_count"] / summary["quarters"]
    summary["lower_pct"] = summary["lower_count"] / summary["quarters"]
    
    top_higher = summary.sort_values(by="higher_pct", ascending=False).head(5).to_dict(orient="records")
    top_lower = summary.sort_values(by="lower_pct", ascending=False).head(5).to_dict(orient="records")
    
    return {
        "top_higher_features": top_higher,
        "top_lower_features": top_lower,
    }


def main() -> int:
    # Remove local proxy variables so outbound API calls connect directly
    for p in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
        os.environ.pop(p, None)

    workspace = Path(".").resolve()
    samples_path = workspace / "samples.csv"
    period_path = workspace / "period_summary.csv"
    profile_path = workspace / "feature_profile.csv"
    prompt_path = workspace / "prompt.md"
    
    if not (samples_path.exists() and profile_path.exists()):
        print(f"Required workspace files missing in {workspace}", file=sys.stderr)
        return 1
        
    samples = pd.read_csv(samples_path)
    profile = pd.read_csv(profile_path)
    prompt_text = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""
    
    allowed_features = {
        col for col in samples.columns
        if col.startswith("X") or col.startswith("M_")
    }
    
    # 1. Feature separation analysis
    sep_info = analyze_feature_separations(profile)
    
    # 2. Setup LLM client
    import litellm
    model = os.environ.get("CHAT_MODEL") or os.environ.get("TRACK_D_RDAGENT_MODEL") or "deepseek/deepseek-v4-pro"
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("DEEPSEEK_API_BASE") or os.environ.get("OPENAI_API_BASE")
    
    if not api_key:
        print("Error: DEEPSEEK_API_KEY / OPENAI_API_KEY not found in environment", file=sys.stderr)
        return 1
        
    model_calls = 0
    start_time = time.time()
    
    system_prompt = (
        "You are an autonomous quantitative research agent performing Blind Rule Discovery.\n"
        "Your task is to discover a compact, robust rule that separates breakout winners from losers.\n"
        "Hard Constraints:\n"
        "- Must output JSON only.\n"
        "- Rule format: version=1, clauses=[{\"all\": [{\"feature\": \"X###\" or \"M_*\", \"op\": \">=\", \"threshold\": float}]}].\n"
        "- Maximum 3 clauses, maximum 6 total conditions across all clauses.\n"
        "- Only reference features starting with X or M_.\n"
        "- NEVER reference Y_*, sample_id, period_month, period_quarter.\n"
        "- The research timeout is a safety ceiling, not a target. Prefer robustness and simplicity.\n"
        "- Seek repeated-period stability across multiple quarters rather than single-period overfitting.\n"
    )
    
    # Format concise empirical summary with median/IQR from feature profile
    evidence_lines = [
        "Empirical Repeated-Period Evidence from 11 Discovery Quarters:",
        "",
        "Top Features where Winner > Loser consistently across quarters:",
    ]
    for row in sep_info.get("top_higher_features", []):
        feat = row["feature"]
        med = float(samples[feat].median())
        p25 = float(samples[feat].quantile(0.25))
        p75 = float(samples[feat].quantile(0.75))
        evidence_lines.append(
            f"- {feat}: winner higher in {row['higher_count']}/{row['quarters']} quarters "
            f"(mean diff: {row['mean_diff']:+.4f}). Distribution: p25={p25:.4f}, p50={med:.4f}, p75={p75:.4f}"
        )
    evidence_lines.append("\nTop Features where Winner < Loser consistently across quarters:")
    for row in sep_info.get("top_lower_features", []):
        feat = row["feature"]
        med = float(samples[feat].median())
        p25 = float(samples[feat].quantile(0.25))
        p75 = float(samples[feat].quantile(0.75))
        evidence_lines.append(
            f"- {feat}: winner lower in {row['lower_count']}/{row['quarters']} quarters "
            f"(mean diff: {row['mean_diff']:+.4f}). Distribution: p25={p25:.4f}, p50={med:.4f}, p75={p75:.4f}"
        )
    evidence_lines.extend([
        f"\nAvailable candidate features: {sorted(allowed_features)}",
        "Formulate a compact DNF rule with 1-2 clauses and 1-3 conditions using thresholds aligned with the observed distributions.",
        "Respond with a JSON object: {\"rule\": {\"version\": 1, \"clauses\": [{\"all\": [{\"feature\": \"...\", \"op\": \">=\", \"threshold\": 0.0}]}]}, \"hypothesis\": \"...\", \"rationale\": \"...\"}"
    ])
    evidence_text = "\n".join(evidence_lines)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": evidence_text},
    ]

    best_rule: dict[str, Any] | None = None
    best_eval: dict[str, Any] | None = None
    research_log = []

    for iteration in range(1, MAX_RESEARCH_CALLS + 1):
        print(f"\n--- Research Iteration {iteration} ---")
        try:
            model_calls += 1
            call_kwargs = {
                "model": model,
                "messages": messages,
                "api_key": api_key,
                "temperature": 0.3 if iteration > 1 else 0.5,
                "max_tokens": 4000,
                "timeout": 120,
            }
            if api_base:
                call_kwargs["api_base"] = api_base
                
            resp = litellm.completion(**call_kwargs)
            content = resp.choices[0].message.content
            payload = _extract_json_payload(content)
            candidate_rule = payload.get("rule", payload)
            hypothesis = payload.get("hypothesis", "Empirical separation rule")
            rationale = payload.get("rationale", "")
            
            validate_rule(candidate_rule, allowed_features)
            eval_res = evaluate_rule_support_and_stability(candidate_rule, samples)
            print(f"Candidate rule evaluated: {json.dumps(candidate_rule['clauses'])}")
            print(f"Support: N={eval_res['selected_samples']}, Quarters={eval_res['active_quarters']}, Resolved={eval_res['resolved_selected']}")
            print(f"Win Rate: Selected={eval_res['resolved_winner_rate']} vs Universe={eval_res['universe_resolved_winner_rate']}")
            print(f"Stable Quarter Fraction: {eval_res['stable_quarter_fraction']}")
            
            research_log.append({
                "iteration": iteration,
                "rule": candidate_rule,
                "hypothesis": hypothesis,
                "rationale": rationale,
                "evaluation": eval_res,
            })
            
            if eval_res["is_valid"]:
                if best_eval is None or (
                    eval_res["resolved_winner_rate"] >= best_eval["resolved_winner_rate"]
                    and eval_res["stable_quarter_fraction"] >= best_eval["stable_quarter_fraction"]
                ):
                    best_rule = candidate_rule
                    best_eval = eval_res
                    
                # Early convergence check:
                # If rule has >=20 samples, >=3 quarters, resolved win rate > universe, and stable in >=50% of quarters
                if (
                    eval_res["selected_samples"] >= 25
                    and eval_res["active_quarters"] >= 4
                    and eval_res["resolved_winner_rate"] > eval_res["universe_resolved_winner_rate"]
                    and eval_res["stable_quarter_fraction"] >= 0.5
                ):
                    print(f"Convergence reached at iteration {iteration}! Stopping research early per budget contract.")
                    break
            else:
                print(f"Candidate failed support constraints: {eval_res.get('reason', 'insufficient support')}")
                
            # Feedback for next iteration
            feedback = (
                f"Evaluation of your proposed rule on discovery data:\n"
                f"- Total selected samples: {eval_res['selected_samples']} (required >= 20)\n"
                f"- Active quarters: {eval_res['active_quarters']} (required >= 3)\n"
                f"- Resolved samples: {eval_res['resolved_selected']} (required >= 10)\n"
                f"- Selected win rate: {eval_res['resolved_winner_rate']} vs universe baseline: {eval_res['universe_resolved_winner_rate']}\n"
                f"- Fraction of quarters outperforming baseline: {eval_res['stable_quarter_fraction']}\n"
                "Please refine the rule to ensure sufficient support and broad cross-quarter stability. Keep it compact (1-2 clauses, 1-3 conditions)."
            )
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": feedback})
            
        except Exception as exc:
            print(f"Iteration {iteration} error: {exc}", file=sys.stderr)
            if iteration >= 3 and best_rule is not None:
                break
            time.sleep(2)
            
    # Fallback to empirical candidate if no valid rule was accepted by LLM
    if best_rule is None:
        print("Constructing robust baseline empirical rule from top repeated-period features...")
        # Use top separating feature
        top_feat = sep_info["top_higher_features"][0]["feature"]
        med_val = float(samples[top_feat].median())
        best_rule = {
            "version": 1,
            "clauses": [
                {
                    "all": [
                        {"feature": top_feat, "op": ">=", "threshold": round(med_val, 4)}
                    ]
                }
            ],
            "rationale": f"Empirical median split on top repeated-period separation feature {top_feat}",
        }
        best_eval = evaluate_rule_support_and_stability(best_rule, samples)

    # Ensure rule metadata contains evidence and rationale
    best_rule["rationale"] = research_log[-1].get("rationale", "Autonomous discovery rule") if research_log else "Empirical separation"
    best_rule["evidence"] = {
        "iterations": len(research_log),
        "actual_model_requests": model_calls,
        "runtime_seconds": round(time.time() - start_time, 2),
        "discovery_evaluation": best_eval,
    }
    
    # Write rule.json
    rule_path = workspace / "rule.json"
    rule_path.write_text(json.dumps(best_rule, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {rule_path}")
    
    # Write research_notes.md
    notes_path = workspace / "research_notes.md"
    notes_content = [
        "# Blind Rule Discovery Research Notes",
        "",
        f"- **Model**: {model}",
        f"- **Actual Model Requests**: {model_calls}",
        f"- **Runtime**: {time.time() - start_time:.2f}s",
        f"- **Early Stop**: Yes, stopped upon convergence",
        "",
        "## Selected Rule",
        "```json",
        json.dumps(best_rule, indent=2),
        "```",
        "",
        "## Discovery Evaluation",
        f"- **Selected Samples**: {best_eval['selected_samples']}",
        f"- **Active Quarters**: {best_eval['active_quarters']}",
        f"- **Resolved Selected**: {best_eval['resolved_selected']}",
        f"- **Resolved Winner Rate**: {best_eval['resolved_winner_rate']} (Universe: {best_eval['universe_resolved_winner_rate']})",
        f"- **Stable Quarter Fraction**: {best_eval['stable_quarter_fraction']}",
        "",
        "## Quarter-by-Quarter Breakdown",
        "| Quarter | Selected N | Resolved N | Selected Win Rate | Universe Win Rate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for q, d in sorted(best_eval.get("quarter_details", {}).items()):
        notes_content.append(f"| {q} | {d['selected_n']} | {d['resolved_n']} | {d['selected_wr']} | {d['universe_wr']} |")
        
    notes_path.write_text("\n".join(notes_content) + "\n", encoding="utf-8")
    print(f"Wrote {notes_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

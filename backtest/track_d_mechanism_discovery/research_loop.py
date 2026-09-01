from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    POLICY_REVIEW_CALLS,
    POLICY_SYNTHESIS_CALLS,
    FINAL_INTERPRETATION_CALLS,
    FEATURE_MANIFEST_PATH,
    MAX_FROZEN_AGENT_POLICIES,
    RESEARCH_ROLE_SEQUENCE,
)
from .llm_client import DeepSeekResearchClient
from .policy_dsl import DSLPolicy, deduplicate_policy_specs, validate_policy_spec
from .research_questions import ResearchQuestion, build_question_plan


ROLE_SYSTEMS={
    "researcher": (
        "You are the Researcher in a quantitative trading research team. Build a causal/mechanistic "
        "hypothesis from the supplied frozen historical evidence. Distinguish PIT-observable evidence "
        "from outcome labels. Seek specific predictions, not generic finance commentary. Return JSON only."
    ),
    "skeptic": (
        "You are the Skeptic. Attack the prior research claim as if rejecting a paper. Search for proxy "
        "variables, conditioning effects, sample artifacts, contradictory cases, and alternative mechanisms. "
        "A strong answer can conclude the original claim is probably wrong. Return JSON only."
    ),
    "experimental_designer": (
        "You are the Experimental Designer. Convert the debate into falsifiable, pre-specified tests using "
        "only the supplied PIT feature allowlist and fixed 3-slot accounting. Never request outer-forward "
        "outcomes. Prefer tests that separate competing mechanisms. Return JSON only."
    ),
    "synthesizer": (
        "You are the Synthesizer. Reconcile Researcher, Skeptic, and Experimental Designer. Produce a "
        "tentative verdict, confidence, falsifiers, and executable-policy implications. Do not claim a "
        "mechanism is proven when evidence is mixed. Return JSON only."
    ),
}


def _compact_json(value: Any, max_chars: int=18000) -> str:
    text=json.dumps(value,ensure_ascii=False,sort_keys=True,default=str)
    if len(text)<=max_chars:
        return text
    return text[:max_chars]+"...<truncated>"


def build_evidence_bundle(
    mechanism_df: pd.DataFrame,
    failure_summary: dict[str,Any],
    discovery_df: pd.DataFrame,
) -> dict[str,Any]:
    manifest=json.loads(FEATURE_MANIFEST_PATH.read_text(encoding="utf-8"))
    pit_allowed={
        name
        for name,meta in manifest["features"].items()
        if meta.get("allowed_for_discovery") is True
    }
    numeric_cols=[
        col for col in discovery_df.columns
        if col in pit_allowed and pd.api.types.is_numeric_dtype(discovery_df[col])
    ]
    distributions={}
    for col in numeric_cols[:40]:
        raw=discovery_df[col]
        # Pandas treats bool as numeric, but NumPy 2.x quantile interpolation
        # cannot subtract boolean values. Normalize every numeric PIT summary
        # series to float64 before quantile calculation.
        if pd.api.types.is_bool_dtype(raw):
            s=raw.astype("Float64").dropna().astype("float64")
        else:
            s=pd.to_numeric(raw,errors="coerce").astype("float64").dropna()
        if s.empty:
            continue
        distributions[col]={
            "p10":round(float(s.quantile(.10)),5),
            "p50":round(float(s.quantile(.50)),5),
            "p90":round(float(s.quantile(.90)),5),
        }
    mechanism_records=mechanism_df.replace({np.nan:None}).to_dict(orient="records")
    return {
        "mechanism_experiments":mechanism_records,
        "failure_archaeology":failure_summary,
        "pit_distributions":distributions,
        "guardrails":{
            "outer_forward_outcomes_visible":False,
            "common_selectable_universe":"b0_eligible == True",
            "capital_accounting":"sum(selected W4 returns) / 3; unused slots are 0 cash",
        },
    }


def _role_prompt(
    question: ResearchQuestion,
    role: str,
    evidence: dict[str,Any],
    prior: dict[str,Any],
) -> str:
    schemas={
        "researcher":{
            "mechanism_claim":"specific claim",
            "evidence_for":["specific observations"],
            "evidence_against":["known contradictions"],
            "falsifiable_predictions":["predictions"],
            "policy_implications":["possible implications"],
        },
        "skeptic":{
            "strongest_counterclaim":"specific alternative",
            "attacks":[{"issue":"...","why_it_matters":"..."}],
            "counterexamples_to_seek":["..."],
            "what_would_change_my_mind":["..."],
        },
        "experimental_designer":{
            "tests":[
                {
                    "name":"...",
                    "features":["PIT feature names"],
                    "comparison":"...",
                    "success_criterion":"...",
                    "failure_criterion":"...",
                }
            ],
            "leakage_checks":["..."],
            "priority_order":["test names"],
        },
        "synthesizer":{
            "verdict":"SUPPORTED|WEAK|CONTEXT_DEPENDENT|REFUTED|UNRESOLVED",
            "confidence_0_100":50,
            "mechanism_summary":"...",
            "falsifiers":["..."],
            "actionable_design_principles":["..."],
            "dsl_policy_ideas":[
                {
                    "idea":"...",
                    "why_distinct":"...",
                    "features":["..."],
                    "capacity_relevance":"...",
                }
            ],
        },
    }
    return f"""# Track D research cycle

Question ID: {question.question_id}
Direction: {question.direction}
Research question:
{question.question}

Frozen evidence available at this stage:
{_compact_json(evidence)}

Prior role outputs for this same question:
{_compact_json(prior)}

Rules:
- Do not use ticker identity as a predictive feature.
- Do not use calendar identity as a predictive feature.
- Do not ask for or infer outer-forward outcomes.
- Separate predictive mechanism from portfolio-construction effects.
- If evidence is insufficient, say UNRESOLVED rather than inventing certainty.
- Avoid repeating a prior role's argument; add a new layer of analysis.

Return exactly one JSON object matching this role schema:
{json.dumps(schemas[role],ensure_ascii=False,indent=2)}
"""


def run_research_cycles(
    client: DeepSeekResearchClient,
    evidence: dict[str,Any],
    ledger_path: Path,
) -> dict[str,Any]:
    plan=build_question_plan()
    existing={}
    if ledger_path.exists():
        old=json.loads(ledger_path.read_text(encoding="utf-8"))
        existing={x["question_id"]:x for x in old.get("cycles",[])}

    cycles=[]
    for q in plan:
        if q.question_id in existing and existing[q.question_id].get("complete") is True:
            cycles.append(existing[q.question_id])
            continue

        prior={}
        role_outputs={}
        for role in RESEARCH_ROLE_SEQUENCE:
            purpose=f"research__{q.question_id}__{role}"
            result=client.call_json(
                purpose,
                ROLE_SYSTEMS[role],
                _role_prompt(q,role,evidence,prior),
                temperature=.72 if role in {"researcher","skeptic"} else .55,
            )
            role_outputs[role]=result
            prior[role]=result

        cycle={
            "question_id":q.question_id,
            "direction":q.direction,
            "question":q.question,
            "fingerprint":q.fingerprint,
            "roles":role_outputs,
            "complete":True,
        }
        cycles.append(cycle)
        ledger_path.parent.mkdir(parents=True,exist_ok=True)
        ledger_path.write_text(json.dumps({
            "question_count":len(plan),
            "cycles":cycles,
            "request_budget":client.ledger.snapshot(),
        },indent=2,ensure_ascii=False),encoding="utf-8")

    result={
        "question_count":len(plan),
        "cycles":cycles,
        "request_budget":client.ledger.snapshot(),
    }
    ledger_path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    return result


DSL_SCHEMA_TEXT="""
Safe Track D DSL:
{
  "policy_id": "unique_snake_case",
  "description": "...",
  "research_origin": "question/direction",
  "base": "zero | b0_rank",
  "terms": [
    {"type":"linear","feature":"PIT numeric feature","transform":"identity|zscore|rank_pct|neg_abs","weight":-10..10},
    {"type":"threshold","logic":"all|any","conditions":[{"feature":"PIT feature","op":"gt|gte|lt|lte|eq|neq|in|is_true|is_false","value":"when required"}],"add":-10..10},
    {"type":"interaction","left":"numeric feature","right":"numeric feature","transform":"zscore|rank_pct|identity|neg_abs","weight":-10..10}
  ],
  "selector": {
    "industry_mode":"distinct_1|max_2_per_ind|unconstrained",
    "capacity":{
      "mode":"fixed|min_score|score_gap|top1_confidence",
      "max_positions":1|2|3,
      "min_score":"optional -20..20",
      "gap":"required for gap modes, .05..10"
    }
  }
}
All policies are automatically restricted to b0_eligible=True.
"""


def synthesize_policy_specs(
    client: DeepSeekResearchClient,
    research_ledger: dict[str,Any],
    evidence: dict[str,Any],
    output_path: Path,
) -> dict[str,Any]:
    synths=[
        {
            "question_id":c["question_id"],
            "direction":c["direction"],
            "question":c["question"],
            "synthesis":c["roles"]["synthesizer"],
        }
        for c in research_ledger["cycles"]
    ]
    raw_specs=[]
    rejected=[]
    for i in range(POLICY_SYNTHESIS_CALLS):
        batch=[synths[j] for j in range(i,len(synths),POLICY_SYNTHESIS_CALLS)][:6]
        prompt=f"""# Track D executable B1 synthesis batch {i+1}/{POLICY_SYNTHESIS_CALLS}

Convert the research conclusions below into 2-4 meaningfully distinct executable policies.
Do not create coefficient-only variants of the same idea. Prefer mechanisms that answer a
specific falsifiable research question. Adaptive capacity is encouraged when justified.

Research conclusions:
{_compact_json(batch,24000)}

Key frozen evidence:
{_compact_json(evidence,12000)}

{DSL_SCHEMA_TEXT}

Return:
{{"policies":[<DSL objects>],"batch_rationale":"why these are non-redundant"}}
"""
        response=client.call_json(
            f"policy_synthesis__{i+1:03d}",
            "You compile mechanism research into safe, auditable quantitative policy DSL. JSON only.",
            prompt,
            temperature=.65,
        )
        policies=response.get("policies",[]) if isinstance(response,dict) else []
        if not isinstance(policies,list):
            policies=[]
        for raw in policies:
            try:
                raw_specs.append(validate_policy_spec(raw))
            except Exception as exc:
                rejected.append({
                    "batch":i+1,
                    "policy_id":str(raw.get("policy_id","")) if isinstance(raw,dict) else "",
                    "reason":str(exc),
                })

    kept,dropped=deduplicate_policy_specs(raw_specs)
    kept=kept[:MAX_FROZEN_AGENT_POLICIES]

    # Conceptual adversarial review is audit-only; it cannot mutate specs after generation.
    reviews=[]
    if kept:
        batch_size=max(1,int(np.ceil(len(kept)/float(POLICY_REVIEW_CALLS))))
        for i in range(POLICY_REVIEW_CALLS):
            batch=kept[i*batch_size:(i+1)*batch_size]
            if not batch:
                break
            prompt=f"""Adversarially review these already-normalized Track D policy specs BEFORE outcome evaluation.
Identify conceptual duplication, proxy risk, implausible mechanisms, and capacity rules likely to
manufacture abstention. You may criticize but may not rewrite the frozen specs.

Policies:
{_compact_json(batch,24000)}

Return {{"reviews":[{{"policy_id":"...","concerns":["..."],"novelty_0_100":50,"mechanism_clarity_0_100":50}}]}}
"""
            reviews.append(client.call_json(
                f"policy_review__{i+1:03d}",
                "You are an adversarial pre-outcome policy reviewer. JSON only.",
                prompt,
                temperature=.4,
            ))

    result={
        "policies":kept,
        "schema_rejected":rejected,
        "exact_spec_duplicates":dropped,
        "adversarial_reviews":reviews,
        "request_budget":client.ledger.snapshot(),
    }
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    return result


def behaviorally_deduplicate_specs(
    specs:list[dict[str,Any]],
    discovery_df:pd.DataFrame,
    discovery_snapshots:list[str],
    *,
    similarity_threshold:float=.98,
) -> tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    signatures=[]
    kept=[]
    dropped=[]

    for spec in specs:
        policy=DSLPolicy(spec)
        picks_by_snap=[]
        for snap in discovery_snapshots:
            s_df=discovery_df[discovery_df["snapshot_date"].astype(str)==str(snap)].copy()
            scored=policy.score_candidates(s_df)
            quotas=policy.allocate_industries(scored)
            picks=policy.pick_stocks(scored,quotas)
            picks_by_snap.append(tuple(picks))

        duplicate_of=None
        duplicate_similarity=0.0
        for prev_spec,prev_sig in signatures:
            sims=[]
            for a,b in zip(picks_by_snap,prev_sig):
                sa,sb=set(a),set(b)
                union=sa|sb
                sims.append(1.0 if not union else len(sa&sb)/float(len(union)))
            sim=float(np.mean(sims)) if sims else 1.0
            if sim>=similarity_threshold:
                duplicate_of=prev_spec["policy_id"]
                duplicate_similarity=sim
                break

        if duplicate_of:
            dropped.append({
                "policy_id":spec["policy_id"],
                "duplicate_of":duplicate_of,
                "mean_jaccard":round(duplicate_similarity,4),
            })
        else:
            kept.append(spec)
            signatures.append((spec,picks_by_snap))
    return kept,dropped


def final_interpretation_calls(
    client:DeepSeekResearchClient,
    result_summary:dict[str,Any],
) -> list[dict[str,Any]]:
    roles=[
        "mechanism scientist",
        "portfolio construction specialist",
        "failure-mode analyst",
        "statistical skeptic",
        "production simplification reviewer",
        "research chair",
    ]
    outputs=[]
    for i,role in enumerate(roles[:FINAL_INTERPRETATION_CALLS],1):
        prompt=f"""You are the Track D final {role}. Policies are already frozen and evaluated; you
cannot propose a new policy or change thresholds. Interpret the locked results and identify the
most defensible mechanism-level conclusion, strongest contradictory evidence, and one unresolved
question suitable for future forward shadow.

Locked result summary:
{_compact_json(result_summary,26000)}

Return {{"conclusion":"...","strongest_evidence":["..."],"contradictions":["..."],"unresolved":["..."]}}.
"""
        result=client.call_json(
            f"final_interpretation__{i:02d}",
            "You interpret locked quantitative research without changing the pre-registered decision rule. JSON only.",
            prompt,
            temperature=.35,
        )
        if not isinstance(result,dict):
            result={
                "conclusion":str(result),
                "strongest_evidence":[],
                "contradictions":[],
                "unresolved":[],
            }
        outputs.append(result)
    return outputs

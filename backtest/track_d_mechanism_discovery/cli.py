from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from .config import (
    CACHE_MIGRATION_ALLOWED_FROM,
    HISTORICAL_END,
    OUT,
    PANEL_SOURCE,
    REQUEST_HARD_LIMIT,
)
from .failure_archaeology import build_failure_archaeology
from .integrity import git_sha, hash_file, verify_phase0, write_or_verify_phase0
from .llm_client import DeepSeekResearchClient
from .lock_manager import seal_final_lock, seal_policy_freeze, verify_policy_freeze
from .mechanism_lab import run_mechanism_lab
from .report import write_final_report
from .research_loop import (
    behaviorally_deduplicate_specs,
    build_evidence_bundle,
    final_interpretation_calls,
    run_research_cycles,
    synthesize_policy_specs,
)
from .walk_forward import build_locked_forward_split, evaluate_locked_forward


PHASE0=OUT/"phase0_manifest.json"
SPLIT=OUT/"locked_split.json"
MECH=OUT/"mechanism_results.parquet"
MECH_CSV=OUT/"mechanism_results.csv"
FAIL_CASES=OUT/"failure_archaeology.parquet"
FAIL_SUMMARY=OUT/"failure_summary.json"
REQUEST_LEDGER=OUT/"request_budget_ledger.json"
RESEARCH_LEDGER=OUT/"research_ledger.json"
SYNTHESIS=OUT/"policy_synthesis.json"
POLICY_FREEZE=OUT/"policy_freeze_manifest.json"
FORWARD=OUT/"locked_forward_results.parquet"
FORWARD_CSV=OUT/"locked_forward_results.csv"
DECISION=OUT/"track_d_decision.json"
COMPONENTS=OUT/"component_verdicts.csv"
INTERPRET=OUT/"final_interpretations.json"
REPORT=OUT/"TRACK_D_FINAL_REPORT.md"
FINAL_LOCK=OUT/"final_lock_manifest.json"


def _panel()->pd.DataFrame:
    df=pd.read_parquet(PANEL_SOURCE)
    df=df[df["snapshot_date"].astype(str)<=HISTORICAL_END].copy()
    return df


def _split()->dict:
    if not SPLIT.exists():
        raise RuntimeError("Track D locked split manifest missing")
    phase0=json.loads(PHASE0.read_text(encoding="utf-8"))
    expected=phase0.get("locked_split_hash")
    actual=hash_file(SPLIT)
    if expected and expected!=actual:
        raise RuntimeError("Track D locked split hash mismatch")
    return json.loads(SPLIT.read_text(encoding="utf-8"))


def _reset_for_source_change(old_source_sha:str,new_source_sha:str)->None:
    """One-time focused-protocol migration that preserves already-paid research cache.

    Only the explicitly allowlisted source revision can migrate. All deterministic
    artifacts are rebuilt under the new source hash; raw successful responses,
    request-attempt accounting, and completed research cycles are retained.
    """
    if old_source_sha not in set(CACHE_MIGRATION_ALLOWED_FROM):
        shutil.rmtree(OUT)
        return

    preserve={"rdagent_raw","request_budget_ledger.json","research_ledger.json"}
    for child in list(OUT.iterdir()):
        if child.name in preserve:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    completed_cycles=0
    if RESEARCH_LEDGER.exists():
        try:
            old_research=json.loads(RESEARCH_LEDGER.read_text(encoding="utf-8"))
            completed_cycles=sum(
                1 for item in old_research.get("cycles",[])
                if item.get("complete") is True
            )
        except Exception:
            completed_cycles=0

    migration={
        "from_source_git_sha":old_source_sha,
        "to_source_git_sha":new_source_sha,
        "preserved_completed_cycles":completed_cycles,
        "preserved_raw_meta_files":len(list((OUT/"rdagent_raw").glob("*.meta.json"))) if (OUT/"rdagent_raw").exists() else 0,
        "request_budget_ledger_preserved":REQUEST_LEDGER.exists(),
        "research_ledger_preserved":RESEARCH_LEDGER.exists(),
        "reason":"focused_track_d_protocol_reuses_only_matching_question_fingerprints",
    }
    (OUT/"cache_migration.json").write_text(
        json.dumps(migration,indent=2),
        encoding="utf-8",
    )


def cmd_prepare(_:argparse.Namespace)->None:
    current_sha=git_sha()
    if PHASE0.exists():
        old=json.loads(PHASE0.read_text(encoding="utf-8"))
        old_sha=str(old.get("source_git_sha") or "")
        if old_sha != current_sha:
            _reset_for_source_change(old_sha,current_sha)
    OUT.mkdir(parents=True,exist_ok=True)
    df=_panel()
    snaps=sorted(df["snapshot_date"].astype(str).unique().tolist())
    manifest=write_or_verify_phase0(PHASE0,snaps)
    split=build_locked_forward_split(snaps)
    SPLIT.write_text(json.dumps(split,indent=2),encoding="utf-8")
    manifest["locked_split_hash"]=hash_file(SPLIT)
    PHASE0.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"Track D Phase0 sealed: {manifest['run_id']}; historical snapshots={len(split['all_used_snapshots'])}")


def cmd_mechanisms(_:argparse.Namespace)->None:
    verify_phase0(PHASE0)
    df=_panel()
    split=_split()
    discovery=split["discovery_train"]
    mech=run_mechanism_lab(df,discovery)
    mech.to_parquet(MECH,index=False)
    mech.to_csv(MECH_CSV,index=False)
    cases,summary=build_failure_archaeology(df,discovery)
    cases.to_parquet(FAIL_CASES,index=False)
    FAIL_SUMMARY.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"Mechanism lab={len(mech)} experiments; archaeology={summary.get('case_count',0)} labeled cases")


def _evidence():
    df=_panel()
    split=_split()
    discovery=df[df["snapshot_date"].astype(str).isin(split["discovery_train"])].copy()
    mech=pd.read_parquet(MECH)
    failure=json.loads(FAIL_SUMMARY.read_text(encoding="utf-8"))
    return build_evidence_bundle(mech,failure,discovery),discovery,split


def _client()->DeepSeekResearchClient:
    return DeepSeekResearchClient(REQUEST_LEDGER,REQUEST_HARD_LIMIT)


def cmd_research(_:argparse.Namespace)->None:
    verify_phase0(PHASE0)
    evidence,_,_= _evidence()
    ledger=run_research_cycles(_client(),evidence,RESEARCH_LEDGER)
    print(f"RD-Agent research complete: {ledger['question_count']} questions; budget={ledger['request_budget']}")


def cmd_synthesize(_:argparse.Namespace)->None:
    verify_phase0(PHASE0)
    if POLICY_FREEZE.exists():
        verify_policy_freeze(
            PHASE0,POLICY_FREEZE,RESEARCH_LEDGER,SYNTHESIS,REQUEST_LEDGER,MECH,FAIL_SUMMARY,SPLIT
        )
        freeze=json.loads(POLICY_FREEZE.read_text(encoding="utf-8"))
        print(f"Policy freeze already valid; reusing {freeze['policy_count']} frozen DSL policies")
        return
    evidence,discovery,split=_evidence()
    research=json.loads(RESEARCH_LEDGER.read_text(encoding="utf-8"))
    result=synthesize_policy_specs(_client(),research,evidence,SYNTHESIS)
    kept,behavioral_drops=behaviorally_deduplicate_specs(
        result["policies"],discovery,split["discovery_train"]
    )
    if not kept:
        raise RuntimeError("Track D produced zero behaviorally distinct executable policies")
    result["pre_behavioral_count"]=len(result["policies"])
    result["behavioral_duplicates"]=behavioral_drops
    result["policies"]=kept
    SYNTHESIS.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    seal_policy_freeze(
        PHASE0,kept,RESEARCH_LEDGER,SYNTHESIS,REQUEST_LEDGER,MECH,FAIL_SUMMARY,SPLIT,POLICY_FREEZE
    )
    print(f"Policy freeze sealed: {len(kept)} behaviorally distinct DSL policies")


def cmd_walk_forward(_:argparse.Namespace)->None:
    verify_policy_freeze(
        PHASE0,POLICY_FREEZE,RESEARCH_LEDGER,SYNTHESIS,REQUEST_LEDGER,MECH,FAIL_SUMMARY,SPLIT
    )
    freeze=json.loads(POLICY_FREEZE.read_text(encoding="utf-8"))
    df=_panel()
    result,decision=evaluate_locked_forward(df,_split(),freeze["policies"])
    result.to_parquet(FORWARD,index=False)
    result.to_csv(FORWARD_CSV,index=False)
    DECISION.write_text(json.dumps(decision,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    pd.DataFrame(decision.get("component_verdicts",[])).to_csv(COMPONENTS,index=False)
    print(f"Locked forward complete: policies={len(result)}; decision={decision['state']}")


def cmd_conclude(_:argparse.Namespace)->None:
    verify_policy_freeze(
        PHASE0,POLICY_FREEZE,RESEARCH_LEDGER,SYNTHESIS,REQUEST_LEDGER,MECH,FAIL_SUMMARY,SPLIT
    )
    decision=json.loads(DECISION.read_text(encoding="utf-8"))
    forward=pd.read_parquet(FORWARD)
    ranked=forward.copy()
    ranked["_confirmation_mean"]=pd.to_numeric(
        ranked.get("confirmation_mean_spread", pd.Series(index=ranked.index, dtype=float)),
        errors="coerce",
    )
    ranked["_screen_mean"]=pd.to_numeric(
        ranked.get("screen_mean_spread", pd.Series(index=ranked.index, dtype=float)),
        errors="coerce",
    )
    ranked=ranked.sort_values(
        ["confirmation_evaluated","b1_gate_pass","_confirmation_mean","_screen_mean"],
        ascending=[False,False,False,False],
    )
    summary={
        "decision":decision,
        "top_forward_policies":ranked.head(20).to_dict(orient="records"),
    }
    client=_client()
    interpretations=final_interpretation_calls(client,summary)
    INTERPRET.write_text(json.dumps(interpretations,indent=2,ensure_ascii=False),encoding="utf-8")

    failure=json.loads(FAIL_SUMMARY.read_text(encoding="utf-8"))
    freeze=json.loads(POLICY_FREEZE.read_text(encoding="utf-8"))
    write_final_report(
        REPORT,decision,pd.read_parquet(MECH),failure,forward,_split(),
        client.ledger.snapshot(),interpretations,len(freeze["policies"])
    )

    rules=[
        x for x in decision.get("component_verdicts",[])
        if x.get("verdict") in {"HARMFUL","REDUNDANT"}
    ]
    (OUT/"RULES_TO_REMOVE_OR_SIMPLIFY.md").write_text(
        "# Rules to remove or simplify\n\n"+
        "\n".join(f"- {x['component']}: {x['verdict']}" for x in rules)+
        ("\n" if rules else "No rule met the deterministic remove/simplify criterion.\n"),
        encoding="utf-8",
    )
    if decision.get("winner_spec"):
        (OUT/"B1_POLICY_SPEC.json").write_text(
            json.dumps(decision["winner_spec"],indent=2,ensure_ascii=False),encoding="utf-8"
        )
    elif decision.get("winner"):
        name="MINIMAL_B0_SPEC.json" if decision["state"].startswith("STATE_C") else "B0_REPAIR_SPEC.json"
        (OUT/name).write_text(json.dumps(decision["winner"],indent=2,ensure_ascii=False,default=str),encoding="utf-8")

    seal_final_lock(PHASE0,POLICY_FREEZE,{
        "policy_freeze":POLICY_FREEZE,
        "forward_results":FORWARD,
        "decision":DECISION,
        "component_verdicts":COMPONENTS,
        "final_interpretations":INTERPRET,
        "final_report":REPORT,
    },FINAL_LOCK)
    print(f"Track D finalized: {decision['state']}; request budget={client.ledger.snapshot()}")


def cmd_materialize(args:argparse.Namespace)->None:
    print("=== Track D sealed deep-research materialization ===")
    cmd_prepare(args)
    cmd_mechanisms(args)
    cmd_research(args)
    cmd_synthesize(args)
    cmd_walk_forward(args)
    cmd_conclude(args)


def main()->None:
    parser=argparse.ArgumentParser(description="Track D B0 mechanism discovery and B1 synthesis")
    sub=parser.add_subparsers(dest="command",required=True)
    for name in ["prepare","mechanisms","research","synthesize","walk-forward","conclude","materialize"]:
        sub.add_parser(name)
    args=parser.parse_args()
    dispatch={
        "prepare":cmd_prepare,
        "mechanisms":cmd_mechanisms,
        "research":cmd_research,
        "synthesize":cmd_synthesize,
        "walk-forward":cmd_walk_forward,
        "conclude":cmd_conclude,
        "materialize":cmd_materialize,
    }
    dispatch[args.command](args)


if __name__=="__main__":
    main()

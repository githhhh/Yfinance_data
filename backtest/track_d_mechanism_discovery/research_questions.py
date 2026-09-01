from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .config import B0_COMPONENTS, DIRECTION_QUESTION_COUNTS


@dataclass(frozen=True)
class ResearchQuestion:
    question_id: str
    direction: str
    question: str
    fingerprint: str


def _q(direction: str, idx: int, text: str) -> ResearchQuestion:
    fp = hashlib.sha256((direction + "::" + text.strip().lower()).encode("utf-8")).hexdigest()
    return ResearchQuestion(
        question_id=f"{direction}__q{idx:03d}",
        direction=direction,
        question=text.strip(),
        fingerprint=fp,
    )


def _mechanism_questions() -> list[str]:
    angles = [
        "predictive information independent of neighboring B0 terms",
        "downside-risk control versus return ranking",
        "proxy/redundancy with other B0 components",
    ]
    out=[]
    # Angle-major ordering ensures a truncated deep plan still covers every component.
    for angle in angles:
        for comp in B0_COMPONENTS:
            out.append(
                f"For B0 component '{comp}', determine whether its historical value comes from {angle}. "
                "State what evidence would falsify the component's claimed role and what a cleaner replacement would look like."
            )
    return out


def _failure_questions() -> list[str]:
    error_types=["false-positive B0 picks that later ruin","false-negative non-picks that become large winners"]
    contexts=["pullback versus breakout setup","near-buy-point geometry","volume confirmation/dry-up","industry concentration"]
    angles=[
        "identify repeated PIT-visible precursors",
        "separate true signal from correlated proxy",
        "propose a counterexample that would invalidate the pattern",
    ]
    return [
        f"Study {err} under {ctx}; {angle}. Do not explain outcomes using information unavailable at snapshot time."
        for err in error_types for ctx in contexts for angle in angles
    ]


def _capacity_questions() -> list[str]:
    contexts=[
        "large Top1-vs-Top2 quality gap",
        "weak third candidate",
        "single-industry crowding",
        "low setup density",
        "high dispersion of candidate quality",
        "pullback-heavy candidate sets",
    ]
    angles=[
        "when cash should dominate filling another slot",
        "what PIT statistic can measure confidence without future leakage",
        "how to falsify an adaptive 0/1/2/3 capacity rule",
    ]
    return [
        f"For {ctx}, determine {angle}. Compare the economic meaning of selecting 1, 2, or 3 positions under fixed 3-slot capital."
        for ctx in contexts for angle in angles
    ]


def _lane_questions() -> list[str]:
    hypotheses=[
        "Lane has independent predictive information",
        "Lane is only a proxy for freshness/geometry/volume",
        "Lane mainly changes portfolio risk rather than expected return",
    ]
    regimes=[
        "fresh breakout weeks",
        "constructive pullback weeks",
        "mixed breakout/pullback weeks",
        "high-volume confirmation weeks",
        "weak-volume weeks",
        "industry-crowded weeks",
    ]
    return [
        f"Test the hypothesis '{h}' specifically in {r}. Give discriminating predictions for B0_LANE, LANE_NEUTRAL, and SCORE_BEFORE_LANE."
        for h in hypotheses for r in regimes
    ]


def _nonlinear_questions() -> list[str]:
    pairs=[
        ("pullback_v_is_dry","current_vs_ibd_candidate_pct"),
        ("pullback_v_is_dry","ibd_entry_volume_ratio"),
        ("base_depth_pct","mom_20"),
        ("ibd_entry_close_position","ibd_entry_volume_ratio"),
        ("dist_to_52w_high_pct","mom_60"),
        ("eps_yoy_growth","mom_20"),
        ("rv_20","mom_20"),
        ("atr_14_pct","current_vs_ibd_candidate_pct"),
        ("volume_ratio","rel_spy_20"),
        ("base_duration_weeks","pullback_pct"),
    ]
    structures=["piecewise threshold","interaction term","conditional ranking plus adaptive capacity"]
    return [
        f"Explore a {structure} using {a} and {b}. Explain the mechanism, likely failure regime, and how to encode it without expanding the B0 eligible universe."
        for a,b in pairs for structure in structures
    ]


def _adversarial_questions() -> list[str]:
    targets=[
        "the apparent industry-allocation advantage",
        "the apparent within-industry stock-selection advantage",
        "the near-parity of SCORE_BEFORE_LANE",
        "the usefulness of dry pullback semantics",
        "the benefit of distinct-1 diversification",
        "the proposed adaptive-capacity direction",
    ]
    attacks=[
        "construct the strongest alternative explanation",
        "identify a hidden conditioning or selection effect",
        "specify a falsification test that could reverse the current interpretation",
    ]
    return [
        f"Adversarially attack {target}: {attack}. Prefer a concrete diagnostic over generic caution."
        for target in targets for attack in attacks
    ]


_GENERATORS={
    "mechanism_falsification":_mechanism_questions,
    "failure_archaeology":_failure_questions,
    "capacity_abstention":_capacity_questions,
    "lane_mechanism":_lane_questions,
    "nonlinear_b1":_nonlinear_questions,
    "adversarial_review":_adversarial_questions,
}


def build_question_plan() -> list[ResearchQuestion]:
    plan=[]
    seen=set()
    for direction,count in DIRECTION_QUESTION_COUNTS.items():
        pool=_GENERATORS[direction]()
        if len(pool)<count:
            raise RuntimeError(f"Question generator for {direction} only produced {len(pool)} < {count}")
        for idx,text in enumerate(pool[:count],1):
            q=_q(direction,idx,text)
            if q.fingerprint in seen:
                raise RuntimeError(f"Duplicate Track D research question fingerprint: {q.question_id}")
            seen.add(q.fingerprint)
            plan.append(q)
    return plan

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from ..protocol import PolicyProposal, ChallengerProtocol


def compute_snapshot_picks_jaccard(
    picks_map_a: dict[str, list[str]],
    picks_map_b: dict[str, list[str]],
) -> float:
    """Compute average Top3 Jaccard similarity across all snapshot dates."""
    all_snaps = set(picks_map_a.keys()) | set(picks_map_b.keys())
    if not all_snaps:
        return 1.0

    jaccards = []
    for s in all_snaps:
        set_a = set(picks_map_a.get(s, []))
        set_b = set(picks_map_b.get(s, []))
        union = set_a | set_b
        if not union:
            jaccards.append(1.0)
        else:
            jaccards.append(len(set_a & set_b) / float(len(union)))

    return float(np.mean(jaccards))


def deduplicate_proposals_behaviorally(
    challengers: list[ChallengerProtocol],
    train_df: pd.DataFrame,
    jaccard_threshold: float = 0.99,
    rank_corr_threshold: float = 0.995,
) -> tuple[list[ChallengerProtocol], list[dict[str, Any]]]:
    """Outcome-blind behavioral deduplication using Train features only."""
    snaps = sorted(train_df["snapshot_date"].astype(str).unique().tolist())

    # Precompute scored dfs and selected picks for each challenger across all snapshots
    picks_cache: dict[str, dict[str, list[str]]] = {}
    ranks_cache: dict[str, dict[str, dict[str, float]]] = {}

    for ch in challengers:
        ch_picks: dict[str, list[str]] = {}
        ch_ranks: dict[str, dict[str, float]] = {}
        for s in snaps:
            s_df = train_df[train_df.snapshot_date.astype(str) == str(s)].copy()
            if s_df.empty:
                ch_picks[s] = []
                ch_ranks[s] = {}
                continue
            scored = ch.score_candidates(s_df)
            quotas = ch.allocate_industries(scored)
            picks = ch.pick_stocks(scored, quotas)
            ch_picks[s] = picks

            # Cache ranks for correlation
            if not scored.empty and "raw_rank" in scored.columns:
                ch_ranks[s] = dict(zip(scored["code"].astype(str), scored["raw_rank"]))
            else:
                ch_ranks[s] = {}

        picks_cache[ch.policy_id] = ch_picks
        ranks_cache[ch.policy_id] = ch_ranks

    # Greedily keep distinct proposals
    kept: list[ChallengerProtocol] = []
    dropped: list[dict[str, Any]] = []

    for ch in challengers:
        is_dup = False
        ch_p = picks_cache[ch.policy_id]

        for k in kept:
            k_p = picks_cache[k.policy_id]
            jaccard = compute_snapshot_picks_jaccard(ch_p, k_p)

            if jaccard >= jaccard_threshold:
                # Compute rank correlation across all snapshots with multiple candidates
                corrs = []
                for s in snaps:
                    r_ch = ranks_cache[ch.policy_id].get(s, {})
                    r_k = ranks_cache[k.policy_id].get(s, {})
                    common_codes = sorted(set(r_ch.keys()) & set(r_k.keys()))
                    if len(common_codes) >= 3:
                        vals_ch = [r_ch[c] for c in common_codes]
                        vals_k = [r_k[c] for c in common_codes]
                        corr, _ = spearmanr(vals_ch, vals_k)
                        if not np.isnan(corr):
                            corrs.append(corr)

                avg_corr = float(np.mean(corrs)) if corrs else 1.0
                if avg_corr >= rank_corr_threshold:
                    is_dup = True
                    dropped.append({
                        "dropped_policy_id": ch.policy_id,
                        "duplicate_of": k.policy_id,
                        "jaccard_similarity": round(jaccard, 4),
                        "rank_correlation": round(avg_corr, 4),
                    })
                    break

        if not is_dup:
            kept.append(ch)

    return kept, dropped

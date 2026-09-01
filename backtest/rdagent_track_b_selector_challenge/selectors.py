from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Sequence
import numpy as np
import pandas as pd
from .config import TOP_N


@dataclass(frozen=True)
class SelectorConfig:
    selector_id: str
    family: str  # 'pure_rank', 'distinct_industry', 'portfolio_aware'
    industry_constraint: str  # 'none', 'hard_distinct', 'soft_penalty'
    portfolio_objective: str
    lambda_vol: float = 0.0
    lambda_overheat: float = 0.0
    lambda_industry_dup: float = 0.0
    candidate_pool_size: int = 10
    complexity: str = 'low'


def select_pure_rank(df: pd.DataFrame, score_col: str = 'score', top_n: int = TOP_N) -> pd.DataFrame:
    """Selector 1: Pure rank descending, no industry constraint."""
    picks = []
    for snap, g in df.groupby('snapshot_date', sort=True):
        sorted_g = g.sort_values([score_col, 'code'], ascending=[False, True])
        top_k = sorted_g.head(top_n)
        for i, (_, r) in enumerate(top_k.iterrows(), 1):
            d = r.to_dict()
            d['model_pick_order'] = i
            picks.append(d)
    return pd.DataFrame(picks) if picks else pd.DataFrame(columns=df.columns)


def select_distinct_industry(df: pd.DataFrame, score_col: str = 'score', top_n: int = TOP_N) -> pd.DataFrame:
    """Selector 2: Score descending, maximum 1 stock per distinct industry."""
    picks = []
    for snap, g in df.groupby('snapshot_date', sort=True):
        sorted_g = g.sort_values([score_col, 'code'], ascending=[False, True])
        used_industries = set()
        chosen = []
        for _, r in sorted_g.iterrows():
            ind = str(r.get('industry', '') or '').strip().lower()
            if ind and ind in used_industries:
                continue
            chosen.append(r)
            if ind:
                used_industries.add(ind)
            if len(chosen) == top_n:
                break
        for i, r in enumerate(chosen, 1):
            d = r.to_dict()
            d['model_pick_order'] = i
            picks.append(d)
    return pd.DataFrame(picks) if picks else pd.DataFrame(columns=df.columns)


def select_portfolio_aware(
    df: pd.DataFrame,
    score_col: str = 'score',
    top_n: int = TOP_N,
    lambda_vol: float = 0.5,
    lambda_overheat: float = 0.3,
    lambda_industry_dup: float = 2.0,
    candidate_pool_size: int = 8,
) -> pd.DataFrame:
    """Selector 3: Portfolio-aware selection from top-K scoring candidates.
    Objective:
        argmax_{S, |S|=top_n} sum(score_i) 
          - lambda_vol * mean(rv_20_i)
          - lambda_overheat * mean(overheat_penalty_i)
          - lambda_industry_dup * (duplicate industry count in S)
    """
    picks = []
    for snap, g in df.groupby('snapshot_date', sort=True):
        sorted_g = g.sort_values([score_col, 'code'], ascending=[False, True])
        pool = sorted_g.head(candidate_pool_size).copy()
        
        if len(pool) <= top_n:
            chosen = pool
        else:
            best_obj = -float('inf')
            best_subset = None
            
            # Precompute normalized features for pool candidates
            scores = pd.to_numeric(pool[score_col], errors='coerce').fillna(0.0).values
            rv = pd.to_numeric(pool.get('rv_20', 0.0), errors='coerce').fillna(0.0).values
            # normalize rv within pool to 0-1 scale if std > 0
            if np.ptp(rv) > 1e-6:
                rv_norm = (rv - np.min(rv)) / (np.ptp(rv) + 1e-6)
            else:
                rv_norm = np.zeros_like(rv)
                
            # Overheat proxy: high extension from buy point (>5%) + high ATR
            ext = pd.to_numeric(pool.get('current_vs_ibd_candidate_pct', 0.0), errors='coerce').fillna(0.0).values
            overheat = np.maximum(0.0, ext - 5.0) / 10.0
            
            industries = [str(x or '').strip().lower() for x in pool.get('industry', [''] * len(pool))]
            
            indices = list(range(len(pool)))
            for comb in combinations(indices, top_n):
                sub_score = np.sum(scores[list(comb)])
                sub_vol = np.mean(rv_norm[list(comb)])
                sub_overheat = np.mean(overheat[list(comb)])
                
                # Count duplicate industry occurrences
                inds = [industries[i] for i in comb if industries[i]]
                dup_count = len(inds) - len(set(inds))
                
                obj = sub_score - (lambda_vol * sub_vol) - (lambda_overheat * sub_overheat) - (lambda_industry_dup * dup_count)
                if obj > best_obj:
                    best_obj = obj
                    best_subset = comb
            
            chosen = pool.iloc[list(best_subset)].sort_values([score_col, 'code'], ascending=[False, True])
            
        for i, (_, r) in enumerate(chosen.iterrows(), 1):
            d = r.to_dict()
            d['model_pick_order'] = i
            picks.append(d)
            
    return pd.DataFrame(picks) if picks else pd.DataFrame(columns=df.columns)


def apply_selector(df: pd.DataFrame, cfg: SelectorConfig, score_col: str = 'score') -> pd.DataFrame:
    """Route dataframe to the configured selector function."""
    if cfg.family == 'pure_rank':
        return select_pure_rank(df, score_col=score_col, top_n=TOP_N)
    elif cfg.family == 'distinct_industry':
        return select_distinct_industry(df, score_col=score_col, top_n=TOP_N)
    elif cfg.family == 'portfolio_aware':
        return select_portfolio_aware(
            df,
            score_col=score_col,
            top_n=TOP_N,
            lambda_vol=cfg.lambda_vol,
            lambda_overheat=cfg.lambda_overheat,
            lambda_industry_dup=cfg.lambda_industry_dup,
            candidate_pool_size=cfg.candidate_pool_size,
        )
    else:
        raise ValueError(f"Unknown selector family: {cfg.family}")

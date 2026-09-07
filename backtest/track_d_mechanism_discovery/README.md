# Track D — B0 Mechanism Discovery & B1 Synthesis

Track D is a fail-closed research program. It does not modify Production B0.

## Research objective

The required result is not "B0 was not beaten". Track D must produce:

1. a component-level B0 mechanism map;
2. B0 false-positive / false-negative failure modes;
3. adaptive-capacity findings;
4. a frozen set of executable RD-Agent DSL policies;
5. locked multi-block forward results; and
6. exactly one deterministic exit state:
   - STATE_A_PROMOTE_B1_TO_FORWARD_SHADOW
   - STATE_B_REPAIR_B0
   - STATE_C_COMPRESS_B0
   - STATE_D_MECHANISM_MAP_WITH_B0_RETAINED

State D is still required to contain rule-level ESSENTIAL / HELPFUL / REDUNDANT / HARMFUL / UNCERTAIN
verdicts. It is not permission to claim B0 is unbeatable.

## DeepSeek request budget

The focused deep profile pre-registers 78 unique questions x 4 roles = 312 research calls, plus:

- 13 executable-policy synthesis calls;
- up to 6 pre-outcome adversarial policy reviews;
- 4 post-evaluation interpretation calls.

Planned maximum before retries: 335 calls. The 20 mechanism-falsification and 22 failure-archaeology
questions remain fully covered; later directions are deliberately pruned to representative, decision-changing questions.

The hard Track D limit is 650 provider attempts including retries. The protocol was authored when the
daily account meter was 202/1000, leaving substantial reserve. Do not raise the hard limit during a run.

Successful calls are cached by purpose ID and prompt hash. Re-running on the same source commit resumes
without re-spending those requests. The one-time focused-protocol migration from source 00770fb... preserves
the already-paid request ledger, raw responses, and completed cycles; every reused completed question must
match its frozen fingerprint. Other source changes still invalidate Track D output/cache.

## Local execution contract for Gemini

Do not edit Track D, Track C, Production B0, or tests during materialization. Any failure is returned to the
code owner.

Run:

    git checkout codex/clean-latest-quant-trade-replay-pools
    git pull --ff-only

    /Users/dev/.conda/envs/quant_env/bin/python -m pytest -q tests/test_track_d_mechanism_discovery.py

    /Users/dev/.conda/envs/quant_env/bin/python -m backtest.track_d_mechanism_discovery.cli materialize

    /Users/dev/.conda/envs/quant_env/bin/python -m pytest -q       tests/test_track_d_mechanism_discovery.py       tests/test_track_c_ranking_discovery.py       tests/test_rdagent_track_b_selector_challenge.py       tests/test_b0_multifactor_challenge.py

After success, commit only:

    git add backtest/track_d_mechanism_discovery/output/
    git commit -m "docs(track_d): materialize mechanism discovery and B1 synthesis artifacts"
    git push origin codex/clean-latest-quant-trade-replay-pools

If materialize fails, do not patch source locally. Preserve the error and return it.

## Safety / leakage constraints

- Historical research window is frozen at 2026-07-24, the last fully W4-mature snapshot in the current panel.
- First 18 snapshots are discovery-train.
- Next 4 snapshots are purged.
- Six non-overlapping 3-snapshot blocks (18 mature weeks total) are locked forward evaluation. Blocks 1-2 are screening-only; blocks 3-6 are untouched confirmation. Screening may advance at most 6 Agent B1 policies plus one Minimal-B0 candidate for each removal count (1..4).
- All outer-forward outcomes are unavailable to RD-Agent before policy freeze. Screening outcomes may shortlist frozen policies, but confirmation outcomes cannot influence shortlist membership.
- Every DSL policy is forced onto b0_eligible=True.
- The DSL cannot access return or stop outcome columns.
- Arbitrary Python from the model is never executed.
- Policy promotion is mechanical; final LLM reviewers cannot change a frozen policy or decision threshold.

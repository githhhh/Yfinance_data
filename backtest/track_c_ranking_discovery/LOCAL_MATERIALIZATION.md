# Track C Local Materialization Runbook

This file is the handoff contract for Gemini/local execution after the research source commit is pushed.

## Non-negotiable rule

**Do not edit any Track C research source while materializing artifacts.**

If any command fails, stop and report the error/log. Do not patch `.py` or protocol `.json` files locally.
The Phase 0 / proposal-freeze / Phase 4 hash gates are intentionally designed to reject such edits.

The root `.env` is local-only and must never be committed. Track C uses the RD-Agent/LiteLLM chat configuration from that file, including `CHAT_MODEL` / `TRACK_C_RDAGENT_MODEL` and the configured DeepSeek credentials.

## Required local sequence

```bash
git checkout codex/clean-latest-quant-trade-replay-pools
git pull --ff-only

/Users/dev/.conda/envs/quant_env/bin/python -m pytest -q tests/test_track_c_ranking_discovery.py

/Users/dev/.conda/envs/quant_env/bin/python -m backtest.track_c_ranking_discovery.cli materialize

/Users/dev/.conda/envs/quant_env/bin/python -m pytest -q   tests/test_track_c_ranking_discovery.py   tests/test_rdagent_track_b_selector_challenge.py   tests/test_b0_multifactor_challenge.py
```

The `materialize` command executes, fail-closed:

```text
Phase 0 source/protocol freeze
→ Phase 1 blind RD-Agent/DeepSeek proposal generation
→ proposal freeze + raw-response provenance seal
→ Phase 2 Train structural/counterfactual evaluation
→ Phase 3 family shortlist
→ Phase 4 research lock
→ Phase 5 locked observed re-validation
→ Phase 6 report
```

## What Gemini may commit

After the entire command succeeds, Gemini may commit generated material only:

```bash
git status

git add backtest/track_c_ranking_discovery/output/
git commit -m "docs(track_c): materialize sealed RD-Agent Track C artifacts"
git push origin codex/clean-latest-quant-trade-replay-pools
```

Before committing, confirm no research source changed:

```bash
git status --porcelain --   backtest/track_c_ranking_discovery   dashboard/skill_industry_eps_known.py
```

Changes under `backtest/track_c_ranking_discovery/output/` are expected. Any other `.py` or Track C protocol `.json` change is a hard stop.

## Failure handling

If `materialize` fails:

1. Preserve the terminal error and relevant generated log/provenance files.
2. Do **not** modify source code.
3. Do **not** rerun Phase 2/5 with altered code.
4. Return the failure to the code owner for a new source commit.

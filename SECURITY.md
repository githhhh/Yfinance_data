# Security Policy for This Public Repository

This repository is intentionally public. Treat every tracked file, commit, GitHub Actions log, generated Pages artifact, and Git history entry as internet-visible.

## Public / Private Boundary

Allowed public content:

- market-data download and normalization code;
- public market facts and BreakoutFollow pool data;
- deterministic review/projection fields used by the static Dashboard;
- GitHub Actions required to update data and deploy the static site.

Never commit or publish:

- brokerage account numbers, account hashes, positions, cost basis, orders, or portfolio state;
- OAuth access/refresh tokens, API keys, client/app secrets, passwords, cookies, private keys, or certificates;
- local `.env` files or credential JSON files;
- private research notes or data whose intended source repository is private.

The private `market_analysis` repository may be referenced as a submodule name/URL, but its content and credentials must never be copied into this repository.

## GitHub Pages Boundary

GitHub Pages is public. `dashboard/build_static.py` must publish row values only from `PUBLIC_DASHBOARD_ROW_FIELDS`.

Adding a new Pool column must **not** make it public automatically. A new field is added to the Pages whitelist only when the static UI intentionally consumes it and the field is confirmed safe for public exposure.

Browser code must not receive brokerage/account/position/credential data, even if that data is not rendered.

## Credential Handling

Use environment variables or GitHub Actions secrets for credentials. Do not pass real secrets as command-line examples in documentation or commit them in test fixtures.

Local credential files are ignored by `.gitignore`, including `.env*`, token/credential JSON files, private-key files, and certificate bundles.

Schwab OAuth token refreshes written by `SchwabRawTokenClient` are saved with owner-only `0600` permissions on supported Unix-like systems.

## GitHub Actions Hardening

- third-party/first-party reusable Actions are pinned to verified full commit SHAs rather than movable tags;
- read-only workflows use `persist-credentials: false`;
- data-update workflows also checkout without persisted credentials even though the job has `contents: write`;
- the write-capable `GITHUB_TOKEN` is exposed only to the final commit/push step;
- manual workflow inputs used by shell commands must be passed through environment variables and validated before use;
- dependencies installed in write-capable data-update jobs should be version-pinned and updated deliberately.

Do not replace these constraints with a long-lived PAT unless there is a demonstrated requirement that `GITHUB_TOKEN` cannot satisfy.

## Automated Scan

Run locally before publishing security-sensitive changes:

```bash
python security_scan.py --history
```

`.github/workflows/security-scan.yml` runs the same fail-closed scan on pull requests, pushes to `main`, weekly, and manual dispatch. It checks current tracked files, historically tracked sensitive filenames, common credential formats, literal secret assignments, and added source/config/documentation lines across Git history.

High-volume market-data history (`us/`, `results_pkl/`, `output/`) is excluded from historical patch scanning for performance, but the current tracked tree and sensitive filenames are still checked.

Commit-author email exposure is reported as a privacy warning only. Configure Git to use the GitHub noreply address for future commits when personal email disclosure is not desired.

## If a Real Secret Is Found

1. Revoke/rotate the credential immediately; deleting a file or commit is not sufficient.
2. Remove the credential from the current tree.
3. Rewrite Git history only when warranted, understanding that a previously public secret must still be treated as compromised.
4. Re-run `python security_scan.py --history` and CI.
5. Check GitHub Actions logs and Pages artifacts if the secret could have been rendered or printed there.

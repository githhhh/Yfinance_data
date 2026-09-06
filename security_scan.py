#!/usr/bin/env python3
"""Fail-closed secret scan for this public repository.

The scanner intentionally has no third-party dependencies. It checks:
- tracked files and historically tracked sensitive filenames;
- current tracked text for common credential formats and literal secrets;
- added lines across Git history (excluding high-volume market-data directories);
- commit author email privacy as a warning only.

It never prints matched secret values.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


MAX_CURRENT_FILE_BYTES = 2_000_000
HISTORY_EXCLUDED_PATHS = (
    ":(exclude)us/**",
    ":(exclude)results_pkl/**",
    ":(exclude)output/**",
)
SAFE_ENV_FILENAMES = {".env.example", ".env.sample", ".env.template"}
SENSITIVE_FILENAMES = {"credentials.json", "secrets.json", "token.json"}
SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}

PLACEHOLDER_MARKERS = (
    "example",
    "dummy",
    "placeholder",
    "redacted",
    "changeme",
    "your_",
    "your-",
    "xxx",
    "yyy",
    "<secret>",
    "<token>",
    "${",
    "$(",
)


@dataclass(frozen=True)
class Finding:
    kind: str
    source: str
    line: int | None = None

    def render(self) -> str:
        location = self.source if self.line is None else f"{self.source}:{self.line}"
        return f"{self.kind}: {location}"


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    return result.stdout


def _looks_placeholder(value: str) -> bool:
    normalized = value.strip().strip("\"'").lower()
    if not normalized:
        return True
    if normalized.startswith("$"):
        return True
    if any(marker in normalized for marker in PLACEHOLDER_MARKERS):
        return True
    compact = re.sub(r"[^a-z0-9]", "", normalized)
    if compact and len(set(compact)) == 1:
        return True
    return normalized in {"none", "null", "secret", "password", "token", "apikey", "api_key"}


TOKEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private-key", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,255}\b")),
    ("github-fine-grained-token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,255}\b")),
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("openai-api-key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("credential-url", re.compile(r"https?://[^\s/:@]+:[^\s/@]{6,}@[^\s]+")),
)

GENERIC_LITERAL = re.compile(
    r"(?i)\b(api[_-]?key|client[_-]?secret|app[_-]?secret|access[_-]?token|"
    r"refresh[_-]?token|password|passwd)\b\s*[:=]\s*[rRuUbBfF]*[\"']([^\"'\r\n]{8,})[\"']"
)
CLI_LITERAL = re.compile(
    r"(?i)--(?:app-secret|client-secret|api-key|access-token|refresh-token|password)"
    r"(?:=|\s+)([^\s]+)"
)


def _scan_line(text: str, source: str, line: int | None) -> list[Finding]:
    findings: list[Finding] = []
    for kind, pattern in TOKEN_PATTERNS:
        if pattern.search(text):
            findings.append(Finding(kind, source, line))

    for match in GENERIC_LITERAL.finditer(text):
        value = match.group(2)
        if not _looks_placeholder(value):
            findings.append(Finding(f"literal-{match.group(1).lower()}", source, line))

    for match in CLI_LITERAL.finditer(text):
        value = match.group(1).strip("\"'")
        if not _looks_placeholder(value):
            findings.append(Finding("cli-literal-secret", source, line))
    return findings


def _sensitive_path_reason(path: str) -> str | None:
    name = PurePosixPath(path).name.lower()
    if name in SAFE_ENV_FILENAMES:
        return None
    if name == ".env" or name.startswith(".env."):
        return "tracked-env-file"
    if name in SENSITIVE_FILENAMES or (name.startswith("token_") and name.endswith(".json")):
        return "tracked-credential-file"
    if PurePosixPath(name).suffix.lower() in SENSITIVE_SUFFIXES:
        return "tracked-private-key-file"
    return None


def _tracked_paths() -> list[str]:
    output = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return [item.decode("utf-8", errors="replace") for item in output.split(b"\0") if item]


def scan_current_tree() -> list[Finding]:
    findings: list[Finding] = []
    for relative in _tracked_paths():
        reason = _sensitive_path_reason(relative)
        if reason:
            findings.append(Finding(reason, relative))

        path = Path(relative)
        try:
            if not path.is_file() or path.stat().st_size > MAX_CURRENT_FILE_BYTES:
                continue
            raw = path.read_bytes()
        except OSError:
            continue
        if b"\0" in raw:
            continue
        text = raw.decode("utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            findings.extend(_scan_line(line, relative, line_number))
    return findings


def scan_historical_paths() -> list[Finding]:
    findings: list[Finding] = []
    seen: set[tuple[str, str]] = set()
    for raw_path in _git("log", "--all", "--name-only", "--format=").splitlines():
        path = raw_path.strip()
        if not path:
            continue
        reason = _sensitive_path_reason(path)
        key = (reason or "", path)
        if reason and key not in seen:
            seen.add(key)
            findings.append(Finding(f"historical-{reason}", path))
    return findings


def scan_history_patches() -> list[Finding]:
    findings: list[Finding] = []
    command = [
        "log",
        "--all",
        "--no-ext-diff",
        "--unified=0",
        "--format=COMMIT:%H",
        "--",
        ".",
        *HISTORY_EXCLUDED_PATHS,
    ]
    current_commit = "unknown"
    for line in _git(*command).splitlines():
        if line.startswith("COMMIT:"):
            current_commit = line.removeprefix("COMMIT:")[:12]
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        findings.extend(_scan_line(line[1:], f"history@{current_commit}", None))
    return findings


def commit_email_warnings() -> list[str]:
    emails = {
        email.strip()
        for email in _git("log", "--all", "--format=%ae").splitlines()
        if email.strip()
    }
    return sorted(
        email
        for email in emails
        if not email.endswith("@users.noreply.github.com")
        and email != "noreply@github.com"
    )


def _deduplicate(findings: list[Finding]) -> list[Finding]:
    return list(dict.fromkeys(findings))


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan the public repository for credential leakage.")
    parser.add_argument(
        "--history",
        action="store_true",
        help="Also scan historically tracked sensitive filenames and added source/config/doc lines.",
    )
    args = parser.parse_args()

    findings = scan_current_tree()
    if args.history:
        findings.extend(scan_historical_paths())
        findings.extend(scan_history_patches())
    findings = _deduplicate(findings)

    if findings:
        print("SECURITY SCAN FAILED: possible public credential exposure detected.", file=sys.stderr)
        for finding in findings:
            print(f"- {finding.render()}", file=sys.stderr)
        print("Matched values are intentionally not printed. Inspect the referenced source/commit.", file=sys.stderr)
        return 1

    print("Security scan passed: no tracked credential files or secret-like literals detected.")
    if args.history:
        exposed_emails = commit_email_warnings()
        if exposed_emails:
            print(
                "Privacy note: Git history contains non-noreply commit author email(s). "
                "This is metadata exposure, not a credential leak. Use a GitHub noreply email for future commits."
            )
        else:
            print("Commit metadata check passed: author emails use noreply-style addresses.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

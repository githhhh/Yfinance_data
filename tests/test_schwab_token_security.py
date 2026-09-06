from __future__ import annotations

import json
import stat

from data_providers.schwab_provider import SchwabCredentials, SchwabRawTokenClient


def test_raw_token_client_saves_token_owner_only(tmp_path) -> None:
    token_path = tmp_path / "schwab-token.json"
    client = SchwabRawTokenClient(SchwabCredentials(token_path=str(token_path)))

    client._save_token({"access_token": "fake_token", "refresh_token": "fake_token"})

    assert json.loads(token_path.read_text(encoding="utf-8"))["access_token"] == "fake_token"
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600


def test_raw_token_client_tightens_existing_token_permissions(tmp_path) -> None:
    token_path = tmp_path / "schwab-token.json"
    token_path.write_text('{"access_token":"fake_token"}', encoding="utf-8")
    token_path.chmod(0o644)
    client = SchwabRawTokenClient(SchwabCredentials(token_path=str(token_path)))

    client._save_token({"access_token": "fake_token"})

    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600

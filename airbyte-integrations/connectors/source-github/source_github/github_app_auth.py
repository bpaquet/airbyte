#
# Copyright (c) 2026 Airbyte, Inc., all rights reserved.
#

import base64
import time
from itertools import cycle
from typing import Any, Mapping, Optional

import jwt
import requests

from airbyte_cdk.sources.streams.http.requests_native_auth.abstract_token import AbstractHeaderAuthenticator

# GitHub always issues installation access tokens valid for exactly 1 hour; refresh a bit early so
# a token already handed to an in-flight request is never right at the edge of expiry.
_REFRESH_MARGIN_SECONDS = 120
_INSTALLATION_TOKEN_LIFETIME_SECONDS = 3600


class _InstallationTokenCache:
    """Mints and caches one GitHub App installation's access token, refreshing it on demand.

    Refresh happens lazily on the next `get_token()` call once the cached token is close to
    expiry — not once at sync startup — so a sync that runs for many hours never ends up making
    calls with a token that died partway through.
    """

    def __init__(self, app_id: str, installation_id: str, private_key: str) -> None:
        self._app_id = app_id
        self._installation_id = installation_id
        self._private_key = private_key
        self._token: Optional[str] = None
        self._expires_at: float = 0.0

    def _mint_app_jwt(self) -> str:
        now = int(time.time())
        payload = {"iat": now - 60, "exp": now + 540, "iss": self._app_id}
        return jwt.encode(payload, self._private_key, algorithm="RS256")

    def _refresh(self) -> None:
        response = requests.post(
            f"https://api.github.com/app/installations/{self._installation_id}/access_tokens",
            headers={
                "Authorization": f"Bearer {self._mint_app_jwt()}",
                "Accept": "application/vnd.github+json",
            },
            timeout=30,
        )
        response.raise_for_status()
        self._token = response.json()["token"]
        self._expires_at = time.time() + _INSTALLATION_TOKEN_LIFETIME_SECONDS - _REFRESH_MARGIN_SECONDS

    def get_token(self) -> str:
        if self._token is None or time.time() >= self._expires_at:
            self._refresh()
        return self._token  # type: ignore[return-value]


def _parse_entries(github_apps: str):
    entries = []
    for line in github_apps.splitlines():
        line = line.strip()
        if not line:
            continue
        app_id, installation_id, private_key_b64 = line.split(":", 2)
        private_key = base64.b64decode(private_key_b64).decode()
        entries.append((app_id, installation_id, private_key))
    return entries


class GithubAppMultiPemAuthenticator(AbstractHeaderAuthenticator):
    """
    Authenticates as one or more GitHub App installations from a single config field
    (`credentials.github_apps`): one entry per line, each formatted as
    `app_id:installation_id:private_key_base64` (the private key .pem, base64-encoded onto a
    single line so the whole thing fits a plain text field).

    Each entry gets its own installation-token cache, minted and refreshed independently and
    lazily — never all at once at startup — which is what lets a single sync outlive any one
    token's ~1h lifetime. With multiple entries, requests round-robin across them for extra
    rate-limit headroom, the same motivation as the existing comma-separated multi-PAT support.
    """

    def __init__(self, github_apps: str, auth_method: str = "token", auth_header: str = "Authorization") -> None:
        self._auth_method = auth_method
        self._auth_header = auth_header
        entries = _parse_entries(github_apps)
        if not entries:
            raise ValueError(
                "credentials.github_apps must have at least one 'app_id:installation_id:private_key_base64' line"
            )
        self._caches = [_InstallationTokenCache(app_id, installation_id, private_key) for app_id, installation_id, private_key in entries]
        self._tokens_iter = cycle(range(len(self._caches)))

    @property
    def auth_header(self) -> str:
        return self._auth_header

    @property
    def token(self) -> str:
        cache = self._caches[next(self._tokens_iter)]
        return f"{self._auth_method} {cache.get_token()}"

    def get_auth_header(self) -> Mapping[str, Any]:
        return {self.auth_header: self.token}

    def __call__(self, request):
        request.headers.update(self.get_auth_header())
        return request

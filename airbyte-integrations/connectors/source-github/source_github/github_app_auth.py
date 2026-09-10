#
# Copyright (c) 2026 Airbyte, Inc., all rights reserved.
#

import logging
import time
from itertools import cycle
from typing import Any, List, Mapping, Optional, Tuple

import jwt
import requests

from airbyte_cdk.sources.streams.http.requests_native_auth.abstract_token import AbstractHeaderAuthenticator

logger = logging.getLogger("airbyte")

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
        logger.info(
            "github_app_auth: minted installation token (app_id=%s, installation_id=%s, expires_at=%s)",
            self._app_id,
            self._installation_id,
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._expires_at)),
        )

    def get_token(self) -> str:
        if self._token is None or time.time() >= self._expires_at:
            self._refresh()
        return self._token  # type: ignore[return-value]


def _parse_entries(github_apps: str) -> List[Tuple[str, str, str]]:
    """Parses repeated {app_id line}\\n{installation_id line}\\n{PEM block} groups, blank lines
    between groups allowed. The PEM block is taken verbatim from its `-----BEGIN` line through
    its `-----END` line, so a real .pem file can be pasted as-is.
    """
    lines = github_apps.splitlines()
    n = len(lines)
    entries: List[Tuple[str, str, str]] = []
    i = 0

    def next_nonblank(i: int) -> int:
        while i < n and not lines[i].strip():
            i += 1
        return i

    while True:
        i = next_nonblank(i)
        if i >= n:
            break
        app_id = lines[i].strip()
        i = next_nonblank(i + 1)
        if i >= n:
            raise ValueError(f"credentials.github_apps: missing installation_id after app_id '{app_id}'")
        installation_id = lines[i].strip()
        i = next_nonblank(i + 1)
        if i >= n or not lines[i].strip().startswith("-----BEGIN"):
            raise ValueError(f"credentials.github_apps: expected a '-----BEGIN...' PEM block after installation_id '{installation_id}'")
        pem_lines = []
        found_end = False
        while i < n:
            pem_lines.append(lines[i])
            if lines[i].strip().startswith("-----END"):
                i += 1
                found_end = True
                break
            i += 1
        if not found_end:
            raise ValueError(f"credentials.github_apps: PEM block for app_id '{app_id}' never reached a '-----END...' line")
        entries.append((app_id, installation_id, "\n".join(pem_lines)))
    return entries


class GithubAppMultiPemAuthenticator(AbstractHeaderAuthenticator):
    """
    Authenticates as one or more GitHub App installations from a single config field
    (`credentials.github_apps`): repeated groups of app_id line, installation_id line, then the
    private key .pem pasted as-is (`-----BEGIN...` through `-----END...`) — repeat for more
    installations, blank lines between groups are fine.

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
            raise ValueError("credentials.github_apps must have at least one app_id/installation_id/PEM group")
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

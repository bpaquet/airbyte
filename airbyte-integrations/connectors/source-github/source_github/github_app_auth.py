#
# Copyright (c) 2026 Airbyte, Inc., all rights reserved.
#

import logging
import threading
import time
from dataclasses import InitVar, dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import jwt
import requests

from airbyte_cdk.models import FailureType
from airbyte_cdk.sources.declarative.auth.declarative_authenticator import DeclarativeAuthenticator
from airbyte_cdk.sources.declarative.interpolation.interpolated_string import InterpolatedString
from airbyte_cdk.utils import AirbyteTracedException

from .errors_handlers import is_rate_limited_response


logger = logging.getLogger("airbyte")

# GitHub always issues installation access tokens valid for exactly 1 hour; refresh a bit early so
# a token already handed to an in-flight request is never right at the edge of expiry.
_REFRESH_MARGIN_SECONDS = 120
_INSTALLATION_TOKEN_LIFETIME_SECONDS = 3600

# Same constants as the PAT path's sticky-token behavior (MultipleTokenAuthenticatorWithRateLimiter):
# stick with an installation until its tracked quota drops into this reserve, then rotate.
_BUDGET_MIN_RESERVE = 50
_MAX_WAIT_SECONDS = 60 * 120

# How far behind a locally-tracked reset a response's reset may be while still counting as the
# current window (ordinary disagreement between `refresh_quota`'s seeding and a later response's
# headers); anything older is a stale, out-of-order response and is ignored. Mirrors
# `RateLimitedMultipleTokenAuthenticator.RESET_SKEW_TOLERANCE` in the CDK.
_RESET_SKEW_TOLERANCE_SECONDS = 60.0


class _InstallationTokenCache:
    """Mints and caches one GitHub App installation's access token, refreshing it on demand, and
    tracks that installation's REST rate-limit quota so the authenticator can stick with it until
    exhausted (mirroring the PAT path) instead of blindly round-robining every call.

    Token refresh happens lazily on the next `get_token()` call once the cached token is close to
    expiry — not once at sync startup — so a sync that runs for many hours never ends up making
    calls with a token that died partway through.

    `remaining`/`reset_at` are seeded once from `refresh_quota()` and decremented locally per
    call thereafter; on their own they drift from GitHub's real count under concurrency (several
    partitions sharing this cache) and retries (a retried request spends real quota the local
    decrement never sees). `GithubAppMultiPemAuthenticator.update_from_response` closes that gap
    by reconciling these fields against the real headers of every response this cache's token
    sent, which is what lets rotation react to a genuine exhaustion immediately instead of only
    after the drift happens to line up.
    """

    def __init__(
        self,
        app_id: str,
        installation_id: str,
        private_key: str,
        on_token_minted: Optional[Callable[["_InstallationTokenCache", str, Optional[str]], None]] = None,
    ) -> None:
        self._app_id = app_id
        self._installation_id = installation_id
        self._private_key = private_key
        self._on_token_minted = on_token_minted
        self._token: Optional[str] = None
        self._expires_at: float = 0.0
        self.remaining: Optional[int] = None
        self.reset_at: Optional[float] = None
        # Guards `_token`/`_expires_at`: `get_token()` can be called concurrently for the same
        # installation (multiple in-flight partition reads sharing one authenticator), and without
        # this a race between the expiry check and the refresh could mint the token twice at once.
        self._token_lock = threading.Lock()

    def _mint_app_jwt(self) -> str:
        now = int(time.time())
        payload = {"iat": now - 60, "exp": now + 540, "iss": self._app_id}
        try:
            return jwt.encode(payload, self._private_key, algorithm="RS256")
        except Exception as e:
            # A malformed/garbage PEM surfaces here (e.g. PyJWT/cryptography's
            # "Could not deserialize key data..."). Wrap it the same way the installation-token
            # exchange's 401/403/404 are wrapped below, instead of letting a raw library
            # exception reach the user with no actionable guidance. Neither that exception nor
            # this message ever includes the key material itself.
            raise AirbyteTracedException(
                message=f"GitHub App authentication failed. The private key for app_id '{self._app_id}' could not be used to "
                "sign a JWT — please verify it is a valid, unencrypted PEM-formatted RSA private key.",
                internal_message=f"Failed to sign the GitHub App JWT for app_id={self._app_id}: {e}",
                failure_type=FailureType.config_error,
            ) from e

    def _refresh_token(self) -> None:
        response = requests.post(
            f"https://api.github.com/app/installations/{self._installation_id}/access_tokens",
            headers={
                "Authorization": f"Bearer {self._mint_app_jwt()}",
                "Accept": "application/vnd.github+json",
            },
            timeout=30,
        )
        if response.status_code in (401, 403, 404):
            raise AirbyteTracedException(
                message="GitHub App authentication failed. Please verify the app_id, installation_id and private key are correct.",
                internal_message=f"Installation token exchange failed for app_id={self._app_id}: {response.status_code} {response.text}",
                failure_type=FailureType.config_error,
            )
        response.raise_for_status()
        old_token = self._token
        self._token = response.json()["token"]
        self._expires_at = time.time() + _INSTALLATION_TOKEN_LIFETIME_SECONDS - _REFRESH_MARGIN_SECONDS
        if self._on_token_minted is not None:
            self._on_token_minted(self, self._token, old_token)
        logger.info(
            "github_app_auth: minted installation token (app_id=%s, installation_id=%s, expires_at=%s)",
            self._app_id,
            self._installation_id,
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._expires_at)),
        )

    def get_token(self) -> str:
        if self._token is None or time.time() >= self._expires_at:
            with self._token_lock:
                # Re-check inside the lock: another thread may have refreshed while this one
                # was waiting to acquire it, in which case minting again would be redundant.
                if self._token is None or time.time() >= self._expires_at:
                    self._refresh_token()
        return self._token  # type: ignore[return-value]

    def refresh_quota(self) -> None:
        """Seed `remaining`/`reset_at` from GitHub's own accounting for this installation.

        This is only ever a *seed*: `/rate_limit` was independently confirmed this session to
        not reliably reflect the same live counter as the `X-RateLimit-*` headers on real
        `/repos/...` calls for these installations, so it is used purely to bootstrap a cache on
        first use (and after a full exhaustion wait, when there is nothing better). Ongoing
        accuracy comes from `update_from_response` reconciling against real request headers.
        """
        response = requests.get(
            "https://api.github.com/rate_limit",
            headers={"Authorization": f"token {self.get_token()}", "Accept": "application/vnd.github+json"},
            timeout=30,
        )
        response.raise_for_status()
        core = response.json()["resources"]["core"]
        self.remaining = core["remaining"]
        self.reset_at = core["reset"]


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


@dataclass
class GithubAppMultiPemAuthenticator(DeclarativeAuthenticator):
    """
    Authenticates as one or more GitHub App installations from a single config field
    (`credentials.github_apps`): repeated groups of app_id line, installation_id line, then the
    private key .pem pasted as-is (`-----BEGIN...` through `-----END...`) — repeat for more
    installations, blank lines between groups are fine.

    Each entry gets its own installation-token cache, minted and refreshed independently and
    lazily — never all at once at startup — which is what lets a single sync outlive any one
    token's ~1h lifetime.

    With multiple entries, mirrors the PAT path's sticky-until-exhausted rotation
    (`MultipleTokenAuthenticatorWithRateLimiter`/`RateLimitedMultipleTokenAuthenticator`) instead
    of blindly round-robining every call: stays on one installation while it has quota, rotates
    to the next once it drops into the reserve, and waits out the earliest reset if all are
    exhausted.

    Also implements the CDK's `ResponseAwareAuthenticator`/`TokenRotatingAuthenticator` protocols
    (`airbyte_cdk.sources.streams.http.requests_native_auth.protocols`, structural — no base
    class needed), the same way `RateLimitedMultipleTokenAuthenticator` does for the PAT path.
    `HttpClient` calls `update_from_response` on every response so a cache's tracked quota is
    reconciled against what GitHub actually reports, instead of only ever drifting further from
    it via local decrements; and calls `has_alternative_token` before sleeping out a computed
    rate-limit backoff so a real exhaustion rotates to one of the other installations immediately
    rather than retrying the same dead one until the retry budget runs out.
    """

    config: Mapping[str, Any]
    parameters: InitVar[Mapping[str, Any]]
    github_apps: Union[InterpolatedString, str]

    def __post_init__(self, parameters: Mapping[str, Any]) -> None:
        # Parsing only — no validation, no network calls. `ModelToComponentFactory
        # .create_selective_authenticator` builds every branch under `authenticators:` eagerly,
        # not just the selected one, so this constructor also runs when the config is in "token"
        # mode and `github_apps` is empty. Raising here (or seeding quota, which would need a
        # real token) would break every PAT/OAuth user. Both are deferred to first actual use
        # in `_ensure_ready`, which only happens if this branch is the one truly selected.
        self._parameters = parameters
        self._caches: Optional[List[_InstallationTokenCache]] = None
        self._active_index = 0
        # Guards `_caches`/`_active_index` and every cache's `remaining` counter.
        # `ConcurrentDeclarativeSource` can read multiple partition streams in parallel, all
        # sharing this one authenticator instance (mirroring `RateLimitedMultipleTokenAuthenticator`,
        # which documents the same requirement) — without a lock, two threads racing the
        # check-then-decrement in `_next_available_cache`/`token` could both pick an already
        # exhausted cache, or step on each other's rotation of `_active_index`. Sleeping while
        # waiting out an exhaustion window happens outside the lock so one thread's wait never
        # blocks another from making progress.
        self._lock = threading.Lock()
        # Separate from `_lock`: a cache's token can be re-minted (`_InstallationTokenCache
        # .get_token`) without going through cache selection at all (e.g. a stream calling
        # `token` again after this authenticator already returned a soon-to-expire one), so
        # indexing which cache issued which token needs its own guard rather than piggybacking
        # on cache-selection's lock.
        self._token_index_lock = threading.Lock()
        self._token_to_cache: Dict[str, _InstallationTokenCache] = {}

    def _on_token_minted(self, cache: _InstallationTokenCache, new_token: str, old_token: Optional[str]) -> None:
        with self._token_index_lock:
            if old_token is not None:
                self._token_to_cache.pop(old_token, None)
            self._token_to_cache[new_token] = cache

    def _ensure_ready(self) -> None:
        if self._caches is not None:
            return
        with self._lock:
            if self._caches is not None:
                return
            github_apps_value = InterpolatedString.create(self.github_apps, parameters=self._parameters).eval(self.config)
            entries = _parse_entries(github_apps_value) if github_apps_value else []
            if not entries:
                raise ValueError("credentials.github_apps must have at least one app_id/installation_id/PEM group")
            caches = [
                _InstallationTokenCache(app_id, installation_id, private_key, on_token_minted=self._on_token_minted)
                for app_id, installation_id, private_key in entries
            ]
            for cache in caches:
                cache.refresh_quota()
            self._caches = caches

    def _select_cache_locked(self) -> Optional[_InstallationTokenCache]:
        """Must be called while holding `self._lock`. Returns the active cache with its
        `remaining` counter already decremented, or `None` if every cache is exhausted."""
        n = len(self._caches)
        for _ in range(n):
            cache = self._caches[self._active_index]
            if cache.remaining is None or cache.remaining > _BUDGET_MIN_RESERVE:
                if cache.remaining is not None:
                    cache.remaining -= 1
                return cache
            self._active_index = (self._active_index + 1) % n
        return None

    def _next_available_cache(self) -> _InstallationTokenCache:
        self._ensure_ready()
        while True:
            with self._lock:
                cache = self._select_cache_locked()
                if cache is not None:
                    return cache
                wait_seconds = max(0.0, min(c.reset_at for c in self._caches) - time.time())

            if wait_seconds > _MAX_WAIT_SECONDS:
                raise AirbyteTracedException(
                    message="Rate limit exceeded for all configured GitHub App installations.",
                    failure_type=FailureType.transient_error,
                )
            logger.info("github_app_auth: all installations exhausted, sleeping %.0fs until the earliest reset", wait_seconds)
            time.sleep(wait_seconds)
            for cache in self._caches:
                cache.refresh_quota()

    @property
    def auth_header(self) -> str:
        return "Authorization"

    @property
    def token(self) -> str:
        cache = self._next_available_cache()
        return f"token {cache.get_token()}"

    def _cache_for_request(self, request: requests.PreparedRequest) -> Optional[_InstallationTokenCache]:
        """Recover the cache whose token signed `request`, from its `Authorization` header.

        Prefix-checked rather than assumed: the header may belong to a different auth mode (this
        method is a no-op unless `github_apps` mode actually built this authenticator), and
        slicing blindly would leave membership in `_token_to_cache` as the only thing standing
        between a mangled value and a wrong attribution.
        """
        value = request.headers.get(self.auth_header, "")
        prefix = "token "
        if not value.startswith(prefix):
            return None
        token = value[len(prefix) :].strip()
        with self._token_index_lock:
            return self._token_to_cache.get(token)

    def update_from_response(self, request: requests.PreparedRequest, response: requests.Response) -> None:
        """Reconcile the sending cache's tracked quota against what GitHub actually reported.

        Called by `HttpClient` once per HTTP attempt (skipping cache hits, which carry stale
        rate-limit headers and consumed no quota). Never raises: an authenticator that cannot
        interpret a response should be a no-op, not a failed sync, and `HttpClient` already
        treats a raised exception here as "quota bookkeeping degraded to local counters only" —
        but that degradation is exactly the bug this method exists to prevent, so it is better
        to fail closed (do nothing) on anything unexpected than to raise into that fallback.
        """
        try:
            if self._caches is None:
                return
            cache = self._cache_for_request(request)
            if cache is None:
                return
            if getattr(response, "from_cache", False):
                return
            self._reconcile_cache(cache, response)
        except Exception:
            logger.debug("github_app_auth: failed to update quota state from response", exc_info=True)

    def _reconcile_cache(self, cache: _InstallationTokenCache, response: requests.Response) -> None:
        remaining = self._header_int(response, "X-RateLimit-Remaining")
        reset_at = self._header_float(response, "X-RateLimit-Reset")
        # GitHub's *primary* rate limit always reports `X-RateLimit-Remaining` explicitly, zero
        # included, on the response that hits it — so a genuine exhaustion is fully captured by
        # `remaining` above with no inference needed. A *secondary* rate limit (abuse detection)
        # rejects with only `Retry-After` and no quota headers at all: that is a different,
        # shared-across-credentials limit, not this cache's primary quota being spent, and must
        # NOT zero `remaining` here — doing so would incorrectly "exhaust" a healthy cache and
        # rotate onto another one that is about to hit the exact same secondary rejection.
        # 403 is not exclusively a rate-limit code on GitHub either (missing scopes, SSO, a
        # disabled repo feature), so status code alone is never used to infer exhaustion.
        if remaining is None and reset_at is None:
            return
        # Still used below to let a real, explicit zero override an otherwise-stale-looking
        # response — never to invent a zero that was not actually reported.
        is_exhaustion_signal = is_rate_limited_response(response, lambda _body: False, logger)

        with self._lock:
            if reset_at is not None and (cache.reset_at is None or reset_at > cache.reset_at):
                # The quota window rolled over: the local count describes a window that no
                # longer exists, so take the server's numbers wholesale.
                cache.reset_at = reset_at
                cache.remaining = remaining
            elif remaining is not None and (
                cache.remaining is None
                or (remaining <= 0 and is_exhaustion_signal)
                or reset_at is None
                or cache.reset_at is None
                or reset_at >= cache.reset_at - _RESET_SKEW_TOLERANCE_SECONDS
            ):
                # Same window: only ever tighten the estimate, since responses can arrive out of
                # order and a slow one carrying a stale, higher count must not resurrect a
                # counter another thread already brought down — except a genuine exhaustion
                # signal always wins regardless of ordering, or a sync fresh off `refresh_quota`
                # (no locally tracked reset yet) always accepts the first real reading it sees.
                cache.remaining = remaining if cache.remaining is None else min(cache.remaining, remaining)

    def has_alternative_token(self, request: requests.PreparedRequest) -> bool:
        """Whether a different installation could serve this request right now.

        Deliberately narrow, mirroring `RateLimitedMultipleTokenAuthenticator`: reports True only
        when the installation that *sent* the request is known-exhausted for the reserve and some
        other configured installation is not — so the next attempt is guaranteed to rotate rather
        than immediately hammering the same rejected installation again. A rejection while the
        sender still shows headroom is not about that installation specifically (e.g. a limit
        shared across all of them), and the computed backoff remains the right response.
        """
        try:
            cache = self._cache_for_request(request)
            if cache is None or self._caches is None:
                return False
            with self._lock:
                sender_exhausted = cache.remaining is not None and cache.remaining <= _BUDGET_MIN_RESERVE
                if not sender_exhausted:
                    return False
                return any(
                    other is not cache and (other.remaining is None or other.remaining > _BUDGET_MIN_RESERVE) for other in self._caches
                )
        except Exception:
            return False

    @staticmethod
    def _header_int(response: requests.Response, header: str) -> Optional[int]:
        value = response.headers.get(header)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _header_float(response: requests.Response, header: str) -> Optional[float]:
        value = response.headers.get(header)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

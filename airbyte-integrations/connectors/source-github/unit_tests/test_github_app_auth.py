#
# Copyright (c) 2026 Airbyte, Inc., all rights reserved.
#

import logging
import threading

import jwt
import pytest
import requests
from freezegun import freeze_time

from airbyte_cdk.models import FailureType
from airbyte_cdk.sources.declarative.auth.rate_limited_multiple_token import (
    RateLimitedMultipleTokenAuthenticator,
)
from airbyte_cdk.sources.declarative.models.declarative_component_schema import (
    SelectiveAuthenticator as SelectiveAuthenticatorModel,
)
from airbyte_cdk.utils import AirbyteTracedException
from components import GithubAppMultiPemAuthenticator, _InstallationTokenCache, _parse_entries, _shared_state_by_credentials

from .utils import make_source


# Captured before the autouse `_mock_jwt` fixture below stubs out `jwt.encode` for every test in
# this module, so tests that need the *real* signing failure path can restore it.
_REAL_JWT_ENCODE = jwt.encode


FAKE_PEM = (
    "-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJBAKj34GkxFhD90vcNLYLInFEX6Ppy1tPf9Cnzj4p4WGeKLs1Pt8Qu\n-----END RSA PRIVATE KEY-----"
)


def _github_apps_field(*entries):
    return "\n\n".join(f"{app_id}\n{installation_id}\n{pem}" for app_id, installation_id, pem in entries)


def _access_token_url(installation_id):
    return f"https://api.github.com/app/installations/{installation_id}/access_tokens"


def _authenticator(github_apps, **config):
    return GithubAppMultiPemAuthenticator(config={"credentials": {"github_apps": github_apps}, **config}, parameters={})


def _sign(authenticator):
    """Drive a request through the real signing path (`__call__`) rather than reading the
    `token` property directly: quota accounting now lives in `__call__`, since `HttpRequester`
    reads `token` once already as a side-effect-free header preview before `HttpClient` invokes
    `__call__` for the actual send."""
    request = requests.Request("GET", "https://api.github.com/").prepare()
    authenticator(request)
    return request.headers["Authorization"]


def _selective_authenticator_from_source(source):
    """Builds the manifest's actual `requester_base.authenticator` (a `SelectiveAuthenticator`)
    the way each stream's requester does — a fresh `create_component` call every time, exactly
    like two different streams independently resolving the same manifest definition."""
    return source._constructor.create_component(
        model_type=SelectiveAuthenticatorModel,
        component_definition=source.resolved_manifest["definitions"]["requester_base"]["authenticator"],
        config=source._config,
    )


def _selective_authenticator(config):
    """Same as `_selective_authenticator_from_source`, building the source from a fresh config
    first — the regression that matters is the manifest wiring, not just the class."""
    return _selective_authenticator_from_source(make_source(catalog=None, config=config, state=None))


@pytest.fixture(autouse=True)
def _mock_jwt(monkeypatch):
    # Signing a real JWT needs a real RSA key; we only test our own usage of PyJWT, not PyJWT
    # itself, so stub it out with a deterministic fake.
    monkeypatch.setattr("components.jwt.encode", lambda payload, key, algorithm: "fake-app-jwt")


@pytest.fixture(autouse=True)
def _reset_shared_state():
    # State is shared per credentials, and nearly every test here reuses the same fixture
    # app_id/installation_id/PEM.
    _shared_state_by_credentials.clear()
    yield
    _shared_state_by_credentials.clear()


class TestParseEntries:
    def test_single_entry(self):
        assert _parse_entries(_github_apps_field(("111", "222", FAKE_PEM))) == [("111", "222", FAKE_PEM)]

    def test_multiple_entries_with_blank_lines_between(self):
        field = _github_apps_field(("111", "222", FAKE_PEM), ("333", "444", FAKE_PEM))
        entries = _parse_entries(field)
        assert [(app_id, installation_id) for app_id, installation_id, _ in entries] == [("111", "222"), ("333", "444")]

    def test_empty_string_returns_no_entries(self):
        assert _parse_entries("") == []

    def test_missing_installation_id_raises(self):
        with pytest.raises(AirbyteTracedException, match="missing installation_id") as exc_info:
            _parse_entries("111")
        assert exc_info.value.failure_type == FailureType.config_error

    def test_missing_pem_raises(self):
        with pytest.raises(AirbyteTracedException, match="expected a '-----BEGIN") as exc_info:
            _parse_entries("111\n222\nnot a pem")
        assert exc_info.value.failure_type == FailureType.config_error

    def test_unterminated_pem_raises(self):
        with pytest.raises(AirbyteTracedException, match="never reached a '-----END") as exc_info:
            _parse_entries("111\n222\n-----BEGIN RSA PRIVATE KEY-----\nAAAA")
        assert exc_info.value.failure_type == FailureType.config_error


class TestInstallationTokenMintingAndCaching:
    def test_mints_once_and_caches(self, requests_mock):
        mint_mock = requests_mock.post(_access_token_url("222"), json={"token": "ghs_abc"})
        requests_mock.get("https://api.github.com/rate_limit", json={"resources": {"core": {"remaining": 5000, "reset": 4070908800}}})
        authenticator = _authenticator(_github_apps_field(("111", "222", FAKE_PEM)))
        authenticator._ensure_ready()  # lazy: entries are only parsed/seeded on first actual use
        cache = authenticator._state.caches[0]
        assert cache.get_token() == "ghs_abc"
        assert cache.get_token() == "ghs_abc"
        assert mint_mock.call_count == 1  # _ensure_ready()'s refresh_quota() already minted once; both get_token() calls hit the cache

    def test_refreshes_after_expiry(self, requests_mock):
        requests_mock.post(_access_token_url("222"), [{"json": {"token": "ghs_first"}}, {"json": {"token": "ghs_second"}}])
        requests_mock.get("https://api.github.com/rate_limit", json={"resources": {"core": {"remaining": 5000, "reset": 4070908800}}})
        with freeze_time("2026-01-01T00:00:00Z") as frozen:
            authenticator = _authenticator(_github_apps_field(("111", "222", FAKE_PEM)))
            authenticator._ensure_ready()
            cache = authenticator._state.caches[0]
            assert cache.get_token() == "ghs_first"
            frozen.tick(delta=__import__("datetime").timedelta(hours=1, minutes=1))
            assert cache.get_token() == "ghs_second"

    def test_bad_credentials_raise_traced_config_error(self, requests_mock):
        requests_mock.post(_access_token_url("222"), status_code=401, json={"message": "Bad credentials"})
        authenticator = _authenticator(_github_apps_field(("111", "222", FAKE_PEM)))
        with pytest.raises(AirbyteTracedException) as exc_info:
            _sign(authenticator)  # construction itself must never raise — see test_manifest_selective_authenticator_builds_in_token_mode
        assert exc_info.value.failure_type == FailureType.config_error


class TestStickyRotation:
    def test_stays_on_active_installation_while_it_has_quota(self, requests_mock):
        requests_mock.post(_access_token_url("222"), json={"token": "ghs_a"})
        requests_mock.post(_access_token_url("444"), json={"token": "ghs_b"})
        requests_mock.get(
            "https://api.github.com/rate_limit",
            [
                {"json": {"resources": {"core": {"remaining": 500, "reset": 4070908800}}}},
                {"json": {"resources": {"core": {"remaining": 500, "reset": 4070908800}}}},
            ],
        )
        authenticator = _authenticator(_github_apps_field(("111", "222", FAKE_PEM), ("333", "444", FAKE_PEM)))
        assert _sign(authenticator) == "token ghs_a"
        assert _sign(authenticator) == "token ghs_a"
        assert _sign(authenticator) == "token ghs_a"

    def test_rotates_once_active_drops_into_reserve(self, requests_mock):
        requests_mock.post(_access_token_url("222"), json={"token": "ghs_a"})
        requests_mock.post(_access_token_url("444"), json={"token": "ghs_b"})
        requests_mock.get(
            "https://api.github.com/rate_limit",
            [
                {"json": {"resources": {"core": {"remaining": 51, "reset": 4070908800}}}},
                {"json": {"resources": {"core": {"remaining": 500, "reset": 4070908800}}}},
            ],
        )
        authenticator = _authenticator(_github_apps_field(("111", "222", FAKE_PEM), ("333", "444", FAKE_PEM)))
        assert _sign(authenticator) == "token ghs_a"  # remaining 51 -> 50, still above reserve check next time? no: 50 is not > 50
        assert _sign(authenticator) == "token ghs_b"  # cache 0 now at 50, not > _BUDGET_MIN_RESERVE (50), rotates

    def test_cascades_through_four_installations(self, requests_mock):
        """The rotation logic (`n = len(self._caches)`, generic loop) is written for any N, but
        every other test here only exercises N=2. This confirms it actually cascades correctly
        across more than two — install 1/2/3 each have just enough quota for one call before
        dropping into reserve, install 4 has plenty, so the sequence should visit all four in
        order and then stay on the fourth.
        """
        for installation_id, token in (("222", "ghs_a"), ("444", "ghs_b"), ("666", "ghs_c"), ("888", "ghs_d")):
            requests_mock.post(_access_token_url(installation_id), json={"token": token})
        requests_mock.get(
            "https://api.github.com/rate_limit",
            [
                {"json": {"resources": {"core": {"remaining": 51, "reset": 4070908800}}}},
                {"json": {"resources": {"core": {"remaining": 51, "reset": 4070908800}}}},
                {"json": {"resources": {"core": {"remaining": 51, "reset": 4070908800}}}},
                {"json": {"resources": {"core": {"remaining": 500, "reset": 4070908800}}}},
            ],
        )
        authenticator = _authenticator(_github_apps_field(("111", "222", FAKE_PEM), ("333", "444", FAKE_PEM), ("555", "666", FAKE_PEM), ("777", "888", FAKE_PEM)))
        assert [_sign(authenticator) for _ in range(5)] == [
            "token ghs_a",
            "token ghs_b",
            "token ghs_c",
            "token ghs_d",
            "token ghs_d",  # install 4 has plenty of quota (500 -> 499), stays put
        ]

    def test_all_exhausted_sleeps_until_earliest_reset_then_reseeds(self, requests_mock, monkeypatch):
        requests_mock.post(_access_token_url("222"), json={"token": "ghs_a"})
        requests_mock.post(_access_token_url("444"), json={"token": "ghs_b"})
        requests_mock.get(
            "https://api.github.com/rate_limit",
            [
                {"json": {"resources": {"core": {"remaining": 0, "reset": 1030}}}},
                {"json": {"resources": {"core": {"remaining": 0, "reset": 2000}}}},
                {"json": {"resources": {"core": {"remaining": 5000, "reset": 4070908800}}}},
                {"json": {"resources": {"core": {"remaining": 5000, "reset": 4070908800}}}},
            ],
        )
        sleeps = []
        monkeypatch.setattr("components.time.sleep", lambda seconds: sleeps.append(seconds))
        monkeypatch.setattr("components.time.time", lambda: 999.0)
        authenticator = _authenticator(_github_apps_field(("111", "222", FAKE_PEM), ("333", "444", FAKE_PEM)))
        token = _sign(authenticator)
        assert token in ("token ghs_a", "token ghs_b")
        assert sleeps == [31.0]  # earliest reset (1030) minus frozen "now" (999)

    def test_all_exhausted_wait_is_floored_against_a_stale_reset(self, requests_mock, monkeypatch):
        """A reset timestamp at or behind "now" (clock skew, or read right at the boundary)
        must not collapse the wait to ~0 and busy-loop re-querying /rate_limit."""
        requests_mock.post(_access_token_url("222"), json={"token": "ghs_a"})
        requests_mock.get(
            "https://api.github.com/rate_limit",
            [
                {"json": {"resources": {"core": {"remaining": 0, "reset": 999}}}},
                {"json": {"resources": {"core": {"remaining": 5000, "reset": 4070908800}}}},
            ],
        )
        sleeps = []
        monkeypatch.setattr("components.time.sleep", lambda seconds: sleeps.append(seconds))
        monkeypatch.setattr("components.time.time", lambda: 999.0)
        authenticator = _authenticator(_github_apps_field(("111", "222", FAKE_PEM)))
        _sign(authenticator)
        assert sleeps == [5.0]  # floored to _MIN_EXHAUSTION_WAIT_SECONDS, not the raw 0.0


class TestMintAppJwtErrors:
    def test_malformed_private_key_raises_traced_config_error(self, monkeypatch):
        # Override the module-level autouse stub for this test only, so `_mint_app_jwt` exercises
        # the real PyJWT/cryptography failure it is meant to wrap.
        monkeypatch.setattr("components.jwt.encode", _REAL_JWT_ENCODE)
        cache = _InstallationTokenCache(app_id="111", installation_id="222", private_key="not a real pem", api_url="https://api.github.com")
        with pytest.raises(AirbyteTracedException) as exc_info:
            cache._mint_app_jwt()
        assert exc_info.value.failure_type == FailureType.config_error
        # The underlying library error is reported, but never the key material itself.
        assert "not a real pem" not in exc_info.value.internal_message
        assert "not a real pem" not in exc_info.value.message


class TestConcurrency:
    def test_concurrent_token_calls_do_not_lose_decrements(self, requests_mock):
        """Regression test for the sticky-rotation state (`_active_index` and each cache's
        `remaining`) being read/decremented under a lock. Without it, concurrent threads racing
        the check-then-decrement could both act on a stale `remaining` value and lose updates —
        this pins the counter to exactly `seeded - total_calls` after every thread finishes.
        """
        requests_mock.post(_access_token_url("222"), json={"token": "ghs_a"})
        n_threads, calls_per_thread = 20, 25
        total_calls = n_threads * calls_per_thread
        seeded = total_calls + 1000  # comfortably above the reserve for the whole run: no rotation/waits involved
        requests_mock.get(
            "https://api.github.com/rate_limit",
            json={"resources": {"core": {"remaining": seeded, "reset": 4070908800}}},
        )
        authenticator = _authenticator(_github_apps_field(("111", "222", FAKE_PEM)))

        barrier = threading.Barrier(n_threads)

        def worker():
            barrier.wait()
            for _ in range(calls_per_thread):
                _sign(authenticator)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert authenticator._state.caches[0].remaining == seeded - total_calls

    def test_concurrent_first_use_seeds_quota_once(self, requests_mock):
        """Regression test for double-checked locking in `_ensure_ready`: concurrent first calls
        to `.token` must parse `github_apps` and seed quota exactly once, not once per thread.
        """
        requests_mock.post(_access_token_url("222"), json={"token": "ghs_a"})
        rate_limit_mock = requests_mock.get(
            "https://api.github.com/rate_limit", json={"resources": {"core": {"remaining": 5000, "reset": 4070908800}}}
        )
        authenticator = _authenticator(_github_apps_field(("111", "222", FAKE_PEM)))

        n_threads = 10
        barrier = threading.Barrier(n_threads)

        def worker():
            barrier.wait()
            _sign(authenticator)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert rate_limit_mock.call_count == 1


class TestGitHubEnterpriseServerRateLimitingDisabled:
    def test_refresh_quota_treats_404_as_untracked(self, requests_mock):
        """GHES with rate limiting disabled answers `/rate_limit` with 404 — must leave the pool
        untracked (never exhausted) instead of raising and crashing check/sync, mirroring
        `requester_base.authenticator`'s `QuotaStatusSource.unavailable_status_codes: [404]`."""
        requests_mock.post(_access_token_url("222"), json={"token": "ghs_a"})
        requests_mock.get("https://api.github.com/rate_limit", status_code=404, json={"message": "Rate limiting is not enabled."})
        authenticator = _authenticator(_github_apps_field(("111", "222", FAKE_PEM)))

        assert _sign(authenticator) == "token ghs_a"
        assert authenticator._state.caches[0].remaining is None


class TestMaxWaitingTimeOverride:
    def test_check_overrides_apply_to_the_exhaustion_wait_ceiling(self, requests_mock, monkeypatch):
        """`check`'s `config_overrides.max_waiting_time: 1` must fail fast here too, not only on
        the token path — otherwise an exhausted-quota `check` with GitHub App credentials hangs
        for the hardcoded default instead of raising quickly."""
        requests_mock.post(_access_token_url("222"), json={"token": "ghs_a"})
        requests_mock.get(
            "https://api.github.com/rate_limit",
            json={"resources": {"core": {"remaining": 0, "reset": 4070908800}}},
        )
        monkeypatch.setattr("components.time.sleep", lambda seconds: None)
        authenticator = _authenticator(_github_apps_field(("111", "222", FAKE_PEM)), max_waiting_time=1)

        with pytest.raises(AirbyteTracedException) as exc_info:
            _sign(authenticator)

        assert exc_info.value.failure_type == FailureType.transient_error

    def test_defaults_to_120_minutes_when_absent(self):
        authenticator = _authenticator("")
        assert authenticator._max_wait_seconds == 120 * 60

    def test_respects_a_configured_value(self):
        authenticator = _authenticator("", max_waiting_time=5)
        assert authenticator._max_wait_seconds == 5 * 60


class TestGitHubEnterpriseServerApiUrl:
    def test_installation_token_and_rate_limit_use_configured_api_url(self, requests_mock):
        """GitHub App auth must work against GHES like the token-based auth methods do, instead
        of always hitting public GitHub regardless of `api_url`."""
        ghes_token_mock = requests_mock.post(
            "https://github.company.org/api/v3/app/installations/222/access_tokens", json={"token": "ghs_a"}
        )
        ghes_quota_mock = requests_mock.get(
            "https://github.company.org/api/v3/rate_limit", json={"resources": {"core": {"remaining": 5000, "reset": 4070908800}}}
        )
        authenticator = _authenticator(_github_apps_field(("111", "222", FAKE_PEM)), api_url="https://github.company.org/api/v3/")
        assert _sign(authenticator) == "token ghs_a"
        assert ghes_token_mock.called
        assert ghes_quota_mock.called

    def test_defaults_to_public_github_when_api_url_is_absent(self, requests_mock):
        requests_mock.post(_access_token_url("222"), json={"token": "ghs_a"})
        requests_mock.get("https://api.github.com/rate_limit", json={"resources": {"core": {"remaining": 5000, "reset": 4070908800}}})
        authenticator = _authenticator(_github_apps_field(("111", "222", FAKE_PEM)))
        assert _sign(authenticator) == "token ghs_a"


class TestManifestWiring:
    def _config(self, github_apps_field):
        return {"credentials": {"github_apps": github_apps_field}, "repositories": ["org/repo"]}

    def test_manifest_selective_authenticator_builds_in_github_apps_mode(self, requests_mock):
        """Regression test: `ModelToComponentFactory.create_selective_authenticator` builds every
        branch under `authenticators:` eagerly, not just the selected one. This is the only test
        that exercises the manifest's `SelectiveAuthenticator` wiring the way `check`/`streams()`
        do — it alone would have caught the "Authentication tokens are missing from the
        configuration" crash from the token branch's `RateLimitedMultipleTokenAuthenticator`
        being constructed with an empty token list.
        """
        requests_mock.post(_access_token_url("222"), json={"token": "ghs_a"})
        requests_mock.get("https://api.github.com/rate_limit", json={"resources": {"core": {"remaining": 5000, "reset": 4070908800}}})
        authenticator = _selective_authenticator(self._config(_github_apps_field(("111", "222", FAKE_PEM))))
        assert isinstance(authenticator, GithubAppMultiPemAuthenticator)

    def test_manifest_selective_authenticator_builds_in_token_mode(self, rate_limit_mock_response):
        """Symmetric regression test: in "token" mode, `credentials.github_apps` is empty, but
        the "github_apps" branch is still constructed eagerly alongside the selected "token"
        branch. `GithubAppMultiPemAuthenticator` must not validate/raise at construction time —
        only lazily, on first actual use — or every PAT/OAuth user would crash on an empty
        `github_apps` the same way github_apps-mode users crashed on an empty `tokens` list.
        """
        config = {"access_token": "pat-token", "repositories": ["org/repo"]}
        authenticator = _selective_authenticator(config)
        assert isinstance(authenticator, RateLimitedMultipleTokenAuthenticator)

    def test_manifest_selective_authenticator_builds_in_oauth_mode(self, rate_limit_mock_response):
        """Same regression as above, but for the OAuth credentials shape specifically
        (credentials.access_token + client_id/client_secret) rather than the legacy root-level
        access_token — a different code path through `ConfigNormalization`."""
        config = {
            "credentials": {"access_token": "oauth-token", "client_id": "id", "client_secret": "secret"},
            "repositories": ["org/repo"],
        }
        authenticator = _selective_authenticator(config)
        assert isinstance(authenticator, RateLimitedMultipleTokenAuthenticator)

    def test_config_normalization_derives_auth_mode(self):
        source = make_source(catalog=None, config=self._config(_github_apps_field(("111", "222", FAKE_PEM))), state=None)
        assert source._config["credentials"]["auth_mode"] == "github_apps"

    def test_config_normalization_defaults_auth_mode_to_token(self):
        source = make_source(catalog=None, config={"access_token": "pat-token", "repositories": ["org/repo"]}, state=None)
        assert source._config["credentials"]["auth_mode"] == "token"

    def test_config_normalization_handles_explicitly_null_credentials(self):
        """`credentials: null` is reachable via the API/Terraform/embedded use, same as the
        `api_url`/`max_waiting_time` cases this connector already guards against."""
        source = make_source(catalog=None, config={"access_token": "pat-token", "credentials": None, "repositories": ["org/repo"]}, state=None)
        assert source._config["credentials"]["auth_mode"] == "token"


class TestSharedStateAcrossStreams:
    """`ModelToComponentFactory.create_custom_component` never caches instances the way it
    caches `RateLimitedMultipleTokenAuthenticator` for the token path — every stream's requester
    gets a fresh `GithubAppMultiPemAuthenticator` object. Without `_get_shared_state`, each would
    mint its own installation tokens and track its own quota, unaware of every other stream's
    consumption."""

    def test_two_authenticators_built_from_the_same_config_share_quota_state(self, requests_mock):
        requests_mock.post(_access_token_url("222"), json={"token": "ghs_a"})
        requests_mock.get("https://api.github.com/rate_limit", json={"resources": {"core": {"remaining": 5000, "reset": 4070908800}}})
        config = {"credentials": {"github_apps": _github_apps_field(("111", "222", FAKE_PEM))}, "repositories": ["org/repo"]}
        source = make_source(catalog=None, config=config, state=None)

        # Two separate CustomAuthenticator instances, exactly as two different streams' requesters
        # would each independently build one by resolving the same manifest definition.
        first = _selective_authenticator_from_source(source)
        second = _selective_authenticator_from_source(source)
        assert first is not second  # the factory really doesn't cache CustomAuthenticator...

        _sign(first)  # ...but charging quota through one must be visible to the other.
        assert second._state.caches[0].remaining == 4999

    def test_sources_using_the_same_installations_share_quota_state(self, requests_mock):
        """GitHub counts quota per installation, so two sources in one process configured with the
        same installations must draw down one shared estimate, not two independent ones."""
        requests_mock.post(_access_token_url("222"), json={"token": "ghs_a"})
        requests_mock.get("https://api.github.com/rate_limit", json={"resources": {"core": {"remaining": 5000, "reset": 4070908800}}})

        def _fresh_config():
            return {"credentials": {"github_apps": _github_apps_field(("111", "222", FAKE_PEM))}, "repositories": ["org/repo"]}

        first = _selective_authenticator_from_source(make_source(catalog=None, config=_fresh_config(), state=None))
        second = _selective_authenticator_from_source(make_source(catalog=None, config=_fresh_config(), state=None))

        _sign(first)
        assert second._state is first._state
        assert second._state.caches[0].remaining == 4999


class TestUnexpectedHttpErrorsAreClassified:
    def test_5xx_during_token_exchange_is_a_transient_error(self, requests_mock):
        requests_mock.post(_access_token_url("222"), status_code=502, text="bad gateway")
        authenticator = _authenticator(_github_apps_field(("111", "222", FAKE_PEM)))

        with pytest.raises(AirbyteTracedException) as exc_info:
            _sign(authenticator)

        assert exc_info.value.failure_type == FailureType.transient_error

    def test_5xx_during_quota_seeding_is_a_transient_error(self, requests_mock):
        requests_mock.post(_access_token_url("222"), json={"token": "ghs_a"})
        requests_mock.get("https://api.github.com/rate_limit", status_code=503, text="service unavailable")
        authenticator = _authenticator(_github_apps_field(("111", "222", FAKE_PEM)))

        with pytest.raises(AirbyteTracedException) as exc_info:
            _sign(authenticator)

        assert exc_info.value.failure_type == FailureType.transient_error

    def test_401_during_quota_seeding_is_a_config_error_not_transient(self, requests_mock):
        """A revoked installation token must read as an auth failure, not as a retryable blip —
        same rule the manifest documents for the token path's `QuotaStatusSource` (401/403 must
        never be treated as 'quota tracking unavailable')."""
        requests_mock.post(_access_token_url("222"), json={"token": "ghs_a"})
        requests_mock.get("https://api.github.com/rate_limit", status_code=401, json={"message": "Bad credentials"})
        authenticator = _authenticator(_github_apps_field(("111", "222", FAKE_PEM)))

        with pytest.raises(AirbyteTracedException) as exc_info:
            _sign(authenticator)

        assert exc_info.value.failure_type == FailureType.config_error


class TestSingleChargePerRequest:
    """`HttpRequester._request_headers()` reads `get_auth_header()` before `HttpClient` invokes
    `__call__` for the actual send; quota must be charged once per real attempt, from `__call__`."""

    def test_header_preview_is_empty_and_call_charges_once(self, requests_mock):
        requests_mock.post(_access_token_url("222"), json={"token": "ghs_a"})
        requests_mock.get("https://api.github.com/rate_limit", json={"resources": {"core": {"remaining": 5000, "reset": 4070908800}}})
        authenticator = _authenticator(_github_apps_field(("111", "222", FAKE_PEM)))

        assert authenticator.get_auth_header() == {}
        assert _sign(authenticator) == "token ghs_a"
        assert authenticator._state.caches[0].remaining == 4999

    def test_graphql_requests_are_not_charged_to_the_rest_quota(self, requests_mock):
        """GraphQL has its own point budget; charging it to the REST `core` estimate would rotate or
        sleep on a budget the request isn't spending."""
        requests_mock.post(_access_token_url("222"), json={"token": "ghs_a"})
        requests_mock.get("https://api.github.com/rate_limit", json={"resources": {"core": {"remaining": 5000, "reset": 4070908800}}})
        authenticator = _authenticator(_github_apps_field(("111", "222", FAKE_PEM)))

        request = requests.Request("POST", "https://api.github.com/graphql").prepare()
        authenticator(request)

        assert request.headers["Authorization"] == "token ghs_a"
        assert authenticator._state.caches[0].remaining == 5000


class TestQuotaLogging:
    def _two_installations(self, requests_mock, first_remaining=5000):
        requests_mock.post(_access_token_url("222"), json={"token": "ghs_a"})
        requests_mock.post(_access_token_url("444"), json={"token": "ghs_b"})
        requests_mock.get(
            "https://api.github.com/rate_limit",
            [
                {"json": {"resources": {"core": {"remaining": first_remaining, "reset": 4070908800}}}},
                {"json": {"resources": {"core": {"remaining": 5000, "reset": 4070908800}}}},
            ],
        )
        return _authenticator(_github_apps_field(("111", "222", FAKE_PEM), ("333", "444", FAKE_PEM)))

    def test_seeding_logs_each_installation(self, requests_mock, caplog):
        caplog.set_level(logging.INFO, logger="airbyte")
        _sign(self._two_installations(requests_mock))

        seeded = [r.getMessage() for r in caplog.records if "quota seeded" in r.getMessage()]
        assert len(seeded) == 2
        assert "app_id=111 installation_id=222 remaining=5000" in seeded[0]
        assert "app_id=333 installation_id=444 remaining=5000" in seeded[1]

    def test_periodic_summary_lists_every_installation(self, requests_mock, caplog, monkeypatch):
        monkeypatch.setattr("components._QUOTA_LOG_EVERY_CALLS", 2)
        caplog.set_level(logging.INFO, logger="airbyte")
        authenticator = self._two_installations(requests_mock)

        _sign(authenticator)
        _sign(authenticator)

        summaries = [r.getMessage() for r in caplog.records if "quota after" in r.getMessage()]
        assert summaries == [
            "github_app_auth: quota after 2 calls: app_id=111 installation_id=222 remaining=4998 reset_at=2099-01-01T00:00:00Z; "
            "app_id=333 installation_id=444 remaining=5000 reset_at=2099-01-01T00:00:00Z"
        ]

    def test_rotation_is_logged(self, requests_mock, caplog):
        caplog.set_level(logging.INFO, logger="airbyte")
        authenticator = self._two_installations(requests_mock, first_remaining=50)

        assert _sign(authenticator) == "token ghs_b"
        assert any("app_id=111 installation_id=222 remaining=50" in r.getMessage() and "rotating" in r.getMessage() for r in caplog.records)


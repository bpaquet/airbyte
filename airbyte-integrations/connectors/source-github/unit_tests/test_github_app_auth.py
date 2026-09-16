#
# Copyright (c) 2026 Airbyte, Inc., all rights reserved.
#

import threading

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
from airbyte_cdk.sources.streams.http.requests_native_auth.protocols import (
    ResponseAwareAuthenticator,
    TokenRotatingAuthenticator,
)
from airbyte_cdk.utils import AirbyteTracedException
from source_github import SourceGithub
from source_github.github_app_auth import GithubAppMultiPemAuthenticator, _parse_entries


FAKE_PEM = (
    "-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJBAKj34GkxFhD90vcNLYLInFEX6Ppy1tPf9Cnzj4p4WGeKLs1Pt8Qu\n-----END RSA PRIVATE KEY-----"
)


def _github_apps_field(*entries):
    return "\n\n".join(f"{app_id}\n{installation_id}\n{pem}" for app_id, installation_id, pem in entries)


def _access_token_url(installation_id):
    return f"https://api.github.com/app/installations/{installation_id}/access_tokens"


@pytest.fixture(autouse=True)
def _mock_jwt(monkeypatch):
    # Signing a real JWT needs a real RSA key; we only test our own usage of PyJWT, not PyJWT
    # itself, so stub it out with a deterministic fake.
    monkeypatch.setattr("source_github.github_app_auth.jwt.encode", lambda payload, key, algorithm: "fake-app-jwt")


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
        with pytest.raises(ValueError, match="missing installation_id"):
            _parse_entries("111")

    def test_missing_pem_raises(self):
        with pytest.raises(ValueError, match="expected a '-----BEGIN"):
            _parse_entries("111\n222\nnot a pem")

    def test_unterminated_pem_raises(self):
        with pytest.raises(ValueError, match="never reached a '-----END"):
            _parse_entries("111\n222\n-----BEGIN RSA PRIVATE KEY-----\nAAAA")


class TestInstallationTokenMintingAndCaching:
    def test_mints_once_and_caches(self, requests_mock):
        mint_mock = requests_mock.post(_access_token_url("222"), json={"token": "ghs_abc"})
        requests_mock.get("https://api.github.com/rate_limit", json={"resources": {"core": {"remaining": 5000, "reset": 4070908800}}})
        authenticator = GithubAppMultiPemAuthenticator(config={}, parameters={}, github_apps=_github_apps_field(("111", "222", FAKE_PEM)))
        authenticator._ensure_ready()  # lazy: entries are only parsed/seeded on first actual use
        cache = authenticator._caches[0]
        assert cache.get_token() == "ghs_abc"
        assert cache.get_token() == "ghs_abc"
        assert mint_mock.call_count == 1  # _ensure_ready()'s refresh_quota() already minted once; both get_token() calls hit the cache

    def test_refreshes_after_expiry(self, requests_mock):
        requests_mock.post(_access_token_url("222"), [{"json": {"token": "ghs_first"}}, {"json": {"token": "ghs_second"}}])
        requests_mock.get("https://api.github.com/rate_limit", json={"resources": {"core": {"remaining": 5000, "reset": 4070908800}}})
        with freeze_time("2026-01-01T00:00:00Z") as frozen:
            authenticator = GithubAppMultiPemAuthenticator(
                config={}, parameters={}, github_apps=_github_apps_field(("111", "222", FAKE_PEM))
            )
            authenticator._ensure_ready()
            cache = authenticator._caches[0]
            assert cache.get_token() == "ghs_first"
            frozen.tick(delta=__import__("datetime").timedelta(hours=1, minutes=1))
            assert cache.get_token() == "ghs_second"

    def test_bad_credentials_raise_traced_config_error(self, requests_mock):
        requests_mock.post(_access_token_url("222"), status_code=401, json={"message": "Bad credentials"})
        authenticator = GithubAppMultiPemAuthenticator(config={}, parameters={}, github_apps=_github_apps_field(("111", "222", FAKE_PEM)))
        with pytest.raises(AirbyteTracedException) as exc_info:
            authenticator.token  # construction itself must never raise — see test_manifest_selective_authenticator_builds_in_token_mode
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
        authenticator = GithubAppMultiPemAuthenticator(
            config={}, parameters={}, github_apps=_github_apps_field(("111", "222", FAKE_PEM), ("333", "444", FAKE_PEM))
        )
        assert authenticator.token == "token ghs_a"
        assert authenticator.token == "token ghs_a"
        assert authenticator.token == "token ghs_a"

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
        authenticator = GithubAppMultiPemAuthenticator(
            config={}, parameters={}, github_apps=_github_apps_field(("111", "222", FAKE_PEM), ("333", "444", FAKE_PEM))
        )
        assert authenticator.token == "token ghs_a"  # remaining 51 -> 50, still above reserve check next time? no: 50 is not > 50
        assert authenticator.token == "token ghs_b"  # cache 0 now at 50, not > _BUDGET_MIN_RESERVE (50), rotates

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
        authenticator = GithubAppMultiPemAuthenticator(
            config={},
            parameters={},
            github_apps=_github_apps_field(
                ("111", "222", FAKE_PEM), ("333", "444", FAKE_PEM), ("555", "666", FAKE_PEM), ("777", "888", FAKE_PEM)
            ),
        )
        assert [authenticator.token for _ in range(5)] == [
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
                {"json": {"resources": {"core": {"remaining": 0, "reset": 1000}}}},
                {"json": {"resources": {"core": {"remaining": 0, "reset": 2000}}}},
                {"json": {"resources": {"core": {"remaining": 5000, "reset": 4070908800}}}},
                {"json": {"resources": {"core": {"remaining": 5000, "reset": 4070908800}}}},
            ],
        )
        sleeps = []
        monkeypatch.setattr("source_github.github_app_auth.time.sleep", lambda seconds: sleeps.append(seconds))
        monkeypatch.setattr("source_github.github_app_auth.time.time", lambda: 999.0)
        authenticator = GithubAppMultiPemAuthenticator(
            config={}, parameters={}, github_apps=_github_apps_field(("111", "222", FAKE_PEM), ("333", "444", FAKE_PEM))
        )
        token = authenticator.token
        assert token in ("token ghs_a", "token ghs_b")
        assert sleeps == [1.0]  # earliest reset (1000) minus frozen "now" (999)


class TestSourceGithubIntegration:
    def _config(self, github_apps_field):
        return {"credentials": {"github_apps": github_apps_field}, "repositories": ["org/repo"]}

    def test_get_access_token_returns_github_app_title(self):
        config = self._config(_github_apps_field(("111", "222", FAKE_PEM)))
        title, value = SourceGithub.get_access_token(config)
        assert title == "GitHub App"
        assert value == config["credentials"]["github_apps"]

    def _dummy_source(self):
        # `_ensure_auth_mode` doesn't touch `self`, but the constructor still needs a valid
        # baseline config (same pattern as the other tests' `_source_and_authenticator` helper).
        return SourceGithub(catalog=None, config={"access_token": "x", "repositories": ["org/repo"]}, state=None)

    def test_ensure_auth_mode_github_apps(self):
        source = self._dummy_source()
        config = self._config(_github_apps_field(("111", "222", FAKE_PEM)))
        transformed = source._ensure_auth_mode(config)
        assert transformed["credentials"]["auth_mode"] == "github_apps"

    def test_ensure_auth_mode_token_for_pat(self):
        source = self._dummy_source()
        config = {"credentials": {"personal_access_token": "pat-token"}}
        transformed = source._ensure_auth_mode(config)
        assert transformed["credentials"]["auth_mode"] == "token"

    def test_ensure_auth_mode_token_for_oauth(self):
        source = self._dummy_source()
        config = {"credentials": {"access_token": "oauth-token", "client_id": "id", "client_secret": "secret"}}
        transformed = source._ensure_auth_mode(config)
        assert transformed["credentials"]["auth_mode"] == "token"

    def test_ensure_auth_mode_token_for_legacy_root_access_token(self):
        source = self._dummy_source()
        config = {"access_token": "legacy-token"}
        transformed = source._ensure_auth_mode(config)
        assert transformed["credentials"]["auth_mode"] == "token"

    def test_get_authenticator_returns_github_app_authenticator(self, requests_mock):
        requests_mock.post(_access_token_url("222"), json={"token": "ghs_a"})
        requests_mock.get("https://api.github.com/rate_limit", json={"resources": {"core": {"remaining": 5000, "reset": 4070908800}}})
        config = self._config(_github_apps_field(("111", "222", FAKE_PEM)))
        config["credentials"]["auth_mode"] = "github_apps"
        source = SourceGithub(catalog=None, config=config, state=None)
        authenticator = source._get_authenticator(config)
        assert isinstance(authenticator, GithubAppMultiPemAuthenticator)

    def test_get_authenticator_pat_path_unaffected(self, rate_limit_mock_response):
        config = {"access_token": "pat-token", "repositories": ["org/repo"]}
        source = SourceGithub(catalog=None, config=config, state=None)
        config = source._validate_and_transform_config(config)
        authenticator = source._get_authenticator(config)
        assert isinstance(authenticator, RateLimitedMultipleTokenAuthenticator)

    def test_manifest_selective_authenticator_builds_in_github_apps_mode(self, requests_mock):
        """Regression test: `ModelToComponentFactory.create_selective_authenticator` builds every
        branch under `authenticators:` eagerly, not just the selected one. `_get_authenticator`
        bypasses that entirely (it resolves the "token"/"github_apps" branch directly), so this
        is the only test that actually exercises the manifest's `SelectiveAuthenticator` wiring
        the way `check_connection`/`streams()` do — it alone would have caught the
        "Authentication tokens are missing from the configuration" crash from the token branch's
        RateLimitedMultipleTokenAuthenticator being constructed with an empty token list.
        """
        requests_mock.post(_access_token_url("222"), json={"token": "ghs_a"})
        requests_mock.get("https://api.github.com/rate_limit", json={"resources": {"core": {"remaining": 5000, "reset": 4070908800}}})
        config = self._config(_github_apps_field(("111", "222", FAKE_PEM)))
        source = SourceGithub(catalog=None, config=config, state=None)
        transformed = source._validate_and_transform_config(config)
        authenticator = source._constructor.create_component(
            model_type=SelectiveAuthenticatorModel,
            component_definition=source.resolved_manifest["definitions"]["requester_base"]["authenticator"],
            config=transformed,
        )
        assert isinstance(authenticator, GithubAppMultiPemAuthenticator)

    def test_manifest_selective_authenticator_builds_in_token_mode(self, rate_limit_mock_response):
        """Symmetric regression test: in "token" mode, `credentials.github_apps` is empty, but
        the "github_apps" branch is still constructed eagerly alongside the selected "token"
        branch. GithubAppMultiPemAuthenticator must not validate/raise at construction time — only
        lazily, on first actual use — or every PAT/OAuth user would crash on an empty
        `github_apps` the same way github_apps-mode users crashed on an empty `tokens` list.
        """
        config = {"access_token": "pat-token", "repositories": ["org/repo"]}
        source = SourceGithub(catalog=None, config=config, state=None)
        transformed = source._validate_and_transform_config(config)
        authenticator = source._constructor.create_component(
            model_type=SelectiveAuthenticatorModel,
            component_definition=source.resolved_manifest["definitions"]["requester_base"]["authenticator"],
            config=transformed,
        )
        assert isinstance(authenticator, RateLimitedMultipleTokenAuthenticator)

    def test_manifest_selective_authenticator_builds_in_oauth_mode(self, rate_limit_mock_response):
        """Same regression as above, but for the OAuth credentials shape specifically
        (credentials.access_token + client_id/client_secret) rather than the legacy root-level
        access_token — a different code path through get_access_token/_ensure_auth_mode.
        """
        config = {
            "credentials": {"access_token": "oauth-token", "client_id": "id", "client_secret": "secret"},
            "repositories": ["org/repo"],
        }
        source = SourceGithub(catalog=None, config=config, state=None)
        transformed = source._validate_and_transform_config(config)
        authenticator = source._constructor.create_component(
            model_type=SelectiveAuthenticatorModel,
            component_definition=source.resolved_manifest["definitions"]["requester_base"]["authenticator"],
            config=transformed,
        )
        assert isinstance(authenticator, RateLimitedMultipleTokenAuthenticator)


def _prepared_request_with_token(token):
    request = requests.Request("GET", "https://api.github.com/repos/org/repo/actions/runs").prepare()
    request.headers["Authorization"] = f"token {token}"
    return request


def _response(status_code=200, remaining=None, reset_at=None, limit=None, retry_after=None):
    response = requests.Response()
    response.status_code = status_code
    if remaining is not None:
        response.headers["X-RateLimit-Remaining"] = str(remaining)
    if reset_at is not None:
        response.headers["X-RateLimit-Reset"] = str(int(reset_at))
    if limit is not None:
        response.headers["X-RateLimit-Limit"] = str(limit)
    if retry_after is not None:
        response.headers["Retry-After"] = str(retry_after)
    return response


def _two_app_authenticator(requests_mock, remaining_a=500, remaining_b=500, reset_at=4070908800):
    requests_mock.post(_access_token_url("222"), json={"token": "ghs_a"})
    requests_mock.post(_access_token_url("444"), json={"token": "ghs_b"})
    requests_mock.get(
        "https://api.github.com/rate_limit",
        [
            {"json": {"resources": {"core": {"remaining": remaining_a, "reset": reset_at}}}},
            {"json": {"resources": {"core": {"remaining": remaining_b, "reset": reset_at}}}},
        ],
    )
    authenticator = GithubAppMultiPemAuthenticator(
        config={}, parameters={}, github_apps=_github_apps_field(("111", "222", FAKE_PEM), ("333", "444", FAKE_PEM))
    )
    assert authenticator.token == "token ghs_a"  # forces _ensure_ready() so both caches exist
    return authenticator


class TestProtocolMembership:
    def test_implements_response_aware_and_token_rotating_protocols(self):
        """`HttpClient` dispatches to these structurally (isinstance against a
        `runtime_checkable` Protocol), not via a declared base class — confirm a real instance
        actually satisfies both, since a typo'd method name would silently opt the connector
        back into local-counter-only drift with no error anywhere."""
        authenticator = GithubAppMultiPemAuthenticator.__new__(GithubAppMultiPemAuthenticator)
        authenticator.__post_init__({})
        assert isinstance(authenticator, ResponseAwareAuthenticator)
        assert isinstance(authenticator, TokenRotatingAuthenticator)


class TestResponseAwareRotation:
    def test_explicit_zero_on_sending_cache_enables_rotation_without_a_refresh_quota_poll(self, requests_mock):
        """This is the exact incident this class exists to prevent: GitHub genuinely exhausts
        app 1 (a real `X-RateLimit-Remaining: 0`), but app 1's locally tracked `remaining` was
        seeded high and never independently re-polled. Without `update_from_response` reconciling
        the real header, `has_alternative_token` would keep seeing app 1 as healthy and the
        connector would retry the same dead app until the retry budget ran out — which is what
        happened in production. Only 2 `/rate_limit` calls are mocked (the initial seed for each
        app); a 3rd call would error, proving rotation here does not depend on re-polling it."""
        authenticator = _two_app_authenticator(requests_mock)
        request = _prepared_request_with_token("ghs_a")

        authenticator.update_from_response(request, _response(status_code=403, remaining=0, reset_at=4070908800))

        assert authenticator.has_alternative_token(request) is True
        assert authenticator.token == "token ghs_b"

    def test_secondary_rate_limit_does_not_zero_the_pool_or_report_an_alternative(self, requests_mock):
        """A secondary/abuse-detection rejection (`Retry-After`, no quota headers) is a
        different, usually shared-across-credentials limit — not this cache's primary quota
        being spent. Zeroing it here would rotate onto another app about to hit the exact same
        secondary rejection, burning a retry for nothing."""
        authenticator = _two_app_authenticator(requests_mock)
        request = _prepared_request_with_token("ghs_a")
        cache_a = authenticator._caches[0]
        before = cache_a.remaining

        authenticator.update_from_response(request, _response(status_code=403, retry_after=120))

        assert cache_a.remaining == before
        assert authenticator.has_alternative_token(request) is False

    def test_permission_403_with_healthy_quota_does_not_park_the_cache(self, requests_mock):
        authenticator = _two_app_authenticator(requests_mock)
        request = _prepared_request_with_token("ghs_a")

        authenticator.update_from_response(request, _response(status_code=403, remaining=487, reset_at=4070908800))

        assert authenticator._caches[0].remaining == 487
        assert authenticator.has_alternative_token(request) is False

    def test_single_app_reports_no_alternative(self, requests_mock):
        requests_mock.post(_access_token_url("222"), json={"token": "ghs_a"})
        requests_mock.get("https://api.github.com/rate_limit", json={"resources": {"core": {"remaining": 500, "reset": 4070908800}}})
        authenticator = GithubAppMultiPemAuthenticator(config={}, parameters={}, github_apps=_github_apps_field(("111", "222", FAKE_PEM)))
        request = _prepared_request_with_token(authenticator.token.split(" ")[1])

        authenticator.update_from_response(request, _response(status_code=403, remaining=0, reset_at=4070908800))

        assert authenticator.has_alternative_token(request) is False

    def test_unknown_token_is_a_no_op(self, requests_mock):
        """A response for a request this authenticator never signed (or from before a cache
        re-minted its token) must not raise and must not touch any cache's state."""
        authenticator = _two_app_authenticator(requests_mock)
        before = [cache.remaining for cache in authenticator._caches]
        request = _prepared_request_with_token("some-other-tokens-value")

        authenticator.update_from_response(request, _response(status_code=403, remaining=0, reset_at=4070908800))

        assert [cache.remaining for cache in authenticator._caches] == before
        assert authenticator.has_alternative_token(request) is False

    def test_cached_response_is_ignored(self, requests_mock):
        """A replayed cache hit carries the rate-limit headers from whenever it was first
        fetched and consumed no quota of its own; reconciling against it would be wrong."""
        authenticator = _two_app_authenticator(requests_mock)
        request = _prepared_request_with_token("ghs_a")
        before = authenticator._caches[0].remaining
        response = _response(status_code=403, remaining=0, reset_at=4070908800)
        response.from_cache = True

        authenticator.update_from_response(request, response)

        assert authenticator._caches[0].remaining == before

    def test_out_of_order_response_never_increases_remaining_within_the_same_window(self, requests_mock):
        authenticator = _two_app_authenticator(requests_mock)
        request = _prepared_request_with_token("ghs_a")

        authenticator.update_from_response(request, _response(status_code=200, remaining=80, reset_at=4070908800))
        assert authenticator._caches[0].remaining == 80

        # A slower, out-of-order response reporting a higher count from the same window must
        # not resurrect the counter another (faster) concurrent request already brought down.
        authenticator.update_from_response(request, _response(status_code=200, remaining=90, reset_at=4070908800))
        assert authenticator._caches[0].remaining == 80

    def test_window_rollover_replaces_the_counter_wholesale(self, requests_mock):
        authenticator = _two_app_authenticator(requests_mock)
        request = _prepared_request_with_token("ghs_a")
        authenticator.update_from_response(request, _response(status_code=200, remaining=10, reset_at=4070908800))
        assert authenticator._caches[0].remaining == 10

        # A response with a newer reset than what's locally tracked means a fresh window arrived;
        # take its numbers wholesale even though 5000 > 10 would otherwise look like a regression.
        authenticator.update_from_response(request, _response(status_code=200, remaining=5000, reset_at=4070908801))
        assert authenticator._caches[0].remaining == 5000
        assert authenticator._caches[0].reset_at == 4070908801

    def test_explicit_zero_overrides_out_of_order_ordering(self, requests_mock):
        """The one case an exhaustion signal must win even though its `reset_at` looks stale
        relative to what is locally tracked: a real, explicit zero must never be dropped, or a
        rate limit whose reset header trails the value already held would silently break
        rotation."""
        authenticator = _two_app_authenticator(requests_mock)
        request = _prepared_request_with_token("ghs_a")
        authenticator.update_from_response(request, _response(status_code=200, remaining=200, reset_at=4070908800))

        authenticator.update_from_response(request, _response(status_code=403, remaining=0, reset_at=4070908700))

        assert authenticator._caches[0].remaining == 0

    def test_concurrent_updates_from_different_apps_do_not_corrupt_state(self, requests_mock):
        authenticator = _two_app_authenticator(requests_mock)
        request_a = _prepared_request_with_token("ghs_a")
        request_b = _prepared_request_with_token("ghs_b")
        errors = []

        def hammer(request, remaining):
            try:
                for _ in range(200):
                    authenticator.update_from_response(request, _response(status_code=200, remaining=remaining, reset_at=4070908800))
            except Exception as e:  # pragma: no cover - failure path only
                errors.append(e)

        threads = [
            threading.Thread(target=hammer, args=(request_a, 300)),
            threading.Thread(target=hammer, args=(request_b, 400)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert authenticator._caches[0].remaining == 300
        assert authenticator._caches[1].remaining == 400

    def test_end_to_end_rate_limited_response_rotates_without_sleeping_out_the_window(self, requests_mock, monkeypatch):
        """Reproduces the production incident through the real CDK `HttpClient`: the first app's
        installation token gets a 403 with `X-RateLimit-Remaining: 0`. Before this fix, the
        authenticator's stale local counter would report no alternative, `HttpClient` would sleep
        out the ~1h reset window, and the retry would hit the same exhausted app again. With
        `update_from_response`/`has_alternative_token` wired in, the retry must go out on app 2's
        token immediately, and no sleep of the reset-window's magnitude should ever be requested.
        """
        import logging

        from airbyte_cdk.sources.streams.http import HttpClient
        from airbyte_cdk.sources.streams.http.error_handlers import HttpStatusErrorHandler
        from airbyte_cdk.sources.streams.http.error_handlers.response_models import ErrorResolution, ResponseAction

        authenticator = _two_app_authenticator(requests_mock)

        sleeps = []
        monkeypatch.setattr("time.sleep", lambda seconds: sleeps.append(seconds))

        requests_mock.get(
            "https://api.github.com/repos/org/repo/actions/runs",
            [
                {"status_code": 403, "headers": {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "4070908800"}, "json": {}},
                {"status_code": 200, "json": {"workflow_runs": []}},
            ],
        )

        error_handler = HttpStatusErrorHandler(
            logger=logging.getLogger("airbyte"),
            error_mapping={403: ErrorResolution(response_action=ResponseAction.RATE_LIMITED, failure_type=None)},
            max_retries=3,
        )
        client = HttpClient(name="test", logger=logging.getLogger("airbyte"), error_handler=error_handler, authenticator=authenticator)
        request, response = client.send_request(
            http_method="GET", url="https://api.github.com/repos/org/repo/actions/runs", request_kwargs={}
        )

        assert response.status_code == 200
        history = [r for r in requests_mock.request_history if r.url == "https://api.github.com/repos/org/repo/actions/runs"]
        # Exactly one retry happened (the 403 was recovered from, not treated as fatal).
        # `request_history` isn't used to check *which* token each attempt carried: the CDK
        # reuses and re-signs the same `PreparedRequest` object across retries, so by the time
        # the history is inspected here every entry aliases the same, now-final-token object —
        # a `requests_mock` artifact, not evidence about what was actually sent on the wire.
        # `_caches` state (updated synchronously as each real response comes in) is the reliable
        # signal instead: app 1 must have been marked exhausted and app 2 must be the one that
        # ultimately went out and is now itself slightly spent.
        assert len(history) == 2
        assert authenticator._caches[0].remaining == 0
        assert authenticator._caches[1].remaining is not None and 0 < authenticator._caches[1].remaining < 500
        # The only sleeps observed must be tiny/incidental (e.g. jittered retry backoff), never
        # anything near the reset window (which would be ~3600s in a real incident).
        assert all(s < 5 for s in sleeps), f"a sleep near the reset-window magnitude was requested: {sleeps}"

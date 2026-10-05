#
# Copyright (c) 2026 Airbyte, Inc., all rights reserved.
#

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
from components import GithubAppMultiPemAuthenticator, _InstallationTokenCache, _parse_entries

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


def _selective_authenticator(config):
    """Builds the manifest's actual `requester_base.authenticator` (a `SelectiveAuthenticator`)
    the way `check`/`streams()` do, rather than reaching for `GithubAppMultiPemAuthenticator`
    directly — the regression that matters is the manifest wiring, not just the class."""
    source = make_source(catalog=None, config=config, state=None)
    return source._constructor.create_component(
        model_type=SelectiveAuthenticatorModel,
        component_definition=source.resolved_manifest["definitions"]["requester_base"]["authenticator"],
        config=source._config,
    )


@pytest.fixture(autouse=True)
def _mock_jwt(monkeypatch):
    # Signing a real JWT needs a real RSA key; we only test our own usage of PyJWT, not PyJWT
    # itself, so stub it out with a deterministic fake.
    monkeypatch.setattr("components.jwt.encode", lambda payload, key, algorithm: "fake-app-jwt")


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
                {"json": {"resources": {"core": {"remaining": 0, "reset": 1030}}}},
                {"json": {"resources": {"core": {"remaining": 0, "reset": 2000}}}},
                {"json": {"resources": {"core": {"remaining": 5000, "reset": 4070908800}}}},
                {"json": {"resources": {"core": {"remaining": 5000, "reset": 4070908800}}}},
            ],
        )
        sleeps = []
        monkeypatch.setattr("components.time.sleep", lambda seconds: sleeps.append(seconds))
        monkeypatch.setattr("components.time.time", lambda: 999.0)
        authenticator = GithubAppMultiPemAuthenticator(
            config={}, parameters={}, github_apps=_github_apps_field(("111", "222", FAKE_PEM), ("333", "444", FAKE_PEM))
        )
        token = authenticator.token
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
        authenticator = GithubAppMultiPemAuthenticator(config={}, parameters={}, github_apps=_github_apps_field(("111", "222", FAKE_PEM)))
        authenticator.token
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
        requests_mock.get("https://api.github.com/rate_limit", json={"resources": {"core": {"remaining": seeded, "reset": 4070908800}}})
        authenticator = GithubAppMultiPemAuthenticator(config={}, parameters={}, github_apps=_github_apps_field(("111", "222", FAKE_PEM)))

        barrier = threading.Barrier(n_threads)

        def worker():
            barrier.wait()
            for _ in range(calls_per_thread):
                authenticator.token

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert authenticator._caches[0].remaining == seeded - total_calls

    def test_concurrent_first_use_seeds_quota_once(self, requests_mock):
        """Regression test for double-checked locking in `_ensure_ready`: concurrent first calls
        to `.token` must parse `github_apps` and seed quota exactly once, not once per thread.
        """
        requests_mock.post(_access_token_url("222"), json={"token": "ghs_a"})
        rate_limit_mock = requests_mock.get(
            "https://api.github.com/rate_limit", json={"resources": {"core": {"remaining": 5000, "reset": 4070908800}}}
        )
        authenticator = GithubAppMultiPemAuthenticator(config={}, parameters={}, github_apps=_github_apps_field(("111", "222", FAKE_PEM)))

        n_threads = 10
        barrier = threading.Barrier(n_threads)

        def worker():
            barrier.wait()
            authenticator.token

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert rate_limit_mock.call_count == 1


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
        authenticator = GithubAppMultiPemAuthenticator(
            config={"api_url": "https://github.company.org/api/v3/"},
            parameters={},
            github_apps=_github_apps_field(("111", "222", FAKE_PEM)),
        )
        assert authenticator.token == "token ghs_a"
        assert ghes_token_mock.called
        assert ghes_quota_mock.called

    def test_defaults_to_public_github_when_api_url_is_absent(self, requests_mock):
        requests_mock.post(_access_token_url("222"), json={"token": "ghs_a"})
        requests_mock.get("https://api.github.com/rate_limit", json={"resources": {"core": {"remaining": 5000, "reset": 4070908800}}})
        authenticator = GithubAppMultiPemAuthenticator(config={}, parameters={}, github_apps=_github_apps_field(("111", "222", FAKE_PEM)))
        assert authenticator.token == "token ghs_a"


class TestResponseAwareReconciliation:
    def _prepared_request(self, token):
        request = requests.Request("GET", "https://api.github.com/repos/org/repo").prepare()
        request.headers["Authorization"] = f"token {token}"
        return request

    def _response(self, status_code=200, remaining=None, reset=None):
        response = requests.Response()
        response.status_code = status_code
        if remaining is not None:
            response.headers["X-RateLimit-Remaining"] = str(remaining)
        if reset is not None:
            response.headers["X-RateLimit-Reset"] = str(reset)
        return response

    def test_update_from_response_tightens_the_local_estimate(self, requests_mock):
        requests_mock.post(_access_token_url("222"), json={"token": "ghs_a"})
        requests_mock.get("https://api.github.com/rate_limit", json={"resources": {"core": {"remaining": 5000, "reset": 4070908800}}})
        authenticator = GithubAppMultiPemAuthenticator(config={}, parameters={}, github_apps=_github_apps_field(("111", "222", FAKE_PEM)))
        authenticator._ensure_ready()
        cache = authenticator._caches[0]

        authenticator.update_from_response(self._prepared_request("ghs_a"), self._response(remaining=42, reset=4070908800))

        assert cache.remaining == 42

    def test_update_from_response_ignores_an_unrelated_token(self, requests_mock):
        requests_mock.post(_access_token_url("222"), json={"token": "ghs_a"})
        requests_mock.get("https://api.github.com/rate_limit", json={"resources": {"core": {"remaining": 5000, "reset": 4070908800}}})
        authenticator = GithubAppMultiPemAuthenticator(config={}, parameters={}, github_apps=_github_apps_field(("111", "222", FAKE_PEM)))
        authenticator._ensure_ready()
        cache = authenticator._caches[0]

        authenticator.update_from_response(self._prepared_request("some-other-token"), self._response(remaining=0, reset=4070908800))

        assert cache.remaining == 5000

    def test_update_from_response_adopts_a_fresh_window(self, requests_mock):
        """A later reset is a new window; the response's count describes it, not a stale local one."""
        requests_mock.post(_access_token_url("222"), json={"token": "ghs_a"})
        requests_mock.get("https://api.github.com/rate_limit", json={"resources": {"core": {"remaining": 1, "reset": 1000}}})
        authenticator = GithubAppMultiPemAuthenticator(config={}, parameters={}, github_apps=_github_apps_field(("111", "222", FAKE_PEM)))
        authenticator._ensure_ready()
        cache = authenticator._caches[0]

        authenticator.update_from_response(self._prepared_request("ghs_a"), self._response(remaining=5000, reset=2000))

        assert (cache.remaining, cache.reset_at) == (5000, 2000)

    def test_has_alternative_token_true_when_sender_exhausted_and_another_has_quota(self, requests_mock):
        requests_mock.post(_access_token_url("222"), json={"token": "ghs_a"})
        requests_mock.post(_access_token_url("444"), json={"token": "ghs_b"})
        requests_mock.get(
            "https://api.github.com/rate_limit",
            [
                {"json": {"resources": {"core": {"remaining": 5000, "reset": 4070908800}}}},
                {"json": {"resources": {"core": {"remaining": 5000, "reset": 4070908800}}}},
            ],
        )
        authenticator = GithubAppMultiPemAuthenticator(
            config={}, parameters={}, github_apps=_github_apps_field(("111", "222", FAKE_PEM), ("333", "444", FAKE_PEM))
        )
        authenticator._ensure_ready()
        authenticator._caches[0].remaining = 0

        assert authenticator.has_alternative_token(self._prepared_request("ghs_a")) is True

    def test_has_alternative_token_false_when_only_one_installation(self, requests_mock):
        requests_mock.post(_access_token_url("222"), json={"token": "ghs_a"})
        requests_mock.get("https://api.github.com/rate_limit", json={"resources": {"core": {"remaining": 0, "reset": 4070908800}}})
        authenticator = GithubAppMultiPemAuthenticator(config={}, parameters={}, github_apps=_github_apps_field(("111", "222", FAKE_PEM)))
        authenticator._ensure_ready()

        assert authenticator.has_alternative_token(self._prepared_request("ghs_a")) is False


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

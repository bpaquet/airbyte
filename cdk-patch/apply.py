"""Lets RATE_LIMITED responses that carry a computed wait retry until the limit resets, instead of
failing after the error handler's max_retries/max_time. Each wait stays capped by its strategy."""

import os
import subprocess
import sys
from importlib.metadata import version
from importlib.util import find_spec


EXPECTED = "7.35.0"
if version("airbyte-cdk") != EXPECTED:
    sys.exit(f"patch written for airbyte-cdk {EXPECTED}, found {version('airbyte-cdk')}")

http_dir = os.path.join(find_spec("airbyte_cdk").submodule_search_locations[0], "sources", "streams", "http")


def patch(name, replacements):
    path = os.path.join(http_dir, name)
    text = open(path).read()
    for old, new in replacements:
        if text.count(old) != 1:
            sys.exit(f"{name}: expected exactly one match for {old[:60]!r}")
        text = text.replace(old, new)
    open(path, "w").write(text)


patch(
    "rate_limiting.py",
    [
        (
            "def user_defined_backoff_handler(\n    max_tries: Optional[int], max_time: Optional[int] = None, **kwargs: Any\n)",
            "def user_defined_backoff_handler(\n    max_tries: Optional[int],\n    max_time: Optional[int] = None,\n"
            "    exception: type = UserDefinedBackoffException,\n    **kwargs: Any,\n)",
        ),
        ("        backoff.constant,\n        UserDefinedBackoffException,\n", "        backoff.constant,\n        exception,\n"),
    ],
)

patch(
    "http_client.py",
    [
        ("    RateLimitBackoffException,\n", "    RateLimitBackoffException,\n    RateLimitedUserDefinedBackoffException,\n"),
        (
            "        user_backoff_handler = user_defined_backoff_handler(max_tries=max_tries, max_time=max_time)(\n"
            "            self._send\n        )",
            "        user_backoff_handler = user_defined_backoff_handler(max_tries=max_tries, max_time=max_time)(\n"
            "            user_defined_backoff_handler(\n"
            "                max_tries=None, max_time=None, exception=RateLimitedUserDefinedBackoffException\n"
            "            )(self._send)\n        )",
        ),
        (
            "            if user_defined_backoff_time:\n                raise UserDefinedBackoffException(",
            "            if user_defined_backoff_time:\n"
            "                exception_class = (\n"
            "                    RateLimitedUserDefinedBackoffException\n"
            "                    if retry_endlessly\n"
            "                    else UserDefinedBackoffException\n"
            "                )\n"
            "                raise exception_class(",
        ),
    ],
)

with open(os.path.join(http_dir, "exceptions.py"), "a") as f:
    f.write(
        "\n\nclass RateLimitedUserDefinedBackoffException(UserDefinedBackoffException):\n"
        '    """A rate limit with a computed wait, retried without a try or time limit."""\n'
    )

# Fresh interpreter, so the check sees the patched files rather than anything cached.
subprocess.run(
    [sys.executable, "-c", "from airbyte_cdk.sources.streams.http.http_client import RateLimitedUserDefinedBackoffException"],
    check=True,
    cwd="/",
)

print("airbyte-cdk rate-limit patch applied")

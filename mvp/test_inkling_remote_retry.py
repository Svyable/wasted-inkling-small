#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import io
import unittest
import urllib.error
from datetime import datetime, timedelta, timezone
from email.message import Message

from inkling_remote_fixture import (
    HttpRangeSource,
    RemoteFixtureError,
    _retry_after_seconds,
)


class _Response(io.BytesIO):
    def __init__(self, payload: bytes = b"ok") -> None:
        super().__init__(payload)
        self.headers = Message()
        self.status = 200


class _Opener:
    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)

    def open(self, _request, timeout=None):
        if not self.outcomes:
            raise AssertionError("unexpected extra request")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _http_error(code: int, retry_after: str | None = None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(
        "https://example.invalid/model",
        code,
        "synthetic",
        headers,
        io.BytesIO(),
    )


class RemoteRetryTest(unittest.TestCase):
    def source(self, outcomes, *, retries=4, maximum=60.0):
        delays = []
        source = HttpRangeSource(
            "https://example.invalid/model",
            "abcdef0",
            retries=retries,
            retry_backoff=1.0,
            max_retry_delay=maximum,
            sleep=delays.append,
        )
        source.opener = _Opener(outcomes)
        return source, delays

    def test_honors_retry_after_then_succeeds(self):
        response = _Response()
        source, delays = self.source([_http_error(429, "3"), response])
        self.assertIs(source.open("config.json"), response)
        self.assertEqual(delays, [3.0])
        self.assertEqual(source.requests, 2)

    def test_transient_errors_use_exponential_backoff(self):
        response = _Response()
        source, delays = self.source(
            [_http_error(503, "nonsense"), urllib.error.URLError("reset"), response]
        )
        self.assertIs(source.open("config.json"), response)
        self.assertEqual(delays, [1.0, 2.0])
        self.assertEqual(source.requests, 3)

    def test_delay_is_capped(self):
        response = _Response()
        source, delays = self.source(
            [_http_error(429, "9999"), response], maximum=5.0
        )
        self.assertIs(source.open("config.json"), response)
        self.assertEqual(delays, [5.0])

    def test_permanent_http_error_is_not_retried(self):
        source, delays = self.source([_http_error(404)], retries=6)
        with self.assertRaisesRegex(RemoteFixtureError, "after 1 attempt"):
            source.open("config.json")
        self.assertEqual(delays, [])
        self.assertEqual(source.requests, 1)

    def test_http_date_retry_after_is_parsed(self):
        now = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)
        future = now + timedelta(seconds=17)
        text = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
        self.assertEqual(_retry_after_seconds(text, now=now), 17.0)
        self.assertIsNone(_retry_after_seconds("not-a-date", now=now))


if __name__ == "__main__":
    unittest.main()

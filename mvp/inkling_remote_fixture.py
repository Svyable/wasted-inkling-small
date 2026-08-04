#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build bounded Inkling fixtures with strict ranges and bounded retries.

The validated extraction implementation lives in ``inkling_remote_fixture_core``.
This compatibility wrapper changes only transient HTTP retry behavior and the
Hugging Face endpoint used for small immutable metadata. Permanent errors and
malformed range responses still fail closed in the core reader.
"""
from __future__ import annotations

import argparse
import email.utils
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import inkling_remote_fixture_core as _core
from inkling_remote_fixture_core import *  # noqa: F401,F403

_required_layer_names = _core._required_layer_names
_RETRYABLE_HTTP = frozenset((408, 425, 429, 500, 502, 503, 504))
_RAW_METADATA = frozenset(("config.json", "model.safetensors.index.json"))


def _retry_after_seconds(
    value: str | None,
    *,
    now: datetime | None = None,
) -> float | None:
    """Parse Retry-After seconds or an HTTP date; reject malformed values."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if text.isdigit():
        return float(text)
    try:
        parsed = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    return max(0.0, (parsed - current).total_seconds())


class HttpRanges(_core.HttpRanges):
    """Strict range source with provider-aware, bounded transient retries."""

    def __init__(
        self,
        base: str,
        revision: str,
        *,
        token: str | None = None,
        timeout: float = 60,
        retries: int = 6,
        retry_backoff: float = 1.0,
        max_retry_delay: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(
            base,
            revision,
            token=token,
            timeout=timeout,
            retries=retries,
        )
        _core.require(retry_backoff > 0.0, "retry backoff must be positive")
        _core.require(max_retry_delay > 0.0, "max retry delay must be positive")
        self.retry_backoff = float(retry_backoff)
        self.max_retry_delay = float(max_retry_delay)
        self._sleep = sleep
        parsed = urllib.parse.urlparse(self.base)
        self._raw_metadata = (
            parsed.scheme == "https" and parsed.netloc == "huggingface.co"
        )

    def url(self, name: str) -> str:
        """Use `/raw/` for small Git metadata, `/resolve/` for LFS ranges."""
        _core.require(
            Path(name).name == name and name not in (".", ".."),
            f"unsafe remote path {name!r}",
        )
        if self._raw_metadata and name in _RAW_METADATA:
            revision = urllib.parse.quote(self.revision, safe="")
            filename = urllib.parse.quote(name, safe="")
            return f"{self.base}/raw/{revision}/{filename}"
        return super().url(name)

    def _delay(self, attempt: int, retry_after: str | None = None) -> float:
        exponential = self.retry_backoff * (2**attempt)
        requested = _retry_after_seconds(retry_after) or 0.0
        return min(self.max_retry_delay, max(exponential, requested))

    def open(self, name: str, byte_range: tuple[int, int] | None = None):
        headers = {
            "Accept-Encoding": "identity",
            "User-Agent": "wasted-inkling-range-fixture/2",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if byte_range:
            headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
        request = urllib.request.Request(self.url(name), headers=headers)
        last_error: BaseException | None = None
        for attempt in range(self.retries):
            self.requests += 1
            try:
                return self.opener.open(request, timeout=self.timeout)
            except urllib.error.HTTPError as exc:
                last_error = exc
                retryable = exc.code in _RETRYABLE_HTTP
                if not retryable or attempt + 1 == self.retries:
                    raise _core.RemoteFixtureError(
                        f"HTTP {exc.code} reading {name} after {attempt + 1} attempt(s)"
                    ) from exc
                delay = self._delay(attempt, exc.headers.get("Retry-After"))
                exc.close()
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt + 1 == self.retries:
                    raise _core.RemoteFixtureError(
                        f"cannot read {name} after {attempt + 1} attempt(s): {exc}"
                    ) from exc
                delay = self._delay(attempt)
            self._sleep(delay)
        raise AssertionError(f"unreachable retry loop: {last_error!r}")


HttpRangeSource = HttpRanges


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="thinkingmachines/Inkling-Small")
    ap.add_argument("--revision", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--experts", default="")
    ap.add_argument("--out")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--max-total-gib", type=float, default=8)
    ap.add_argument("--max-entry-gib", type=float, default=2)
    ap.add_argument("--base-url", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    try:
        _core.require(
            _core.MODEL_RE.fullmatch(args.repo),
            f"invalid model id {args.repo!r}",
        )
        base = args.base_url or f"https://huggingface.co/{args.repo}"
        source = HttpRanges(
            base,
            args.revision,
            token=os.environ.get("HF_TOKEN"),
        )
        plan = _core.build_plan(
            source,
            args.repo,
            [int(value) for value in args.layers.split(",") if value],
            _core.parse_experts(args.experts),
            int(args.max_total_gib * (1 << 30)),
            int(args.max_entry_gib * (1 << 30)),
        )
        if args.plan_only:
            result = plan.summary()
        else:
            _core.require(
                args.out,
                "--out is required unless --plan-only is used",
            )
            result = _core.extract(source, plan, args.out)
    except (_core.RemoteFixtureError, ValueError, OSError) as exc:
        ap.error(str(exc))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

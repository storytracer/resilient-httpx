from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx
from tenacity import (
    AsyncRetrying,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
    wait_random,
)


def retry_after_seconds(response: httpx.Response) -> float | None:
    """The seconds a response's ``Retry-After`` header asks for (delta-seconds or an HTTP-date; never below 0), or
    None without a usable one."""
    value = response.headers.get("Retry-After")
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def _default_retry_on_exception() -> list[type[Exception]]:
    return [httpx.TimeoutException, httpx.ConnectError]


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    backoff: str = "exponential"
    min_wait: float = 1.0
    max_wait: float = 30.0
    retry_on: list[int] = field(
        default_factory=lambda: [429, 500, 502, 503, 504],
    )
    retry_on_exception: list[type[Exception]] = field(
        default_factory=_default_retry_on_exception,
    )

    def build_retrying(
        self,
        extra_exceptions: list[type[Exception]] | None = None,
    ) -> AsyncRetrying:
        if self.backoff == "fixed":
            wait = wait_fixed(self.min_wait)
        elif self.backoff == "random_jitter":
            wait = wait_random(min=self.min_wait, max=self.max_wait)
        else:
            wait = wait_exponential(min=self.min_wait, max=self.max_wait)

        exceptions = list(self.retry_on_exception)
        if extra_exceptions:
            exceptions.extend(extra_exceptions)

        return AsyncRetrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait,
            retry=retry_if_exception_type(tuple(exceptions)),
        )

    def build_sync_retrying(
        self,
        extra_exceptions: list[type[Exception]] | None = None,
    ) -> Retrying:
        if self.backoff == "fixed":
            wait = wait_fixed(self.min_wait)
        elif self.backoff == "random_jitter":
            wait = wait_random(min=self.min_wait, max=self.max_wait)
        else:
            wait = wait_exponential(min=self.min_wait, max=self.max_wait)

        exceptions = list(self.retry_on_exception)
        if extra_exceptions:
            exceptions.extend(extra_exceptions)

        return Retrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait,
            retry=retry_if_exception_type(tuple(exceptions)),
        )

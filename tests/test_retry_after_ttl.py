"""blacklist_ttl_from_retry_after: a retryable status (e.g. 429) with a Retry-After shorter than blacklist_ttl
blacklists the proxy for that long; a longer one, none, or the option off keeps blacklist_ttl."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from resilient_httpx import AsyncProxyHttpClient, ProxyHttpClient, RetryPolicy
from resilient_httpx.retry import retry_after_seconds

URL = "https://api.example.com/data"
POLICY = RetryPolicy(max_attempts=2, min_wait=0, max_wait=0)
PROXIES = ["http://proxy1:8080", "http://proxy2:8080"]


def _response(status: int, retry_after: str | None = None) -> httpx.Response:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return httpx.Response(status, text="x", headers=headers, request=httpx.Request("GET", URL))


@contextmanager
def mock_async(side_effect):
    with (
        patch.object(httpx.AsyncClient, "request", AsyncMock(side_effect=side_effect)),
        patch.object(httpx.AsyncClient, "aclose", new_callable=AsyncMock),
    ):
        yield


@contextmanager
def mock_sync(side_effect):
    with patch.object(httpx.Client, "request", MagicMock(side_effect=side_effect)), patch.object(httpx.Client, "close"):
        yield


def _ttl(snapshot: dict, proxy: str) -> float | None:
    entry = next(e for pool in snapshot.values() for e in pool if e["proxy"] == proxy)
    return entry["ttl_remaining_seconds"]


@pytest.mark.parametrize("retry_after, expected", [("60", 60), ("7200", 3600), (None, 3600)])
async def test_a_shorter_retry_after_shortens_the_blacklisting(retry_after, expected):
    with mock_async([_response(429, retry_after), _response(200)]):
        async with AsyncProxyHttpClient(proxies=PROXIES, retry=POLICY, blacklist_threshold=1, blacklist_ttl=3600,
                                        blacklist_ttl_from_retry_after=True) as client:
            assert (await client.get(URL)).status_code == 200
            assert _ttl(client.blacklist_snapshot(), PROXIES[0]) == pytest.approx(expected, abs=2)


async def test_without_the_option_retry_after_is_ignored():
    with mock_async([_response(429, "60"), _response(200)]):
        async with AsyncProxyHttpClient(proxies=PROXIES, retry=POLICY, blacklist_threshold=1,
                                        blacklist_ttl=3600) as client:
            await client.get(URL)
            assert _ttl(client.blacklist_snapshot(), PROXIES[0]) == pytest.approx(3600, abs=2)


async def test_an_http_date_retry_after_counts_too():
    when = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=120), usegmt=True)
    with mock_async([_response(503, when), _response(200)]):
        async with AsyncProxyHttpClient(proxies=PROXIES, retry=POLICY, blacklist_threshold=1, blacklist_ttl=3600,
                                        blacklist_ttl_from_retry_after=True) as client:
            await client.get(URL)
            assert _ttl(client.blacklist_snapshot(), PROXIES[0]) == pytest.approx(120, abs=3)


def test_the_sync_client_shortens_it_the_same_way():
    with (
        mock_sync([_response(429, "30"), _response(200)]),
        ProxyHttpClient(proxies=PROXIES, retry=POLICY, blacklist_threshold=1, blacklist_ttl=3600,
                        blacklist_ttl_from_retry_after=True) as client,
    ):
        assert client.get(URL).status_code == 200
        assert _ttl(client.blacklist_snapshot(), PROXIES[0]) == pytest.approx(30, abs=2)


@pytest.mark.parametrize("value, seconds", [("42", 42.0), ("-5", 0.0), ("soon", None), ("", None)])
def test_retry_after_seconds(value, seconds):
    response = _response(429, value or None)
    assert retry_after_seconds(response) == seconds

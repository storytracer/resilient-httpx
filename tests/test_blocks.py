"""Blocks (a retried status in blacklist_on, an exception in block_exceptions) that do not count as attempts
(blocks_count_as_attempts=False), and blacklist_on_exception: only these retried exceptions count against a proxy."""
import time

import httpx
import pytest

from resilient_httpx import AllProxiesExhausted, AsyncProxyHttpClient, MaxRetriesExceeded, ProxyHttpClient, RetryPolicy
from tests.test_blacklisting import URL, _response, mock_async, mock_sync

PROXIES = ["http://p1:80", "http://p2:80", "http://p3:80", "http://p4:80"]


class Blocked(httpx.TransportError):
    """A block page, raised as a transport error (as linecaster raises an HTML page through a proxy)."""


def _client(cls=AsyncProxyHttpClient, attempts=2, wait=0.0, **kw):
    retry = RetryPolicy(max_attempts=attempts, backoff="fixed", min_wait=wait, max_wait=wait,
                        retry_on=[429, 500, 502, 503, 504], retry_on_exception=[httpx.TransportError])
    return cls(proxies=PROXIES, retry=retry, blacklist_threshold=1, blacklist_ttl=3600, blacklist_on={429}, **kw)


def _left_out(client) -> int:
    return sum(1 for pool in client.blacklist_snapshot().values() for p in pool if not p["in_rotation"])


def _error(cls):
    return cls("x", request=httpx.Request("GET", URL))


async def test_blocks_count_as_attempts_by_default():
    client = _client()
    with mock_async([_response(429), _response(429), _response(200)]):
        with pytest.raises(MaxRetriesExceeded):
            await client.get(URL)
    await client.aclose()


async def test_blocks_do_not_count_as_attempts_nor_wait():
    client = _client(wait=5.0, blocks_count_as_attempts=False)
    started = time.monotonic()
    with mock_async([_response(429), _response(429), _response(429), _response(200)]):
        assert (await client.get(URL)).status_code == 200
    assert time.monotonic() - started < 1 and _left_out(client) == 3       # three proxies blocked, no waits
    await client.aclose()


async def test_other_failures_still_count_with_free_blocks():
    client = _client(blocks_count_as_attempts=False)
    with mock_async([_response(429), _response(500), _response(500), _response(200)]):
        with pytest.raises(MaxRetriesExceeded):                            # two counted attempts, the block free
            await client.get(URL)
    await client.aclose()


async def test_blocks_end_where_the_pool_does():
    client = _client(blocks_count_as_attempts=False)
    with mock_async([_response(429)] * 10):
        with pytest.raises(AllProxiesExhausted):
            await client.get(URL)
    assert _left_out(client) == 4
    await client.aclose()


async def test_a_block_exception_blacklists_and_is_free():
    client = _client(blocks_count_as_attempts=False, blacklist_on_exception=[httpx.ProxyError],
                     block_exceptions=[Blocked])
    with mock_async([_error(Blocked), _error(Blocked), _response(200)]):
        assert (await client.get(URL)).status_code == 200
    assert _left_out(client) == 2
    await client.aclose()


async def test_only_blacklist_on_exception_counts_against_the_proxy():
    client = _client(attempts=3, blacklist_on_exception=[httpx.ProxyError, httpx.ConnectError])
    with mock_async([_error(httpx.ReadTimeout), _error(httpx.ConnectError), _response(200)]):
        assert (await client.get(URL)).status_code == 200
    assert _left_out(client) == 1                                          # the connect error's, not the timeout's
    await client.aclose()


def test_sync_blocks_do_not_count_as_attempts():
    client = _client(ProxyHttpClient, blocks_count_as_attempts=False, blacklist_on_exception=[httpx.ProxyError],
                     block_exceptions=[Blocked])
    with mock_sync([_response(429), _error(Blocked), _error(httpx.ReadTimeout), _response(200)]):
        assert client.get(URL).status_code == 200
    assert _left_out(client) == 2
    client.close()

"""What counts against a proxy, and blocks that do not count as attempts.

A *block* is a proxy turned away by the server: a retried status in ``blacklist_on`` or an exception in
``block_exceptions``, on an attempt through a proxy of a pool. With ``blocks_count_as_attempts=False`` a block does not
use up one of the request's ``max_attempts`` and the next attempt (through the next proxy) goes out without a backoff
wait; each block blacklists its proxy, so the pool bounds how many there can be. ``blacklist_on_exception`` names the
retried exceptions that count against a proxy (e.g. ``httpx.ProxyError``, ``httpx.ConnectError``: the proxy is broken);
the others (a read timeout: the server is slow) are retried through another proxy without blacklisting this one.
``None`` keeps the original rule: every retried exception counts against the proxy.
"""
from __future__ import annotations

from collections.abc import Collection

_BLOCK = "_resilient_httpx_block"


def mark_block(error: BaseException) -> BaseException:
    setattr(error, _BLOCK, True)
    return error


def is_block(error: BaseException | None) -> bool:
    return bool(getattr(error, _BLOCK, False))


def free_blocks(retrying, max_attempts: int):
    """``retrying`` (tenacity's) with blocks left out of the attempts and of the waits: it stops after
    ``max_attempts`` attempts that were not blocks."""
    counted = 0
    wait = retrying.wait

    def stop(state) -> bool:
        nonlocal counted
        if not is_block(state.outcome.exception() if state.outcome is not None else None):
            counted += 1
        return counted >= max_attempts

    def wait_unless_block(state) -> float:
        return 0.0 if is_block(state.outcome.exception() if state.outcome is not None else None) else wait(state)

    return retrying.copy(stop=stop, wait=wait_unless_block)


class BlockRules:
    """The rules both clients share (``blacklist_on`` is theirs)."""

    _blacklist_on: frozenset[int] | None

    def _set_block_rules(self, blacklist_on_exception: Collection[type[BaseException]] | None,
                         block_exceptions: Collection[type[BaseException]], blocks_count_as_attempts: bool) -> None:
        self._blacklist_on_exception = None if blacklist_on_exception is None else tuple(blacklist_on_exception)
        self._block_exceptions = tuple(block_exceptions)
        self._blocks_count_as_attempts = blocks_count_as_attempts

    def _exception_counts(self, error: BaseException) -> bool:
        """Whether a retried exception counts against the proxy it came through."""
        if isinstance(error, self._block_exceptions):
            return True
        return self._blacklist_on_exception is None or isinstance(error, self._blacklist_on_exception)

    def _mark_exception(self, error: BaseException, pool, proxy) -> None:
        if pool and proxy and isinstance(error, self._block_exceptions):
            mark_block(error)

    def _status_error(self, error: BaseException, pool, proxy) -> BaseException:
        """``error`` (a retried status's), marked a block where its status is in ``blacklist_on``."""
        status = error.response.status_code
        if pool and proxy and self._blacklist_on is not None and status in self._blacklist_on:
            mark_block(error)
        return error

    def _count_attempts(self, retrying):
        return retrying if self._blocks_count_as_attempts else free_blocks(retrying, self._retry.max_attempts)

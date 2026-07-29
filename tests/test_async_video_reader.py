"""Process-lifecycle tests for :class:`AsyncVideoReader`.

Every failure mode covered here originally presented as a *hang* rather than an
error: the worker process died, and the caller was left blocked on a future that
nothing remained to resolve. So every wait in this module is bounded — a
regression has to fail the suite, not stall it.
"""

import re
import threading
import time
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pytest

from asyncvideo import AsyncVideoReader
from asyncvideo.utils import ReaderError

# Long enough for a cold decode on a slow CI runner, short enough that a genuine
# hang ends the run instead of hanging it.
RESULT_TIMEOUT = 30.0
RELEASE_TIMEOUT = 10.0


@pytest.fixture()
def reader(video_path):
    """An ``AsyncVideoReader`` that is always shut down, even if the test fails.

    ``shutdown`` is idempotent, so tests that shut down explicitly are fine too.
    """
    r = AsyncVideoReader(video_path)
    try:
        yield r
    finally:
        r.shutdown()


def _segment_names(reader) -> tuple[str, ...]:
    return tuple(shm.name for shm in reader.shared_mems)


def _assert_segment_removed(name: str) -> None:
    """Fail unless the named segment is gone from the system.

    Deliberately only ever *attempts* the attach: on POSIX, successfully
    attaching to a segment also registers it with this process's
    resource_tracker, which would then try to unlink it again at exit.
    """
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=name)


def _wait_until(predicate, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


# ---------------------------------------------------------------------------
# Decoding correctness — the reader must return the frame that was asked for
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("idx", [0, 5, 42, 99])
def test_int_index_returns_correct_frame(reader, reference, idx):
    """An int index must decode, not hang, and return the *right* frame.

    ``VideoHandler`` returns a bare frame for an int but a list for a slice;
    the worker previously assumed a list and died on the subscript.
    """
    packed, height = reference
    y, _u, _v = reader[(idx,)].result(timeout=RESULT_TIMEOUT)
    np.testing.assert_array_equal(y[0], packed[idx][:height])


@pytest.mark.parametrize("idx", [0, 5, 42, 99])
def test_slice_index_returns_correct_frame(reader, reference, idx):
    packed, height = reference
    y, _u, _v = reader[(slice(idx, idx + 1),)].result(timeout=RESULT_TIMEOUT)
    np.testing.assert_array_equal(y[0], packed[idx][:height])


def test_planar_yuv_planes_are_correctly_subsampled(reader, reference):
    """U and V are half-resolution in both axes for yuv420p."""
    _packed, height = reference
    y, u, v = reader[(10,)].result(timeout=RESULT_TIMEOUT)
    assert y.shape[1:] == (height, y.shape[2])
    assert u.shape[1:] == (y.shape[1] // 2, y.shape[2] // 2)
    assert u.shape == v.shape


def test_packed_yuv_matches_reference(video_path, reference):
    packed, _height = reference
    r = AsyncVideoReader(video_path, yuv_packed=True)
    try:
        frame = r[(10,)].result(timeout=RESULT_TIMEOUT)
        np.testing.assert_array_equal(frame[0], packed[10])
    finally:
        r.shutdown()


# ---------------------------------------------------------------------------
# Worker failure — must surface as an exception, never as a hang
# ---------------------------------------------------------------------------

def test_failed_request_raises_instead_of_hanging(reader):
    """A request the worker cannot serve must fail the future.

    Before the fix the exception killed the worker process outright, so the
    future stayed pending forever and this call never returned.
    """
    # re.escape, not a raw string: ``match`` is a regex, and a raw string only
    # stops Python processing backslashes — it would not stop the regex engine
    # treating a future category name containing ``.`` or ``+`` as a wildcard.
    with pytest.raises(RuntimeError, match=re.escape(ReaderError.type_error.name)):
        reader[("not-an-index",)].result(timeout=RESULT_TIMEOUT)


def test_worker_survives_failed_request(reader, reference):
    """One bad request must not take the reader down with it."""
    packed, height = reference

    with pytest.raises(RuntimeError):
        reader[("not-an-index",)].result(timeout=RESULT_TIMEOUT)

    assert reader._worker.is_alive()

    y, _u, _v = reader[(3,)].result(timeout=RESULT_TIMEOUT)
    np.testing.assert_array_equal(y[0], packed[3][:height])


# ---------------------------------------------------------------------------
# Shutdown — segments released exactly once, whatever the caller does
# ---------------------------------------------------------------------------

def test_shutdown_unlinks_all_segments(reader):
    names = _segment_names(reader)
    assert names  # guard against the fixture silently changing shape

    reader.shutdown()

    for name in names:
        _assert_segment_removed(name)


def test_shutdown_is_idempotent(reader):
    """A second shutdown must not re-unlink and raise FileNotFoundError."""
    names = _segment_names(reader)

    reader.shutdown()
    reader.shutdown()

    for name in names:
        _assert_segment_removed(name)


class _BlockingLock:
    """Real lock that pins the *first* holder inside the critical section.

    Racing N threads and hoping they collide is not a test: they may serialize
    naturally and never overlap, so it passes without ever entering the window
    it claims to cover. This forces the interleaving instead — one thread is
    held inside the critical section while another provably blocks at the door.
    """

    def __init__(self, hold_for: float):
        self._inner = threading.Lock()
        self._hold_for = hold_for
        self.first_holder_inside = threading.Event()
        self._first = True

    def __enter__(self):
        self._inner.acquire()
        if self._first:
            self._first = False
            self.first_holder_inside.set()
            time.sleep(self._hold_for)
        return self

    def __exit__(self, *exc_info):
        self._inner.release()
        return False


def test_concurrent_shutdown_blocks_and_releases_exactly_once(reader):
    """A second shutdown must wait for the first, then not re-unlink.

    ``unlink`` is not idempotent, so a second caller that either slips past the
    guard or acts on a flag set but not yet honoured raises FileNotFoundError.
    """
    names = _segment_names(reader)
    # patch with the lock forcing a race condition
    reader._release_lock = lock = _BlockingLock(hold_for=0.5)
    errors: list[BaseException] = []

    def call_shutdown():
        try:
            reader.shutdown()
        except BaseException as exc:  # noqa: BLE001 - recorded, asserted on below
            errors.append(exc)

    first = threading.Thread(target=call_shutdown)
    first.start()

    # first thread is now inside the critical section, holding the lock
    assert lock.first_holder_inside.wait(RELEASE_TIMEOUT), "first shutdown never started"

    second = threading.Thread(target=call_shutdown)
    second.start()

    # while the first still holds the lock the second must be parked at it, not
    # sailing through into a concurrent teardown
    second.join(timeout=0.1)  # this gives time to reach the lock
    # this makes sure the lock is reached (aka race condition avoided)
    assert second.is_alive(), "second shutdown did not block on the release lock"

    for t in (first, second):
        t.join(timeout=RELEASE_TIMEOUT)
        assert not t.is_alive(), "shutdown deadlocked"

    assert not errors, f"concurrent shutdown raised: {errors}"
    for name in names:
        _assert_segment_removed(name)


def test_shutdown_wait_false_returns_promptly_and_still_releases(reader):
    """``wait=False`` must not block, but must still tear down eventually.

    The teardown cannot run inline: ``_buffer`` is a numpy view onto the shared
    segments and the listener thread may still be copying out of it, so
    unmapping underneath it would be a use-after-free rather than an exception.
    """
    names = _segment_names(reader)

    started = time.monotonic()
    reader.shutdown(wait=False)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"wait=False blocked for {elapsed:.2f}s"

    # the deferred teardown runs on a helper thread; give it a bounded window
    assert _wait_until(lambda: reader._released, RELEASE_TIMEOUT), (
        "deferred teardown never completed"
    )
    for name in names:
        _assert_segment_removed(name)

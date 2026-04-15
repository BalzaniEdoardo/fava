"""Unit tests for FrameBuffer."""
import pytest

from fava._pyav_video_reader import FrameBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frames(n):
    """Return n distinct sentinel objects standing in for av.VideoFrame."""
    return [object() for _ in range(n)]


# ---------------------------------------------------------------------------
# Basic get / put
# ---------------------------------------------------------------------------

def test_get_miss_returns_none():
    buf = FrameBuffer(maxsize=4)
    assert buf.get(0) is None


def test_put_then_get_returns_frame():
    buf = FrameBuffer(maxsize=4)
    f = object()
    buf.put(0, f)
    assert buf.get(0) is f


def test_contains_after_put():
    buf = FrameBuffer(maxsize=4)
    f = object()
    buf.put(3, f)
    assert 3 in buf
    assert 0 not in buf


def test_multiple_entries_retrieved_correctly():
    buf = FrameBuffer(maxsize=8)
    frames = _frames(5)
    for i, f in enumerate(frames):
        buf.put(i, f)
    for i, f in enumerate(frames):
        assert buf.get(i) is f


# ---------------------------------------------------------------------------
# Duplicate put
# ---------------------------------------------------------------------------

def test_duplicate_put_does_not_overwrite():
    """Putting the same index twice keeps the first frame (no-op on duplicate)."""
    buf = FrameBuffer(maxsize=4)
    f1, f2 = object(), object()
    buf.put(0, f1)
    buf.put(0, f2)
    assert buf.get(0) is f1


def test_duplicate_put_does_not_grow_cache():
    buf = FrameBuffer(maxsize=4)
    f = object()
    buf.put(0, f)
    buf.put(0, object())
    assert len(buf._cache) == 1


# ---------------------------------------------------------------------------
# FIFO eviction
# ---------------------------------------------------------------------------

def test_fifo_evicts_oldest_when_full():
    buf = FrameBuffer(maxsize=3)
    frames = _frames(4)
    for i, f in enumerate(frames):
        buf.put(i, f)
    # index 0 was inserted first → evicted
    assert buf.get(0) is None
    # indices 1, 2, 3 remain
    assert buf.get(1) is frames[1]
    assert buf.get(2) is frames[2]
    assert buf.get(3) is frames[3]


def test_fifo_eviction_order_across_multiple_overflows():
    buf = FrameBuffer(maxsize=2)
    frames = _frames(5)
    for i, f in enumerate(frames):
        buf.put(i, f)
    # only the two most recently inserted survive
    assert buf.get(3) is frames[3]
    assert buf.get(4) is frames[4]
    for i in range(3):
        assert buf.get(i) is None


def test_size_never_exceeds_maxsize():
    buf = FrameBuffer(maxsize=5)
    for i in range(20):
        buf.put(i, object())
    assert len(buf._cache) == 5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_maxsize_one():
    buf = FrameBuffer(maxsize=1)
    f0, f1 = object(), object()
    buf.put(0, f0)
    buf.put(1, f1)
    assert buf.get(0) is None
    assert buf.get(1) is f1


def test_empty_buffer_contains_nothing():
    buf = FrameBuffer(maxsize=10)
    for i in range(10):
        assert buf.get(i) is None
        assert i not in buf
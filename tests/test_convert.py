"""Tests for ``to_rgb``.

The ground truth throughout is PyAV's own ``frame.to_ndarray(format="rgb24")``,
taken from a frame decoded directly by PyAV. Each test hands ``to_rgb`` a
different representation of that *same* frame and asserts it lands on the same
RGB pixels, so the conversion is checked against an independent decode rather
than against itself.
"""

import av
import numpy as np
import pytest

from asyncvideo import AsyncVideoReader, pyav_trim_plane, to_rgb

RESULT_TIMEOUT = 30
FRAME_IDX = 10


@pytest.fixture(scope="module")
def frame_and_truth(video_path):
    """``(frame, rgb_truth)`` for a single frame decoded straight from PyAV."""
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i == FRAME_IDX:
                return frame, frame.to_ndarray(format="rgb24")
    pytest.fail(f"video has no frame {FRAME_IDX}")


def test_video_frame(frame_and_truth):
    frame, truth = frame_and_truth
    np.testing.assert_array_equal(to_rgb(frame), truth)


def test_video_frame_sequence(frame_and_truth):
    frame, truth = frame_and_truth
    out = to_rgb([frame, frame])
    assert out.shape == (2, *truth.shape)
    np.testing.assert_array_equal(out[0], truth)
    np.testing.assert_array_equal(out[1], truth)


def test_packed_yuv420p(frame_and_truth):
    """The packed (H*3//2, W) layout is what to_ndarray() gives with no format."""
    frame, truth = frame_and_truth
    packed = frame.to_ndarray()
    assert packed.ndim == 2
    np.testing.assert_array_equal(to_rgb(packed), truth)


def test_packed_yuv420p_stack(frame_and_truth):
    frame, truth = frame_and_truth
    packed = frame.to_ndarray()
    out = to_rgb(np.stack([packed, packed]))
    assert out.shape == (2, *truth.shape)
    np.testing.assert_array_equal(out[0], truth)


def test_plane_triple(frame_and_truth):
    """(Y, U, V) planes — the shape AsyncVideoReader hands back."""
    frame, truth = frame_and_truth
    planes = tuple(pyav_trim_plane(p) for p in frame.planes)
    np.testing.assert_array_equal(to_rgb(planes), truth)


def test_plane_triple_stack(frame_and_truth):
    frame, truth = frame_and_truth
    y, u, v = (pyav_trim_plane(p) for p in frame.planes)
    stacked = (np.stack([y, y]), np.stack([u, u]), np.stack([v, v]))
    out = to_rgb(stacked)
    assert out.shape == (2, *truth.shape)
    np.testing.assert_array_equal(out[0], truth)


def test_yuv444p_needs_explicit_format(frame_and_truth):
    """(3, H, W) is ambiguous with a yuv420p stack, so it must be named."""
    frame, truth = frame_and_truth
    arr = frame.to_ndarray(format="yuv444p")
    assert arr.shape == (3, *truth.shape[:2])

    out = to_rgb(arr, from_format="yuv444p")
    # yuv420p -> yuv444p upsamples chroma and back again, so this is not
    # bit-exact; it must still be the same image.
    np.testing.assert_allclose(out, truth, atol=2)


def test_rgb_passthrough(frame_and_truth):
    _frame, truth = frame_and_truth
    np.testing.assert_array_equal(to_rgb(truth), truth)
    stack = np.stack([truth, truth])
    np.testing.assert_array_equal(to_rgb(stack), stack)


def test_async_reader_output_round_trips(video_path, frame_and_truth):
    """End to end: the reader's real output converts to the reference pixels."""
    _frame, truth = frame_and_truth
    reader = AsyncVideoReader(video_path)
    try:
        planes = reader[(FRAME_IDX,)].result(timeout=RESULT_TIMEOUT)
        assert isinstance(planes, tuple) and len(planes) == 3
        out = to_rgb(planes)
        assert out.shape == (1, *truth.shape)
        np.testing.assert_array_equal(out[0], truth)
    finally:
        reader.shutdown()


def test_packed_async_reader_output_round_trips(video_path, frame_and_truth):
    _frame, truth = frame_and_truth
    reader = AsyncVideoReader(video_path, yuv_packed=True)
    try:
        packed = reader[(FRAME_IDX,)].result(timeout=RESULT_TIMEOUT)
        out = to_rgb(packed)
        assert out.shape == (1, *truth.shape)
        np.testing.assert_array_equal(out[0], truth)
    finally:
        reader.shutdown()


def test_video_handler_yuv_matches_rgb(video_path):
    """VideoHandler's own yuv420p output converts to its rgb24 output."""
    from asyncvideo import VideoHandler

    with VideoHandler(video_path, pixel_format="rgb24") as vh_rgb:
        expected = vh_rgb[FRAME_IDX]
    with VideoHandler(video_path, pixel_format="yuv420p") as vh_yuv:
        packed = vh_yuv[FRAME_IDX]

    np.testing.assert_array_equal(to_rgb(packed), expected)


# ---------------------------------------------------------------------------
# method forms — the reader supplies from_format itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pixel_format", [None, "rgb24", "yuv420p", "yuv444p"])
def test_video_handler_to_rgb_method(video_path, frame_and_truth, pixel_format):
    """vh.to_rgb() works for every pixel_format without naming it again."""
    from asyncvideo import VideoHandler

    _frame, truth = frame_and_truth
    with VideoHandler(video_path, pixel_format=pixel_format) as vh:
        out = vh.to_rgb(vh[FRAME_IDX])
    assert out.shape == truth.shape
    # yuv444p costs a chroma up/downsample round trip; the rest are exact
    atol = 2 if pixel_format == "yuv444p" else 0
    np.testing.assert_allclose(out, truth, atol=atol)


def test_video_handler_to_rgb_method_yuv444p_needs_no_hint(video_path):
    """The (3, H, W) case the function cannot infer is resolved by the method."""
    from asyncvideo import VideoHandler

    with VideoHandler(video_path, pixel_format="yuv444p") as vh:
        frame = vh[FRAME_IDX]
        assert frame.ndim == 3 and frame.shape[0] == 3
        # bare function would read this as a stack of packed yuv420p frames
        assert not np.array_equal(to_rgb(frame), vh.to_rgb(frame))


@pytest.mark.parametrize("yuv_packed", [False, True])
def test_async_reader_to_rgb_method(video_path, frame_and_truth, yuv_packed):
    _frame, truth = frame_and_truth
    reader = AsyncVideoReader(video_path, yuv_packed=yuv_packed)
    try:
        out = reader.to_rgb(reader[(FRAME_IDX,)].result(timeout=RESULT_TIMEOUT))
        assert out.shape == (1, *truth.shape)
        np.testing.assert_array_equal(out[0], truth)
    finally:
        reader.shutdown()


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


def test_empty_sequence_raises():
    with pytest.raises(ValueError, match="empty sequence"):
        to_rgb([])


def test_wrong_length_sequence_raises(frame_and_truth):
    _frame, truth = frame_and_truth
    with pytest.raises(TypeError, match="plane triple"):
        to_rgb([truth, truth])


def test_unsupported_from_format_raises(frame_and_truth):
    _frame, truth = frame_and_truth
    with pytest.raises(ValueError, match="unsupported from_format"):
        to_rgb(truth, from_format="gray")


def test_uninferable_shape_raises():
    with pytest.raises(ValueError, match="cannot infer"):
        to_rgb(np.zeros((2, 3, 4, 5, 6), dtype=np.uint8))


def test_bad_ndim_for_named_format_raises():
    with pytest.raises(ValueError, match="not a single yuv420p frame"):
        to_rgb(np.zeros((2, 3, 4, 5), dtype=np.uint8), from_format="yuv420p")

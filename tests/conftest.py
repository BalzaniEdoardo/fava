"""Shared fixtures for the asyncvideo test suite."""

import pathlib

import av
import numpy as np
import pytest

TEST_VIDEO_DIR = pathlib.Path(__file__).parent / "test_video"


@pytest.fixture(scope="session")
def video_path() -> pathlib.Path:
    """A single well-behaved video, used by the async reader tests."""
    path = TEST_VIDEO_DIR / "numbered_video_libx264.mp4"
    if not path.exists():
        pytest.skip(f"{path} missing — run 'nox -s video_gen' to generate test videos")
    return path


@pytest.fixture(scope="session")
def reference(video_path):
    """``(packed_frames, height)`` decoded straight from PyAV.

    This is the ground truth the async reader is compared against: decoding
    here goes through PyAV directly, with none of asyncvideo's seeking, buffering or
    shared-memory machinery involved. Decoded once per session.
    """
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        height = stream.height
        frames = np.stack([frame.to_ndarray() for frame in container.decode(stream)])
    return frames, height

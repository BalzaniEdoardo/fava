from __future__ import annotations

from concurrent.futures import Future
from multiprocessing.shared_memory import SharedMemory
from typing import TypeAlias

import av
import numpy as np
from numpy.typing import NDArray
from enum import StrEnum

UInt8Array: TypeAlias = NDArray[np.uint8]
TupleYUV: TypeAlias = tuple[UInt8Array, UInt8Array, UInt8Array]
FutureArray: TypeAlias = Future[UInt8Array | TupleYUV]
SharedMemRGB: TypeAlias = tuple[SharedMemory]
SharedMemYUV: TypeAlias = tuple[SharedMemory, SharedMemory, SharedMemory]


class Colorspace(StrEnum):
    rgb24 = "rgb24"
    yuv420p = "yuv420p"


def create_shared_memory(
    frame: av.VideoFrame, n_frames: int = 1
) -> SharedMemYUV | SharedMemRGB:
    colorspace = frame.format.name

    rows, cols = frame.height, frame.width

    if colorspace == Colorspace.rgb24:
        return (SharedMemory(create=True, size=rows * cols * 3 * n_frames),)

    elif colorspace == Colorspace.yuv420p:
        y = SharedMemory(create=True, size=rows * cols * n_frames)

        rows_chroma, cols_chroma = (
            frame.format.chroma_height(),
            frame.format.chroma_width(),
        )
        u = SharedMemory(create=True, size=rows_chroma * cols_chroma * n_frames)
        v = SharedMemory(create=True, size=rows_chroma * cols_chroma * n_frames)

        return y, u, v

    else:
        raise ValueError(
            f"only rgb24 and yuv420p colorspaces are currently supported, "
            f"provided video with colorspace: {colorspace}"
        )


def create_buffers(
    shared_mems: SharedMemYUV | SharedMemRGB,
    colorspace: Colorspace,
    shape_frame: tuple[int, int],
    shape_chroma: tuple[int, int] | None,
    n_frames: int = 1,
) -> UInt8Array | TupleYUV:
    if colorspace == Colorspace.yuv420p:
        buffer_y = np.ndarray(
            shape=(n_frames, *shape_frame), dtype=np.uint8, buffer=shared_mems[0].buf
        )
        buffer_u = np.ndarray(
            shape=(n_frames, *shape_chroma), dtype=np.uint8, buffer=shared_mems[1].buf
        )
        buffer_v = np.ndarray(
            shape=(n_frames, *shape_chroma), dtype=np.uint8, buffer=shared_mems[2].buf
        )

        return buffer_y, buffer_u, buffer_v

    elif colorspace == Colorspace.rgb24:
        # rgb
        buffer_rgb = np.ndarray(
            shape=(n_frames, *shape_frame), dtype=np.uint8, buffer=shared_mems[0].buf
        )

        return buffer_rgb

    else:
        raise ValueError(
            f"only rgb24 and yuv420p colorspaces are currently supported, "
            f"provided video with colorspace: {colorspace}"
        )

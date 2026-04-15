import queue
from multiprocessing import Queue, Event, Lock
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import av
import numpy as np

from ._pyav_video_reader import VideoHandler, pyav_trim_plane
from .utils import SharedMemYUV, SharedMemRGB, Colorspace, create_buffers


def _reader_process(
        path: Path,
        shared_mem_names: str,
        colorspace: Colorspace,
        shape_frame: tuple[int, int],
        shape_chroma: tuple[int, int] | None,
        request_queue: Queue,
        response_queue: Queue,
        stop_event: Event,
        cancel_event: Event,
        buffer_lock: Lock,
):
    vr = VideoHandler(path, pixel_format=None)

    shared_mems: SharedMemRGB | SharedMemYUV = tuple(SharedMemory(name=n) for n in shared_mem_names)

    buffer = create_buffers(
        shared_mems,
        colorspace=colorspace,
        shape_frame=shape_frame,
        shape_chroma=shape_chroma,
        n_frames=1
    )

    try:
        while not stop_event.is_set():
            try:
                request = request_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if request is None:
                break

            if cancel_event.is_set():
                cancel_event.clear()
                continue

            rid, index = request
            frame: av.VideoFrame = vr[index][0]

            if cancel_event.is_set():
                cancel_event.clear()
                continue

            # TODO: Deal with n_frames changing
            # if frame.shape != buf.shape or frame.dtype != dtype:
            #     shared_mems.close()
            #     shared_mems = SharedMemory(create=True, size=frame.nbytes)
            #     buf = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shared_mems.buf)
            #     dtype = frame.dtype

            if cancel_event.is_set():
                cancel_event.clear()
                continue

            with buffer_lock:
                if frame.format.name == Colorspace.rgb24:
                    np.copyto(buffer, pyav_trim_plane(frame.planes[0]), casting="no")

                elif frame.format.name == Colorspace.yuv420p:
                    np.copyto(buffer[0], pyav_trim_plane(frame.planes[0]), casting="no")
                    np.copyto(buffer[1], pyav_trim_plane(frame.planes[1]), casting="no")
                    np.copyto(buffer[2], pyav_trim_plane(frame.planes[2]), casting="no")

                response_queue.put(rid)
    finally:
        try:
            if hasattr(vr, "close"):
                vr.close()
        except Exception as e:
            print(f"[_reader_process] Failed to close video reader: {e}")
        try:
            shared_mems.close()
        except Exception:
            pass

import queue
from multiprocessing import Queue, Event, Lock
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.sharedctypes import Synchronized
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
        yuv_packed: bool,
        request_queue: Queue,
        response_queue: Queue,
        stop_event: Event,
        latest_rid: Synchronized,
        buffer_lock: Lock,
):
    vr = VideoHandler(path, pixel_format=None)

    shared_mems: SharedMemRGB | SharedMemYUV = tuple(SharedMemory(name=n) for n in shared_mem_names)

    buffer = create_buffers(
        shared_mems,
        colorspace=colorspace,
        shape_frame=shape_frame,
        shape_chroma=shape_chroma,
        n_frames=1,
        yuv_packed=yuv_packed,
    )

    try:
        while not stop_event.is_set():
            try:
                request = request_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if request is None:
                break

            rid, index = request

            # skip if a newer request has already been submitted - precise
            # per-rid check, unlike a single shared cancel bit which can't
            # distinguish which request was cancelled
            if rid < latest_rid.value:
                continue

            frame: av.VideoFrame = vr[index][0]

            # re-check after decode (decode can be slow; a newer request may
            # have arrived in the meantime)
            if rid < latest_rid.value:
                continue

            # TODO: Deal with n_frames changing
            # if frame.shape != buf.shape or frame.dtype != dtype:
            #     shared_mems.close()
            #     shared_mems = SharedMemory(create=True, size=frame.nbytes)
            #     buf = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shared_mems.buf)
            #     dtype = frame.dtype

            with buffer_lock:
                # final check before writing; if we've been superseded, don't
                # clobber the buffer for whatever rid the listener may still be
                # mid-read on
                if rid < latest_rid.value:
                    continue

                if frame.format.name == Colorspace.rgb24:
                    np.copyto(buffer, pyav_trim_plane(frame.planes[0]), casting="no")

                elif frame.format.name == Colorspace.yuv420p:
                    if yuv_packed:
                        np.copyto(buffer, frame.to_ndarray(), casting="no")
                    else:
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

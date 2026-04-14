import os
import queue

# decord uses up all the RAM otherwise
os.environ["DECORD_EOF_RETRY_MAX"] = "128"

from multiprocessing import Queue, Event, Lock
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np


def _reader_process(
    path: Path,
    kwargs: dict,
    shm_name: str,
    frame_shape: tuple,
    dtype: str,
    request_queue: Queue,
    response_queue: Queue,
    stop_event: Event,
    cancel_event: Event,
    buffer_lock: Lock,
):
    import os
    from . import config

    if config.backend == "decord":
        from decord import VideoReader, gpu

        os.environ["DECORD_EOF_RETRY_MAX"] = "128"
        vr = VideoReader(str(path), ctx=gpu(0))
        as_numpy = lambda frame: frame.asnumpy()
        as_numpy(vr[slice(0, 1)])
        vr.seek(0)
    else:
        from ._pyav_video_reader import VideoHandler

        as_numpy = lambda frame: frame
        vr = VideoHandler(path)

    dtype = np.dtype(dtype)
    shm = SharedMemory(name=shm_name)
    buf = np.ndarray(frame_shape, dtype=dtype, buffer=shm.buf)

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
            frame = as_numpy(vr[index])

            if cancel_event.is_set():
                cancel_event.clear()
                continue

            if frame.shape != buf.shape or frame.dtype != dtype:
                shm.close()
                shm = SharedMemory(create=True, size=frame.nbytes)
                buf = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
                dtype = frame.dtype

            if cancel_event.is_set():
                cancel_event.clear()
                continue

            with buffer_lock:
                np.copyto(buf, frame)
                response_queue.put((rid, frame.shape, str(frame.dtype), shm.name))
    finally:
        try:
            if hasattr(vr, "close"):
                vr.close()
        except Exception as e:
            print(f"[_reader_process] Failed to close video reader: {e}")
        try:
            shm.close()
        except Exception:
            pass
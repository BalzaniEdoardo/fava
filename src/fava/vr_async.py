from __future__ import annotations

import os

# decord uses up all the RAM otherwise
os.environ["DECORD_EOF_RETRY_MAX"] = "128"

from concurrent.futures import Future
import multiprocessing
from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
import threading

import sys

import numpy as np

try:
    from decord import VideoReader
except ImportError:
    VideoReader = None

from ._pyav_video_reader import VideoHandler
from .config import config
from ._vr_process import _reader_process

# fork clones the parent process directly — no re-import of the main script,
# so AsyncVideoReader can be instantiated at module level (e.g. in scripts or
# IPython) without hitting the spawn bootstrap trap. Windows only has spawn.
if sys.platform == "win32":
    mp_ctx = multiprocessing.get_context("spawn")
else:
    mp_ctx = multiprocessing.get_context("fork")


class AsyncVideoReader:
    def __array__(self) -> AsyncVideoReader:
        return self

    def __init__(self, path: str | Path, **kwargs):
        self._path = Path(path)
        self._kwargs = kwargs

        if config.backend == "decord":
            vr = VideoReader(str(self._path), num_threads=1)
            frame0 = vr[10].asnumpy()
            vr.seek(0)
        else:
            vr = VideoHandler(self._path, pixel_format="rgb24")
            frame0 = vr[0]

        self._shape = (len(vr), *frame0.shape)
        self._dtype = np.dtype(frame0.dtype)
        if hasattr(vr, "close"):
            vr.close()
        del vr

        self._shm = SharedMemory(create=True, size=frame0.nbytes)

        self._request_queue: Queue = mp_ctx.Queue()
        self._response_queue: Queue = mp_ctx.Queue()

        self._stop_event = mp_ctx.Event()
        self._cancel_event = mp_ctx.Event()
        self._buffer_lock = mp_ctx.Lock()

        self._pending_rid: int = 0
        self._pending_future: Future | None = None
        self._lock = threading.Lock()

        self._result = np.ndarray((1, *frame0.shape), dtype=self.dtype, buffer=self._shm.buf)

        self._worker = mp_ctx.Process(
            target=_reader_process,
            kwargs=dict(
                path=self._path,
                kwargs=self._kwargs,
                shm_name=self._shm.name,
                frame_shape=(1, *frame0.shape),
                dtype=str(self._dtype),
                request_queue=self._request_queue,
                response_queue=self._response_queue,
                stop_event=self._stop_event,
                cancel_event=self._cancel_event,
                buffer_lock=self._buffer_lock,
            ),
            daemon=True,
        )
        self._worker.start()

        self._listener = threading.Thread(target=self._listen, daemon=True)
        self._listener.start()

    def _listen(self):
        while True:
            msg = self._response_queue.get()
            if msg is None:
                break

            rid, frame_shape, dtype, shm_name = msg

            with self._lock:
                if rid != self._pending_rid:
                    continue

                if shm_name != self._shm.name:
                    self._shm.unlink()
                    self._shm.close()
                    self._shm = SharedMemory(name=shm_name)
                    self._result = np.ndarray(frame_shape, dtype=np.dtype(dtype), buffer=self._shm.buf)

                future = self._pending_future

            with self._buffer_lock:
                frame_copy = self._result.copy()

            future.set_result(frame_copy)

    def __getitem__(self, index) -> Future:
        with self._lock:
            if self._pending_future is not None and not self._pending_future.done():
                self._pending_future.cancel()
                print("cancelled")
                self._cancel_event.set()

            self._pending_rid += 1
            # a trick to get the resultant shape of the sliced array with a zero-memory dummy array
            future = Future()
            self._pending_future = future

        self._request_queue.put((self._pending_rid, index[0]))
        return future

    def shutdown(self, wait: bool = True):
        self._stop_event.set()
        self._request_queue.put(None)  # wake up the worker if blocked on get()
        if wait:
            self._worker.join()
            self._response_queue.put(None)
            self._listener.join()
        self._shm.unlink()
        self._shm.close()

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self._shape)
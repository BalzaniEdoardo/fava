from __future__ import annotations

from concurrent.futures import Future
import multiprocessing
from multiprocessing import Queue
from pathlib import Path
import threading

import sys

import numpy as np

from ._pyav_video_reader import VideoHandler
from ._vr_process import _reader_process
from .utils import Colorspace, FutureArray, SharedMemRGB, SharedMemYUV, create_shared_memory, create_buffers

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

    def __init__(
            self,
            path: str | Path,
            yuv_packed: bool = False,
            **kwargs,
    ):
        self._path = Path(path)
        self._kwargs = kwargs

        vr = VideoHandler(self._path, pixel_format=None)
        frame0 = vr[(slice(0, 1),)][0]

        # width, height to rows, cols
        self._shape_frame = vr.shape[2], vr.shape[1]
        n_frames = vr.shape[0]

        colorspace = Colorspace(frame0.format.name)

        self._colorspace = colorspace
        frame0_numpy = frame0.to_ndarray()
        self._dtype = frame0_numpy.dtype

        if self.colorspace == Colorspace.rgb24:
            self._shape = (n_frames, *self._shape_frame, 3)
            self._shape_chroma = None

        elif self.colorspace == Colorspace.yuv420p:
            self._shape = (n_frames, *self._shape_frame)
            self._shape_chroma = frame0.format.chroma_height(), frame0.format.chroma_width()

        n_frames = 1

        self._yuv_packed = yuv_packed

        self._shared_mems = create_shared_memory(frame0, n_frames=n_frames, yuv_packed=self._yuv_packed)
        shared_mem_names = tuple(b.name for b in self.shared_mems)

        vr.close()
        del vr

        self._request_queue: Queue = mp_ctx.Queue()
        self._response_queue: Queue = mp_ctx.Queue()

        self._stop_event = mp_ctx.Event()
        self._cancel_event = mp_ctx.Event()
        self._buffer_lock = mp_ctx.Lock()

        self._pending_rid: int = 0
        self._pending_future: FutureArray | None = None
        self._listener_lock = threading.Lock()

        self._buffer = create_buffers(
            self._shared_mems,
            colorspace=self.colorspace,
            shape_frame=self._shape_frame,
            shape_chroma=self._shape_chroma,
            n_frames=1,
            yuv_packed=self._yuv_packed,
        )

        self._worker = mp_ctx.Process(
            target=_reader_process,
            kwargs=dict(
                path=self._path,
                shared_mem_names=shared_mem_names,
                colorspace=self.colorspace,
                shape_frame=self._shape_frame,
                shape_chroma=self._shape_chroma,
                yuv_packed=yuv_packed,
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

    @property
    def shared_mems(self) -> SharedMemRGB | SharedMemYUV:
        return self._shared_mems

    @property
    def colorspace(self) -> Colorspace:
        return self._colorspace

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self._shape)

    def _listen(self):
        while True:
            rid = self._response_queue.get()
            if rid is None:
                break

            with self._listener_lock:
                if rid != self._pending_rid:
                    continue

                # TODO: if shared mem changes due to different number of frames
                # if shm_name != self._shared_mems.name:
                #     self._shared_mems.unlink()
                #     self._shared_mems.close()
                #     self._shared_mems = SharedMemory(name=shm_name)
                #
                #     self._result = np.ndarray(
                #         frame_shape, dtype=np.dtype(dtype), buffer=self._shared_mems.buf
                #     )

                future = self._pending_future

            with self._buffer_lock:
                if self.colorspace == Colorspace.rgb24 or self._yuv_packed:
                    future.set_result(self._buffer.copy())

                elif self.colorspace == Colorspace.yuv420p:
                    future.set_result(
                        (
                            self._buffer[0].copy(),
                            self._buffer[1].copy(),
                            self._buffer[2].copy(),
                        )
                    )

    def __getitem__(self, index) -> FutureArray:
        with self._listener_lock:
            if self._pending_future is not None and not self._pending_future.done():
                self._pending_future.cancel()
                print("cancelled")
                self._cancel_event.set()

            self._pending_rid += 1
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
        self._shared_mems.unlink()
        self._shared_mems.close()

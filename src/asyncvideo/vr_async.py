from __future__ import annotations

import multiprocessing
import queue as _stdlib_queue
import sys
import threading
from concurrent.futures import Future
from multiprocessing import Queue
from pathlib import Path

import numpy as np

from ._pyav_video_reader import VideoHandler
from ._vr_process import _reader_process
from .convert import to_rgb
from .utils import (
    Colorspace,
    FutureArray,
    ReaderError,
    SharedMemRGB,
    SharedMemYUV,
    create_buffers,
    create_shared_memory,
)

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

        # take the pixel grid from the frame itself: VideoHandler.shape reports the
        # native to_ndarray() layout here, which for yuv420p is (h * 3 // 2, w)
        self._shape_frame = frame0.height, frame0.width
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
            self._shape_chroma = (
                frame0.format.chroma_height(),
                frame0.format.chroma_width(),
            )

        n_frames = 1

        self._yuv_packed = yuv_packed

        self._shared_mems = create_shared_memory(
            frame0, n_frames=n_frames, yuv_packed=self._yuv_packed
        )
        shared_mem_names = tuple(b.name for b in self.shared_mems)

        vr.close()
        del vr

        self._request_queue: Queue = mp_ctx.Queue()
        self._response_queue: Queue = mp_ctx.Queue()

        self._stop_event = mp_ctx.Event()
        self._buffer_lock = mp_ctx.Lock()

        self._pending_rid: int = 0
        # shared with the worker process. The worker uses this to skip any
        # request whose rid has already been superseded by a newer one,
        # without false-dropping the current request (a single ``cancel_event``
        # bit can't distinguish *which* request was cancelled).
        self._latest_rid = mp_ctx.Value("q", 0)
        self._pending_future: FutureArray | None = None
        self._listener_lock = threading.Lock()

        # guards the shared-memory teardown so a second shutdown() cannot
        # unlink segments that are already gone
        self._release_lock = threading.Lock()
        self._released = False

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
            kwargs={
                "path": self._path,
                "shared_mem_names": shared_mem_names,
                "colorspace": self.colorspace,
                "shape_frame": self._shape_frame,
                "shape_chroma": self._shape_chroma,
                "yuv_packed": yuv_packed,
                "request_queue": self._request_queue,
                "response_queue": self._response_queue,
                "stop_event": self._stop_event,
                "latest_rid": self._latest_rid,
                "buffer_lock": self._buffer_lock,
            },
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

    def to_rgb(self, frames) -> np.ndarray:
        """
        Convert a resolved request from this reader to RGB.

        Same as the module-level `asyncvideo.to_rgb`, except that the source
        format is taken from this reader's ``colorspace``.

        Parameters
        ----------
        frames :
            The result of a future returned by ``__getitem__`` — either a
            ``(Y, U, V)`` plane tuple or, with ``yuv_packed=True``, a packed
            array.

        Returns
        -------
        :
            ``(n, H, W, 3)`` uint8.

        Examples
        --------
        >>> reader = AsyncVideoReader("example.mp4")  # doctest: +SKIP
        >>> rgb = reader.to_rgb(reader[(10,)].result())  # doctest: +SKIP
        """
        return to_rgb(frames, from_format=str(self.colorspace))

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self._shape)

    def _listen(self):
        while True:
            msg = self._response_queue.get()
            if msg is None:
                break

            rid, status = msg

            with self._listener_lock:
                if rid != self._pending_rid:
                    continue

                if status != ReaderError.ok:
                    # Fail the future rather than leave the caller blocked: the
                    # worker survives the error, so nothing else will ever
                    # resolve this request. The traceback is in the worker log.
                    self._pending_future.set_exception(
                        RuntimeError(
                            f"reader process failed to decode request {rid} "
                            f"({ReaderError(status).name})"
                        )
                    )
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

                # set_result must happen while holding _listener_lock, otherwise
                # __getitem__ can cancel ``future`` between this point and the
                # set_result call below, raising InvalidStateError and killing
                # the listener thread (the reader then permanently hangs).
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

    @staticmethod
    def _frame_index(index):
        """Extract the frame selector from an index.

        Accepts ``reader[i]`` and ``reader[i:j]`` as well as the tuple form
        ``reader[(i,)]``, where only the first entry selects frames and any
        remaining entries are spatial slices handled downstream.

        Unpacking happens before ``__getitem__`` touches any state: it cancels
        the previous request and bumps the request id, so an index that cannot
        even be unpacked must fail before that, not halfway through.
        """
        if isinstance(index, tuple):
            if not index:
                raise IndexError("an empty tuple is not a valid frame index")
            index = index[0]
        if isinstance(index, np.integer):
            return int(index)
        # anything else (including a plain int or slice) is forwarded as-is; the
        # worker reports an unservable index by failing the future
        return index

    def __getitem__(self, index) -> FutureArray:
        frame_index = self._frame_index(index)

        with self._listener_lock:
            if self._pending_future is not None and not self._pending_future.done():
                self._pending_future.cancel()

            self._pending_rid += 1
            # publish to the worker so it knows the newest rid in flight
            self._latest_rid.value = self._pending_rid
            future = Future()
            self._pending_future = future

        # drain stale entries before enqueuing the new one. The worker will
        # skip them via the rid check anyway, but each multiprocessing.Queue
        # ``get`` is several ms of cross-process IPC - hundreds of stale
        # entries (from rapid slider dragging) turn into multi-second delays
        # before the worker reaches the latest request.
        while True:
            try:
                self._request_queue.get_nowait()
            except _stdlib_queue.Empty:
                break

        self._request_queue.put((self._pending_rid, frame_index))
        return future

    def shutdown(self, wait: bool = True):
        self._stop_event.set()
        self._request_queue.put(None)  # wake up the worker if blocked on get()
        if wait:
            self._release()
        else:
            # The teardown cannot run inline here: ``_buffer`` is a numpy view
            # onto the shared segments and ``_listen`` may still be copying out
            # of it, so unmapping them while it runs is a use-after-free (a
            # segfault, not an exception). Hand the join + teardown to a helper
            # thread so the caller still returns immediately.
            threading.Thread(target=self._release, daemon=True).start()

    def _release(self):
        """Stop the worker and listener, then unmap and destroy the segments."""
        self._worker.join()
        # only stop the listener once the worker is gone and no further results
        # can land on the queue
        self._response_queue.put(None)
        self._listener.join()

        # Held across the whole teardown, not just the flag check: a concurrent
        # shutdown() must block until the segments are actually gone rather than
        # return early on a flag that is set but not yet acted on. (An Event is
        # the wrong primitive here — is_set()/set() is a check-then-act, so two
        # callers can both reach unlink() and the second raises FileNotFoundError.)
        with self._release_lock:
            if self._released:
                return
            self._released = True

            # drop the numpy views before unmapping the memory they point at
            self._buffer = None

            # _shared_mems is a tuple: 1 segment for rgb24/packed-yuv, 3 for
            # planar yuv. close() releases this process's mapping; unlink()
            # destroys the segment and must happen exactly once, from the owner
            # — this process created them, so it unlinks and the worker only
            # closes.
            for shm in self._shared_mems:
                shm.close()
                shm.unlink()

# asyncvideo

Async video reader with fast random frame access.

Video files are compressed, and that compression has two practical consequences:

1. **Decoding an arbitrary frame is non-trivial.** Frames are not stored independently, so you cannot simply jump to a frame and read it — getting the *right* frame out means dealing with how the stream is encoded.
2. **Decoding is computationally intensive**, which limits the frame rate a single process can sustain. This becomes a problem when you stream more than one video at a time.

`asyncvideo` takes on both.

**`VideoHandler`** makes reading one frame, or many, as simple as slicing an array:

```python
from asyncvideo import VideoHandler

with VideoHandler("example.mp4", pixel_format="rgb24") as video:
    print(video.shape)               # (5000, 480, 640, 3) — frames, height, width, RGB

    frame = video[4200]              # one frame, by number
    clip = video[1000:2000:10]       # a strided range
    crop = video[0:100, 0:64, 0:64]  # frames, plus a spatial crop

    # or by time in seconds, using each frame's timestamp
    at_time = video.get(12.5)
    window = video.get_slice(12.5, 13.0)   # a time range -> a slice
    half_second = video[window]
```

**`AsyncVideoReader`** streams several videos in parallel. It runs one decoder process per open video and returns a `Future` instead of blocking, so the streams decode concurrently. Frames are indexed the same way, one at a time:

```python
from asyncvideo import AsyncVideoReader

# Assume multi-camera recordings
paths = ["cam0.mp4", "cam1.mp4", "cam2.mp4"]
readers = [AsyncVideoReader(p) for p in paths]

# every video starts decoding at once; 
# the futures are returned immediately, before the frames are available
futures = [r[4200] for r in readers]

# result() waits for the frame
frames = [f.result() for f in futures]

for r in readers:
    r.shutdown()
```

## Overview

The two readers exist for different jobs.

**`VideoHandler` is for analysis and inspection.** You have a recording, and something computed from it: per-frame classifier labels, tracking points, scored behavioural epochs. For a concrete example, you want to look at the frames a result refers to. Say you have the start and end times of mouse grooming bouts — you want the frames for one bout, as an array, to check them or plot them. `VideoHandler` lets you index and slice a video by frame number or by time, and hands back numpy arrays.

**`AsyncVideoReader` is for multi-view display.** Decoding is expensive, so showing several videos at the same time — a multi-camera rig, for instance — needs more than one decoder. `AsyncVideoReader` gives each video its own process and returns futures, so the streams decode in parallel and the displaying thread never waits on any single one of them.

Because the jobs differ, so do the APIs:

| | `VideoHandler` | `AsyncVideoReader` |
|---|---|---|
| Returns | frames, immediately | a `Future` |
| Decodes in | the calling thread | a separate process, one per reader |
| Frames per request | one, or a slice of many | **one only** |
| Indexing | `[i]`, `[i:j:k]`, `[-1]`, spatial crop | `[i]` |
| By timestamp | `get`, `get_slice` | `get` |
| Your own timestamps | `time=` | `time=` |
| Pixel format | `pixel_format=` | always native; use `to_rgb` |
| Array attributes | `shape`, `frame_shape`, `time`, `dtype`, `ndim`, `len()` | `shape`, `time`, `dtype`, `ndim` |
| Best for | scripts, analysis, batch work | interactive UIs, sliders, live display |

Two behaviours of `AsyncVideoReader` worth knowing before you use it:

- **It returns one frame per request.** The shared-memory buffer holds a single frame, so a slice does not fetch a range. Results also keep a leading axis of length 1, so a converted frame is `(1, H, W, 3)` and displaying it means taking `[0]`.
- **It supersedes in-flight requests.** If a new frame is requested while an older request is still decoding, the old one is cancelled. Dragging a slider therefore stays responsive, because the reader does not work through a backlog of frames that are no longer needed.

Each reader owns a process, so remember to call `shutdown()` when you are done with it.

### Analysis: the frames for a behavioural epoch

Frame times rarely start at zero or fall on an exact grid, so pass `time=` — one timestamp per frame, from the acquisition system — and every lookup uses your clock rather than a frame rate guessed from the container:

```python
import numpy as np
from asyncvideo import VideoHandler

frame_times = np.load("frame_times.npy")     # one timestamp per frame, in seconds
grooming_start, grooming_end = 412.5, 415.0  # a scored behavioural epoch

with VideoHandler("session.mp4", time=frame_times, pixel_format="rgb24") as video:
    window = video.get_slice(grooming_start, grooming_end)
    bout = video[window]                     # (n_frames, height, width, 3)

    print(bout.shape)
    print(video.time[window])                # the timestamp of each frame returned
```

Video and other recorded signals — spike times, a behavioural trace — can then be indexed by the same number, without converting between clocks at every call.

Note that `get_slice` returns a `slice`, not the frames. This is deliberate: a time range says nothing about how many frames it covers, so slicing straight by time risks materialising an enormous array. Ten minutes of 640x480 video at 30 fps is 18,000 frames, which is 16.6 GB as `rgb24`. Returning the slice first lets you inspect what you asked for before deciding to read it:

```python
window = video.get_slice(0.0, 600.0)        # ten minutes
print(window.stop - window.start)           # 18000 frames — probably not what you want

bout = video[window]                        # nothing is decoded until this line
```

### Multi-view: several cameras at once

Issue every request before collecting any result. That is what makes the decodes overlap rather than run one after another:

```python
from asyncvideo import AsyncVideoReader

paths = ["cam_top.mp4", "cam_side.mp4", "cam_front.mp4"]
readers = [AsyncVideoReader(p) for p in paths]
try:
    frame_index = 4200
    futures = [r[frame_index] for r in readers]   # all three decode at the same time
    # to_rgb converts the reader's YUV output for display (see Pixel formats below)
    views = [r.to_rgb(f.result())[0] for r, f in zip(readers, futures)]
finally:
    for r in readers:
        r.shutdown()
```

Showing frame 4200 from three cameras therefore costs about as much as showing it from the slowest one, rather than the sum of all three.

In practice the cameras have their own timestamps and need not share a frame rate, so one moment in the experiment is a *different frame index* in each view. Ask by time instead and there is no index to map:

```python
# times: one timestamp per frame, from the acquisition system
readers = {
    label: AsyncVideoReader(path, time=times)
    for label, (path, times) in cameras.items()
}

futures = {label: r.get(300.0) for label, r in readers.items()}   # t = 300 s in every view
views = {label: readers[label].to_rgb(f.result())[0] for label, f in futures.items()}
```

[`examples/ibl_multiview.py`](examples/ibl_multiview.py) is a runnable version of this against a public [International Brain Laboratory](https://www.internationalbrainlab.com) session, which records three cameras at different rates. It needs the docs extra (`pip install -e ".[docs]"`) and downloads the session's video once — 2.25 GB for the body camera alone, so it is not a quick first run.

![Frames read from a video by index and by timestamp](docs/images/random_access.png)

## Installation

```bash
pip install asyncvideo
```

Requires Python 3.11 or newer. The only dependencies are [numpy](https://numpy.org) and [PyAV](https://pyav.org), which provides the FFmpeg bindings — PyAV ships binary wheels for common platforms, so a system FFmpeg install is usually not needed.

For development, including the test suite:

```bash
git clone https://github.com/BalzaniEdoardo/asyncvideo
cd asyncvideo
pip install -e ".[dev]"
nox -s video_gen   # generate the test videos
nox -s tests
```

## Pixel formats and converting to RGB

`pixel_format` controls what you get back, and the choice is a real trade-off:

| `pixel_format` | You get | Notes |
|---|---|---|
| `None` (default) | `av.VideoFrame` | No conversion at all — cheapest |
| `"rgb24"` | `(H, W, 3)` uint8 | What plotting libraries expect |
| `"yuv420p"` | packed `(H * 3 // 2, W)` uint8 | Half the bytes of RGB |
| `"yuv444p"` | `(3, H, W)` uint8 | Full-resolution chroma |

YUV is more compact than RGB because the colour channels are stored at reduced resolution — `yuv420p` carries a frame in half the bytes of `rgb24`. When frames are being moved around rather than looked at (between processes, over a network, into a GPU texture that samples YUV directly) that is a real saving, and it is why `AsyncVideoReader` uses YUV for its shared-memory transfer.

The drawback is that plotting libraries do not accept YUV. To display a frame, convert it with `to_rgb`:

```python
import matplotlib.pyplot as plt
from asyncvideo import VideoHandler

with VideoHandler("example.mp4", pixel_format="yuv420p") as video:
    plt.imshow(video.to_rgb(video[7]))
```

The reader's `to_rgb` knows which format it was configured with, so you never repeat it. There is also a module-level function, for arrays that have outlived their reader:

```python
from asyncvideo import to_rgb

to_rgb(frame)                              # av.VideoFrame, or a list of them
to_rgb(planes)                             # (Y, U, V) tuple from AsyncVideoReader
to_rgb(packed)                             # packed yuv420p array
to_rgb(arr, from_format="yuv444p")         # (3, H, W) must be named explicitly
```

Conversion is done by libav through PyAV rather than by a hand-written matrix, so the colour coefficients and range used are the ones FFmpeg would use for that stream.

A single `yuv444p` frame is `(3, H, W)`, which is indistinguishable from a stack of three packed `yuv420p` frames — hence `from_format` for that one case. The method form never needs it.

### One shape caveat

With the default `pixel_format=None`, `shape` reports the layout of `frame.to_ndarray()` in the stream's *native* format. For a 480x640 `yuv420p` video that is `(n, 720, 640)`, because the packed layout stacks the colour planes underneath the luma plane. `frame_shape` is `(480, 640)` either way:

```python
from asyncvideo import VideoHandler

with VideoHandler("example.mp4") as video:      # pixel_format=None
    print(video.shape)        # (100, 720, 640)  — packed Y + U + V
    print(video.frame_shape)  # (480, 640)       — actual frame size
```

## Supported formats

Support means *covered by the test suite*. `VideoHandler` is tested against every combination below; `AsyncVideoReader` is currently tested against H.264 in MP4 only.

| Codec | Container |
|---|---|
| H.264 (`libx264`) | `.mp4`, `.mkv` |
| H.265 (`libx265`) | `.mp4` |
| MPEG-4 (`mpeg4`) | `.mp4`, `.avi` |
| VP9 (`vp9`) | `.webm` |

Other codecs may work, since nothing here is codec-specific, but they are not verified. Codecs whose packet order differs from display order are the most likely to have seeking problems. AV1 is known not to work correctly yet. If you need a format that is not listed, please open an issue with a sample file.

## License

MIT — see [LICENSE](LICENSE).

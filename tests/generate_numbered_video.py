import pathlib
from fractions import Fraction

import av
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# (codec, extension) pairs to generate by default.
# Covers both B-frame codecs (mpeg4, libx264) and non-B-frame codecs (vp9, libtheora).
DEFAULT_COMBOS = [
    ("mpeg4", "mp4"),
    ("libx264", "mp4"),
    ("libx264", "mkv"),
    ("mpeg4", "avi"),
    ("vp9", "webm"),
    ("libx265", "mp4"),
]


def generate_numbered_video(
    base_name: str = "numbered_video",
    extension: str = "mp4",
    codec: str = "mpeg4",
    num_frames: int = 100,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
):
    """Generate a test video where each frame displays its frame index.

    The output file is written to ``tests/test_video/{base_name}_{codec}.{extension}``.

    Parameters
    ----------
    base_name:
        Stem prefix for the output filename.
    extension:
        Container format extension (e.g. ``"mp4"``, ``"mkv"``, ``"avi"``).
    codec:
        PyAV codec name (e.g. ``"mpeg4"``, ``"libx264"``, ``"vp9"``).
    num_frames:
        Number of frames to encode.
    fps:
        Frames per second.
    width:
        Frame width in pixels.
    height:
        Frame height in pixels.
    """
    output_path = (
        pathlib.Path(__file__).resolve().parent
        / "test_video"
        / f"{base_name}_{codec}.{extension}"
    )
    output_path.parent.mkdir(exist_ok=True)

    with av.open(str(output_path), mode="w") as container:
        stream = container.add_stream(codec, rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"

        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        canvas = FigureCanvas(fig)
        ax.axis("off")

        for i in range(num_frames):
            ax.clear()
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                str(i),
                fontsize=60,
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

            canvas.draw()
            buf = np.asarray(canvas.buffer_rgba())[:, :, :3]  # RGB

            frame = av.VideoFrame.from_ndarray(buf, format="rgb24")
            frame = frame.reformat(width, height, format="yuv420p")

            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

    plt.close(fig)
    print(f"Saved video to {output_path}")


def variable_frame_intervals(num_frames: int = 100) -> np.ndarray:
    """Frame timestamps, in seconds, whose spacing is deliberately not uniform.

    Alternating short and long gaps, so no constant frame rate reproduces them.
    A reader that invents timestamps from the nominal rate cannot match these,
    which is the point: it makes the difference observable.
    """
    gaps = np.where(np.arange(num_frames - 1) % 2 == 0, 0.020, 0.060)
    return np.concatenate([[0.0], np.cumsum(gaps)])


def generate_variable_rate_video(
    base_name: str = "variable_rate_video",
    extension: str = "mp4",
    codec: str = "libx264",
    num_frames: int = 100,
    width: int = 640,
    height: int = 480,
    time_base_den: int = 1000,
):
    """Generate a numbered video whose frames are unevenly spaced in time.

    Every other gap is 20 ms and the rest are 60 ms, so the presentation
    timestamps cannot be reproduced by any single frame rate. Timestamps are
    assigned explicitly rather than left to the encoder.

    Parameters
    ----------
    time_base_den :
        Denominator of the stream time base, i.e. timestamp resolution. 1000
        gives millisecond ticks, which represents the gaps above exactly.
    """
    output_path = (
        pathlib.Path(__file__).resolve().parent
        / "test_video"
        / f"{base_name}_{codec}.{extension}"
    )
    output_path.parent.mkdir(exist_ok=True)

    times = variable_frame_intervals(num_frames)
    time_base = Fraction(1, time_base_den)

    with av.open(str(output_path), mode="w") as container:
        # rate is only metadata here; the per-frame pts below decide the timing
        stream = container.add_stream(codec, rate=30)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = time_base

        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        canvas = FigureCanvas(fig)
        ax.axis("off")

        for i in range(num_frames):
            ax.clear()
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                str(i),
                fontsize=60,
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

            canvas.draw()
            buf = np.asarray(canvas.buffer_rgba())[:, :, :3]  # RGB

            frame = av.VideoFrame.from_ndarray(buf, format="rgb24")
            frame = frame.reformat(width, height, format="yuv420p")
            frame.time_base = time_base
            frame.pts = round(times[i] / time_base)

            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

    plt.close(fig)
    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    for _codec, _ext in DEFAULT_COMBOS:
        generate_numbered_video(
            base_name="numbered_video", extension=_ext, codec=_codec
        )
    generate_variable_rate_video()

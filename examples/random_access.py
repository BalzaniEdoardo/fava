"""The same frames, addressed two ways: by number and by timestamp.

Both rows below show the same five frames. The top row asks for them by position,
``video[i]``. The bottom row asks by time, ``video.get(t)``, using the frame
timestamps recorded by the acquisition system. They agree, which is the point:
whichever way you think about a recording, you get the same pixels.

Run it::

    python examples/random_access.py

The clip is downloaded on first run, a few MB. See ``asyncvideo.fetch`` for the
data's licence and citations.
"""

import matplotlib.pyplot as plt

from asyncvideo import VideoHandler
from asyncvideo.fetch import fetch_times, fetch_video

CAMERA = "left"
# spread across the clip: the left camera runs at 60 fps, so these span ~9 seconds
FRAMES = [0, 120, 240, 360, 540]
OUT_PATH = "random_access.png"  # written next to wherever you run this

# time= gives the reader the acquisition system's clock, so get() takes a
# timestamp on that clock rather than seconds-from-the-start.
with VideoHandler(
    fetch_video(CAMERA), time=fetch_times(CAMERA), pixel_format="rgb24"
) as video:
    print(f"{CAMERA} camera: {video.shape} frames x height x width x RGB")
    # These timestamps are on the experiment clock, so they do not start at zero:
    # this clip was cut from two minutes into the session.
    print(f"first timestamps: {video.time[:4].round(3)}  ...  last {video.time[-1]:.2f}")

    by_number = [video[i] for i in FRAMES]

    # video.time is one timestamp per frame, so time[i] is the moment frame i was
    # captured -- asking for it should return frame i again.
    timestamps = [video.time[i] for i in FRAMES]
    by_time = [video.get(t) for t in timestamps]

# constrained layout, so the row titles and the suptitle get room instead of
# being clipped
fig, axes = plt.subplots(
    2,
    len(FRAMES),
    figsize=(2.4 * len(FRAMES), 4.8),
    dpi=140,
    layout="constrained",
)
for col, (index, timestamp) in enumerate(zip(FRAMES, timestamps)):
    axes[0, col].imshow(by_number[col])
    axes[0, col].set_title(f"video[{index}]", fontsize=9, family="monospace")
    axes[1, col].imshow(by_time[col])
    axes[1, col].set_title(f"get({timestamp:.2f})", fontsize=9, family="monospace")

for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
axes[0, 0].set_ylabel("by number", fontsize=10)
axes[1, 0].set_ylabel("by time", fontsize=10)

fig.suptitle("Same frames, addressed by number and by timestamp", fontsize=12)
fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"wrote {OUT_PATH}")

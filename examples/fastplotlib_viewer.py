"""Play a video in fastplotlib without ever blocking the render loop.

The point of `AsyncVideoReader` is that requesting a frame returns immediately, so
the interesting part here is what ``update`` does *not* do: it never calls
``future.result()`` on a future that is still pending. It checks whether the frame
has arrived, draws it if so, and otherwise returns and lets the next render tick
try again. Blocking on ``result()`` inside a render callback would stall the
window, which is exactly the problem the async reader exists to avoid.

Run it::

    pip install -e ".[docs]"
    python examples/fastplotlib_viewer.py

The clip is downloaded on first run, a few MB. See ``asyncvideo.fetch`` for the
data's licence and citations.
"""

import fastplotlib as fpl

from asyncvideo import AsyncVideoReader
from asyncvideo.fetch import fetch_video

CAMERA = "body"

# fetch video and get the path
path = fetch_video(CAMERA)
reader = AsyncVideoReader(path)
n_frames = reader.shape[0]

figure = fpl.Figure(size=(700, 560))
# seed the graphic with frame 0. to_rgb converts the reader's native YUV output,
# and [0] drops the leading single-frame axis.
image = figure[0, 0].add_image(reader.to_rgb(reader[0].result())[0])

# One request is in flight at a time. ``pending`` is that request, or None.
state = {"frame": 0, "pending": None}


def update(figure):
    """Advance one frame per render tick, without ever waiting for a decode."""
    pending = state["pending"]

    # still decoding: draw nothing, try again on the next tick
    if pending is not None and not pending.done():
        return

    # the frame arrived, so show it
    if pending is not None:
        image.data = reader.to_rgb(pending.result())[0]

    # ask for the next one and return immediately
    state["frame"] = (state["frame"] + 1) % n_frames
    state["pending"] = reader[state["frame"]]


figure.add_animations(update)

try:
    figure.show()
    fpl.loop.run()
finally:
    # the reader owns a process, so it has to be shut down
    reader.shutdown()

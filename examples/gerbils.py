import numpy as np
from pathlib import Path

import fastplotlib as fpl
from fastplotlib.widgets.nd_widget import VideoProcessor
import pyinstrument
from fava import AsyncVideoReader

adapter = fpl.enumerate_adapters()[1]
print(adapter.info)
fpl.select_adapter(adapter)


paths = sorted(Path("/home/kushal/data/gerbils/").glob("*.mp4"))
# paths = sorted(Path("/home/kushal/data/alyx/cortexlab/Subjects/SP058/2024-07-18/raw_video_data/").glob("*.mp4"))
paths.extend(
    sorted(
        Path(
            "/home/kushal/data/alyx/cortexlab/Subjects/SP058/2024-07-18/raw_video_data/"
        ).glob("*.mp4")
    )
)

vrs = list()
for p in paths:
    vrs.append(AsyncVideoReader(p, yuv_packed=False))

ref_ranges = {"t": (0, vrs[0].shape[0], 1)}
ndw = fpl.NDWidget(
    ref_ranges=ref_ranges,
    shape=(1, len(vrs)),
    size=(1800, 500),
    canvas_kwargs={"max_fps": 999},
)


# print(vrs[0][(slice(0, 1),)].result().shape)


# def mean(data, axis, keepdims):
#     return np.mean(data, axis=axis, keepdims=keepdims).astype(np.uint8, copy=False)


for i, vr in enumerate(vrs):
    ndw[0, i].add_nd_image(
        vr,
        dims=list("tmn"),
        spatial_dims=list("mn"),
        # window_funcs={"t": (mean, 11)},
        # window_order=("t",),
        # spatial_func=lambda img: img[::2, ::2],
        compute_histogram=False,
        colorspace="yuv420p",
        colorrange="full",
        processor_type=VideoProcessor,
    )

ndw.show()
# ndw.indices["t"] = 6000
ndw._sliders_ui._playing["t"] = True
ndw.figure.imgui_show_fps = True
run_profile = False

if run_profile:

    with pyinstrument.Profiler(async_mode="enabled") as profiler:
        fpl.loop.run()

    profiler.print()
    profiler.open_in_browser()

else:
    fpl.loop.run()

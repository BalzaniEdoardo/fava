from importlib.metadata import PackageNotFoundError, version

from ._pyav_video_reader import VideoHandler, pyav_trim_plane
from .convert import to_rgb
from .vr_async import AsyncVideoReader

try:
    # set at build time by setuptools_scm, from the git tag
    __version__ = version("asyncvideo")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "unknown"

__all__ = [
    "AsyncVideoReader",
    "VideoHandler",
    "__version__",
    "pyav_trim_plane",
    "to_rgb",
]
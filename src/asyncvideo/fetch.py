"""Download the example videos.

The examples use short clips of real multi-camera recordings, hosted on OSF. This
module fetches them on demand and caches them, so an example is two lines rather
than a page of download code.

Nothing here is needed to use `asyncvideo` on your own videos. It exists so the
examples can be short, and it is kept in the package rather than hidden in a
script so that the download is inspectable.

`pooch` does the work: it verifies each file against a known SHA256, so a
truncated or tampered download fails loudly instead of producing a video that
decodes strangely. Both `pooch` and `tqdm` are optional::

    pip install asyncvideo[docs]

Set ``ASYNCVIDEO_DATA_DIR`` to control where files are cached; otherwise pooch
picks the platform's cache directory.

The clips are third-party data under a different licence from this package. See
`DATA_ATTRIBUTION` for the licence, the fact that the files are modified, and the
citations their creators ask for.
"""

from __future__ import annotations

import hashlib
import pathlib

import numpy as np
from numpy.typing import NDArray

try:
    import pooch
except ImportError:  # pragma: no cover - exercised by the error path below
    pooch = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

__all__ = ["DATA_ATTRIBUTION", "available_examples", "fetch_times", "fetch_video"]

#: Attribution for the example clips, as CC-BY 4.0 requires: who made the data,
#: the licence and a link to it, a statement that the files were modified, and the
#: citations the creators ask for. Note the clips are *not* covered by this
#: package's MIT licence.
DATA_ATTRIBUTION = """\
The example clips are derived from public data of the International Brain
Laboratory (IBL), session ebce500b-c530-47de-8cb1-963c552703ea of the
`ibl_neuropixel_brainwide_01` project, obtained from
openalyx.internationalbrainlab.org.

Licence: CC-BY 4.0 — https://creativecommons.org/licenses/by/4.0/
This package's own MIT licence does not apply to these files.

Modifications: the original camera videos are 2.25-13.52 GB each and several
hours long. Each clip here is the opening few seconds only, copied without
re-encoding, and each timestamp array has been truncated to match.

If you use this data, IBL asks that you cite:
  International Brain Laboratory et al. (2025). A brain-wide map of neural
  activity during complex behaviour. Nature.
  https://www.nature.com/articles/s41586-025-09235-0
  and the accompanying technical paper:
  https://doi.org/10.6084/m9.figshare.21400815
"""

OSF_TEMPLATE = "https://osf.io/download/{}/"

_ENV_VAR = "ASYNCVIDEO_DATA_DIR"

# Clips from one public IBL session, cut to a few seconds each. The three cameras
# record simultaneously at different frame rates, which is what makes them a good
# example: the same instant is a different frame number in each video.
#
# Each video ships with a .npy of its per-frame timestamps, on the session clock.
#
# See DATA_ATTRIBUTION below for the licence and required citations.
#
# Cameras are addressed by name; each has a video and a timestamps file.
CAMERAS = ("left", "body", "right")

# OSF keys, from the IBL-Snippets folder of https://osf.io/gpj9w/. Regenerate the
# files with ``_scripts/make_ibl_snippet.py``, which prints the hashes below.
_OSF_KEYS: dict[str, str] = {
    "ibl_left.mp4": "6a6a6ad32fa77146790e4e6d",
    "ibl_left_times.npy": "6a6a6ad211089e41be0e4e93",
    "ibl_body.mp4": "mwnsv",
    "ibl_body_times.npy": "6a6a6acf11089e41be0e4e8f",
    "ibl_right.mp4": "6a6a6ad39164aabbe20e4db2",
    "ibl_right_times.npy": "6a6a6acfe5c22542ed22f1c6",
}

_HASHES: dict[str, str] = {
    "ibl_left.mp4": "4b71573cebe4b3ac2384cc9aeecb314a40c39f80f3feebdfd96082de67bb7e83",
    "ibl_left_times.npy": "8f66199bced933687cb6003981bc2bc92402b8c310775a5324eaa433b45ed118",
    "ibl_body.mp4": "0ea780e82236f4d00a693303e173aef37081b6ee98ac0884ebfe63c218205aa6",
    "ibl_body_times.npy": "564cf62ac6974684cda830a943b56df3db48a6fede542174f59bda0852decafd",
    "ibl_right.mp4": "d3a6459fe3b09d67894fbe12b9b1871958852e4e989f4e44a97135ca901120ba",
    "ibl_right_times.npy": "c4129ae5c45eeecdd1f374430df22f803a9da6ccd2f9ebd8aed0a0f8d8afb182",
}


def available_examples() -> tuple[str, ...]:
    """Camera names accepted by `fetch_video` and `fetch_times`."""
    return CAMERAS


def _filename(camera: str, kind: str) -> str:
    if camera not in CAMERAS:
        raise KeyError(f"unknown camera {camera!r}; available: {', '.join(CAMERAS)}")
    return f"ibl_{camera}.mp4" if kind == "video" else f"ibl_{camera}_times.npy"


def sha256(path: str | pathlib.Path, chunk_size: int = 1 << 20) -> str:
    """Hash a local file, to fill in the registry after uploading a new clip."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require_pooch() -> None:
    missing = [name for name, mod in (("pooch", pooch), ("tqdm", tqdm)) if mod is None]
    if missing:
        raise ImportError(
            f"fetching example data needs {' and '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}"
        )


def _retriever(path: pathlib.Path | str | None):
    _require_pooch()
    unregistered = [name for name, key in _OSF_KEYS.items() if not key]
    if unregistered:
        raise RuntimeError(
            f"no download location recorded for {len(unregistered)} example "
            f"file(s): {', '.join(unregistered)}. The clips have not been "
            f"published yet."
        )
    return pooch.create(
        path=path if path is not None else pooch.os_cache("asyncvideo"),
        base_url="",
        urls={name: OSF_TEMPLATE.format(key) for name, key in _OSF_KEYS.items()},
        registry=dict(_HASHES),
        retry_if_failed=2,
        env=_ENV_VAR,
    )


def _fetch(name: str, path: pathlib.Path | str | None) -> pathlib.Path:
    return pathlib.Path(_retriever(path).fetch(name, progressbar=tqdm is not None))


def fetch_video(camera: str, path: pathlib.Path | str | None = None) -> pathlib.Path:
    """
    Download one example video, or return it from the cache.

    Parameters
    ----------
    camera :
        Which camera: ``"left"``, ``"body"`` or ``"right"``. They record the same
        session at 60, 30 and 150 fps respectively.
    path :
        Directory to cache in. Defaults to the platform cache directory, or
        ``$ASYNCVIDEO_DATA_DIR`` when set.

    Returns
    -------
    :
        Path to the local video file, ready to pass to a reader.

    Examples
    --------
    >>> from asyncvideo import VideoHandler  # doctest: +SKIP
    >>> from asyncvideo.fetch import fetch_video  # doctest: +SKIP
    >>> with VideoHandler(fetch_video("body"), pixel_format="rgb24") as video:  # doctest: +SKIP
    ...     frame = video[42]
    """
    return _fetch(_filename(camera, "video"), path)


def fetch_times(camera: str, path: pathlib.Path | str | None = None) -> NDArray:
    """
    Download one camera's frame timestamps, or read them from the cache.

    These are the times the acquisition system recorded for each frame, in
    seconds, one per frame of the matching clip. Pass them as ``time=`` so
    lookups use that clock instead of a nominal frame rate.

    Parameters
    ----------
    camera :
        Which camera: ``"left"``, ``"body"`` or ``"right"``.
    path :
        Directory to cache in. Defaults to the platform cache directory, or
        ``$ASYNCVIDEO_DATA_DIR`` when set.

    Returns
    -------
    :
        One timestamp per frame, in seconds.

    Examples
    --------
    >>> from asyncvideo import VideoHandler  # doctest: +SKIP
    >>> from asyncvideo.fetch import fetch_times, fetch_video  # doctest: +SKIP
    >>> with VideoHandler(  # doctest: +SKIP
    ...     fetch_video("body"), time=fetch_times("body"), pixel_format="rgb24"
    ... ) as video:
    ...     frame = video.get(1.5)
    """
    return np.load(_fetch(_filename(camera, "times"), path))

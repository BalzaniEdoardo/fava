import importlib.metadata as md
from typing import Literal

def has_package(name):
    try:
        md.version(name)
        return True
    except md.PackageNotFoundError:
        return False

class Config:
    """Configuration for the brainhack package."""
    def __init__(self):
        if has_package("decord"):
           self._backend = "decord"
        else:
            self._backend = "av"

    @property
    def backend(self):
        return self._backend

    def update(self, backend: Literal["decord", "av"] = "decord"):
        """Update configuration settings.

        Parameters
        ----------
        root_folder : str or Path, optional
            The root folder path for datasets.
        """
        if not backend in ["decord", "av"]:
            raise ValueError("backend must be 'decord', or 'av' for decord "
                             "or pyav respectively.")
        self._backend = backend

config = Config()
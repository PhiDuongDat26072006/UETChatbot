"""Ingestion v1: standalone staged ingestion. No eager OCR/model imports."""

__version__ = "1.0.0"


def __getattr__(name):
    if name == "DataManifestTracker":
        from .manifest import DataManifestTracker

        return DataManifestTracker
    if name == "clean_whitespace":
        from .utils import clean_whitespace

        return clean_whitespace
    if name == "UETDataLoader":
        from .loader import UETDataLoader

        return UETDataLoader
    raise AttributeError(name)

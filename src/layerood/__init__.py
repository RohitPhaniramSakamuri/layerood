"""Local embeddings and ID-calibrated out-of-distribution detection."""

__version__ = "0.1.0"

from .artifacts import EmbeddingBatch, load_embeddings
from .detector import LayerOOD, DetectorConfig

__all__ = ["EmbeddingBatch", "load_embeddings", "LayerOOD", "DetectorConfig"]

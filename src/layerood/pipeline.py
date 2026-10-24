"""Load once, embed locally, and route to the saved detector."""
from dataclasses import replace
from pathlib import Path
import threading

from .detector import LayerOOD
from .embeddings import EncoderConfig, LocalEncoder


class OODPipeline:
    def __init__(self, detector: LayerOOD, local_files_only=False):
        config = dict(detector.metadata["encoder"]["encoder_config"])
        if config.get("layers") is not None:
            config["layers"] = tuple(config["layers"])
        resolved = detector.metadata["encoder"].get("resolved_revision")
        if resolved and not resolved.startswith("local-sha256:"):
            config["revision"] = resolved
        config["local_files_only"] = local_files_only
        self.detector = detector
        self.encoder = LocalEncoder(EncoderConfig(**config))
        if self.encoder.signature != detector.metadata["embedding_signature"]:
            raise ValueError("Saved model/tokenizer identity differs from the current encoder")
        self._lock = threading.Lock()

    @classmethod
    def load(cls, path, local_files_only=False):
        return cls(LayerOOD.load(path), local_files_only)

    def predict(self, texts: list[str], alpha=.05):
        with self._lock:
            batch = self.encoder.encode([{"id":f"request-{i}","text":text} for i,text in enumerate(texts)])
            return self.detector.score(batch,alpha)

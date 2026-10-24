"""Portable, non-pickle artifacts with explicit content and model identities."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile
import unicodedata

import numpy as np


def text_hash(text: str) -> str:
    canonical = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(canonical.encode()).hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_jsonl(path: str | Path) -> list[dict]:
    rows, ids = [], set()
    with Path(path).open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError(f"{path}:{line_no}: expected an object with nonempty text")
            row = dict(row)
            row["id"] = str(row.get("id", f"row-{line_no}"))
            if row["id"] in ids:
                raise ValueError(f"Duplicate id {row['id']!r} in {path}")
            if "is_ood" in row and type(row["is_ood"]) is not bool:
                raise ValueError("is_ood must be a JSON boolean")
            ids.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError(f"No text records in {path}")
    return rows


def write_json(path, value, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8") as out:
        json.dump(value, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write("\n")


def save_npz(path, arrays: dict, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Artifact exists: {path}; choose a new path or --overwrite")
    fd, temporary = tempfile.mkstemp(prefix=".layerood-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            np.savez_compressed(out, **arrays)
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            os.unlink(temporary)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass
class EmbeddingBatch:
    values: np.ndarray  # examples, selected layers, hidden dimension
    records: list[dict]
    metadata: dict

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=np.float32)
        if self.values.ndim != 3 or len(self.values) != len(self.records) or not len(self.values):
            raise ValueError("Embeddings must have shape (nonempty examples, layers, hidden_dim)")
        if self.values.shape[1] < 2 or self.values.shape[2] < 1 or not self.metadata.get("embedding_signature"):
            raise ValueError("Need >=2 layers, positive hidden width and an embedding signature")
        if not np.isfinite(self.values).all():
            raise ValueError("Embedding values must be finite")
        if len(self.metadata.get("layers", [])) != self.values.shape[1]:
            raise ValueError("Layer metadata does not match embedding shape")
        ids = [r["id"] for r in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("Embedding record ids must be unique")
        for r in self.records:
            if not r.get("text_hash") or r.get("token_count", 0) < 1:
                raise ValueError("Every record needs text_hash and positive token_count")

    @property
    def signature(self):
        return self.metadata["embedding_signature"]

    @property
    def hashes(self):
        return [r["text_hash"] for r in self.records]

    @property
    def lengths(self):
        return np.asarray([r["token_count"] for r in self.records], dtype=float)

    def save(self, path, overwrite=False):
        save_npz(path, {"values": self.values,
                        "records": np.asarray(json.dumps(self.records)),
                        "metadata": np.asarray(json.dumps(self.metadata))}, overwrite)


def load_embeddings(path) -> EmbeddingBatch:
    with np.load(path, allow_pickle=False) as stored:
        return EmbeddingBatch(stored["values"], json.loads(str(stored["records"])),
                              json.loads(str(stored["metadata"])))


def assert_disjoint(*batches: EmbeddingBatch):
    seen = set()
    for b in batches:
        current = set(b.hashes)
        if len(current) != len(b.hashes):
            raise ValueError("Duplicate normalized texts within a split; deduplicate before fitting")
        if seen & current:
            raise ValueError("Text overlap between fit/reference/calibration splits")
        seen |= current

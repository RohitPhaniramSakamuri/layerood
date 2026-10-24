"""Extract pooled hidden states without allocating language-model vocabulary logits."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np

from .artifacts import EmbeddingBatch, digest, save_npz, text_hash


@dataclass(frozen=True)
class EncoderConfig:
    model: str = "HuggingFaceTB/SmolLM2-135M"
    revision: str = "main"
    layers: tuple[int, ...] | None = None
    pooling: str = "mean"
    max_length: int = 128
    batch_size: int = 4
    threads: int = 4
    device: str = "cpu"
    local_files_only: bool = False
    cache_dir: str | None = ".cache/embeddings"


class LocalEncoder:
    def __init__(self, config: EncoderConfig | None = None):
        self.config = config or EncoderConfig()
        c = self.config
        if c.pooling not in ("mean", "last") or c.max_length < 2 or c.batch_size < 1 or c.threads < 1:
            raise ValueError("Use mean/last pooling, max_length >= 2, and positive batch_size/threads")
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(c.threads)
        self.torch = torch
        started = time.perf_counter()
        kwargs = dict(revision=c.revision, local_files_only=c.local_files_only, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(c.model, torch_dtype=torch.float32,
                                                use_safetensors=True, **kwargs).to(c.device).eval()
        resolved = getattr(self.model.config, "_commit_hash", None)
        tokenizer_kwargs = {**kwargs, "revision": resolved or c.revision}
        self.tokenizer = AutoTokenizer.from_pretrained(c.model, **tokenizer_kwargs)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer needs a pad or EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model.requires_grad_(False)
        depth = getattr(self.model.config, "num_hidden_layers", None)
        if not isinstance(depth, int) or depth < 3:
            raise ValueError("Model must expose num_hidden_layers >= 3 and hidden_states")
        requested = c.layers or (max(1, depth // 3), max(2, 2 * depth // 3), depth)
        self.layers = tuple(x if x >= 0 else depth + 1 + x for x in requested)
        if len(self.layers) < 2 or tuple(sorted(set(self.layers))) != self.layers or any(x < 1 or x > depth for x in self.layers):
            raise ValueError(f"Layers must be >=2 distinct ascending transformer outputs in 1..{depth}")
        # Local model weights are fingerprinted too: a directory name alone is not an identity.
        if resolved is None and Path(c.model).is_dir():
            import hashlib
            h = hashlib.sha256()
            for p in sorted(Path(c.model).glob("*.safetensors")):
                with p.open("rb") as f:
                    for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                        h.update(block)
            resolved = "local-sha256:" + h.hexdigest()
        self.identity = {"model": c.model, "resolved_revision": resolved or c.revision,
                         "layers": list(self.layers), "pooling": c.pooling, "max_length": c.max_length,
                         "dtype": "float32", "add_special_tokens": True,
                         "tokenizer_hash": digest({"backend": self.tokenizer.backend_tokenizer.to_str(),
                                                   "special_tokens": self.tokenizer.special_tokens_map,
                                                   "padding_side": self.tokenizer.padding_side,
                                                   "truncation_side": self.tokenizer.truncation_side}),
                         "model_config_hash": digest(self.model.config.to_dict()), "format": 2}
        self.signature = digest(self.identity)
        self.load_seconds = time.perf_counter() - started

    def encode(self, rows: list[dict]) -> EmbeddingBatch:
        if not rows:
            raise ValueError("Provide at least one text")
        started = time.perf_counter()
        torch, c = self.torch, self.config
        values, records, pending = [None] * len(rows), [None] * len(rows), []
        cache_hits = 0
        for i, row in enumerate(rows):
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError("Each record requires nonempty text")
            # Exact text in the cache key: whitespace/case can change model features.
            cache_key = digest({"encoder": self.signature, "text": row["text"]})
            cache = Path(c.cache_dir) / self.signature / (cache_key + ".npz") if c.cache_dir else None
            if cache is not None and cache.exists():
                with np.load(cache, allow_pickle=False) as saved:
                    values[i] = saved["values"]
                    info = json.loads(str(saved["info"]))
                cache_hits += 1
                records[i] = self._record(row, i, info)
            else:
                pending.append((i, row, cache))
        for start in range(0, len(pending), c.batch_size):
            chunk = pending[start:start+c.batch_size]
            texts = [r[1]["text"] for r in chunk]
            raw = self.tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]
            encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=c.max_length,
                                     return_tensors="pt", add_special_tokens=True).to(c.device)
            mask = encoded["attention_mask"]
            with torch.inference_mode():
                output = self.model(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
                layers = []
                for index in self.layers:
                    hidden = output.hidden_states[index]
                    if c.pooling == "mean":
                        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
                    else:
                        pooled = hidden[torch.arange(len(chunk), device=c.device), mask.sum(1)-1]
                    layers.append(pooled.float().cpu().numpy())
                batch = np.stack(layers, axis=1)
            counts = mask.sum(1).cpu().tolist()
            for offset, (i, row, cache) in enumerate(chunk):
                info = {"token_count": int(counts[offset]), "original_token_count": len(raw[offset]),
                        "truncated": len(raw[offset]) > c.max_length}
                values[i], records[i] = batch[offset], self._record(row, i, info)
                if cache is not None:
                    try:
                        save_npz(cache, {"values": values[i], "info": np.asarray(json.dumps(info))})
                    except FileExistsError:
                        pass
            del output
        metadata = {**self.identity, "embedding_signature": self.signature,
                    "encoder_config": asdict(c), "model_load_seconds": self.load_seconds,
                    "extraction_seconds": time.perf_counter()-started, "cache_hits": cache_hits,
                    "n_records": len(rows), "device": c.device}
        return EmbeddingBatch(np.stack(values), records, metadata)

    @staticmethod
    def _record(row, i, info):
        keep = {k: row[k] for k in ("label", "is_ood", "group", "split") if k in row}
        return {"id": str(row.get("id", f"row-{i}")), "text_hash": text_hash(row["text"]), **keep, **info}

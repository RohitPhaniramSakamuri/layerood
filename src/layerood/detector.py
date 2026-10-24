"""ID-only conditional residual detector, comparison scores and split calibration.

Higher scores mean more anomalous. p_id is an ID-calibrated upper-tail rank,
not P(ID | text). Fitting never reads OOD evaluation labels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import time

import numpy as np
from sklearn.covariance import LedoitWolf
from threadpoolctl import threadpool_limits

from .artifacts import EmbeddingBatch, assert_disjoint, save_npz


@dataclass(frozen=True)
class DetectorConfig:
    rank: int = 32
    neighbors: int = 10
    ridge_alpha: float = 10.0
    variance_shrinkage: float = 0.5
    query_chunk: int = 128
    threads: int = 4


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def _quadratic(x, mean, precision):
    delta = x - mean
    return np.maximum(np.einsum("ij,jk,ik->i", delta, precision, delta), 0)


class LayerOOD:
    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        c = self.config
        if c.rank < 1 or c.neighbors < 2 or c.ridge_alpha <= 0 or not 0 < c.variance_shrinkage <= 1 or c.query_chunk < 1 or c.threads < 1:
            raise ValueError("rank/chunk/threads positive, neighbors >=2, ridge_alpha >0, shrinkage in (0,1]")
        self.state: dict[str, np.ndarray] = {}
        self.metadata: dict = {}

    def _gaussian(self, name, x):
        fit = LedoitWolf().fit(x)
        self.state[name+"_mean"] = fit.location_
        self.state[name+"_precision"] = fit.precision_

    def _maha(self, name, x):
        return _quadratic(x, self.state[name+"_mean"], self.state[name+"_precision"])

    def _project(self, values):
        x = _unit(np.asarray(values, dtype=float))
        result = []
        for l in range(x.shape[1]):
            result.append((x[:, l] - self.state[f"pca_mean_{l}"]) @ self.state[f"pca_basis_{l}"] / self.state[f"pca_scale_{l}"])
        return x, result

    def _residual(self, projected, lengths):
        length = (np.log1p(lengths)-self.state["length_mean"])/self.state["length_scale"]
        return np.concatenate([
            projected[i+1] - np.column_stack((projected[i], length, np.ones(len(length)))) @ self.state[f"transport_{i}"]
            for i in range(len(projected)-1)
        ], axis=1)

    def _neighbors(self, query, bank, exclude_self=False):
        n, k = len(query), self.metadata["effective_neighbors"]
        distances, indices = [], []
        for start in range(0, n, self.config.query_chunk):
            stop = min(n, start+self.config.query_chunk)
            sq = np.maximum(2 - 2 * query[start:stop] @ bank.T, 0)
            if exclude_self:
                sq[np.arange(stop-start), np.arange(start, stop)] = np.inf
            ix = np.argpartition(sq, k-1, axis=1)[:, :k]
            d = np.take_along_axis(sq, ix, axis=1)
            order = np.argsort(d, axis=1, kind="stable")
            indices.append(np.take_along_axis(ix, order, axis=1))
            distances.append(np.sqrt(np.take_along_axis(d, order, axis=1)))
        return np.concatenate(distances), np.concatenate(indices)

    def fit(self, train: EmbeddingBatch, reference: EmbeddingBatch, calibration: EmbeddingBatch):
        started = time.perf_counter()
        assert_disjoint(train, reference, calibration)
        if len(train.values) < 8 or len(reference.values) < 4 or len(calibration.values) < 20:
            raise ValueError("Need >=8 fit, >=4 reference, >=20 independent ID calibration texts")
        for b in (train, reference, calibration):
            if b.signature != train.signature or b.values.shape[1:] != train.values.shape[1:]:
                raise ValueError("All splits must use identical model/layers/pooling/truncation")
            if any(r.get("is_ood") is True for r in b.records):
                raise ValueError("Only ID records are allowed in fit, reference and calibration")
        self.state = {}
        self.metadata = {"format": 1, "config": asdict(self.config), "embedding_signature": train.signature,
                         "encoder": train.metadata, "effective_neighbors": min(self.config.neighbors, len(reference.values)-1),
                         "split_sizes": {"fit": len(train.values), "reference": len(reference.values), "calibration": len(calibration.values)},
                         "used_text_hashes": train.hashes + reference.hashes + calibration.hashes,
                         "reference_ids": [r["id"] for r in reference.records],
                         "reference_labels": [str(r.get("label", "")) for r in reference.records],
                         "hypothesis": "local conditional layer residual; unvalidated research candidate"}
        with threadpool_limits(limits=self.config.threads):
            x = _unit(np.asarray(train.values, dtype=float))
            for l in range(x.shape[1]):
                mean = x[:, l].mean(axis=0)
                _, singular, vt = np.linalg.svd(x[:, l]-mean, full_matrices=False)
                rank = min(self.config.rank, len(x)-2, int(np.sum(singular > max(singular[0]*1e-5, 1e-10))))
                if rank < 1:
                    raise ValueError("Fit embeddings have no nonconstant directions")
                self.state[f"pca_mean_{l}"] = mean
                self.state[f"pca_basis_{l}"] = vt[:rank].T
                self.state[f"pca_scale_{l}"] = np.maximum(singular[:rank] / np.sqrt(len(x)-1), 1e-6)
            _, z = self._project(train.values)
            self.state["length_mean"] = np.asarray(np.log1p(train.lengths).mean())
            self.state["length_scale"] = np.asarray(max(np.log1p(train.lengths).std(), .1))
            length = (np.log1p(train.lengths)-self.state["length_mean"])/self.state["length_scale"]
            for i in range(len(z)-1):
                a = np.column_stack((z[i], length, np.ones(len(length))))
                penalty = np.eye(a.shape[1]) * self.config.ridge_alpha
                penalty[-1, -1] = 0  # unpenalized intercept
                self.state[f"transport_{i}"] = np.linalg.solve(a.T @ a + penalty, a.T @ z[i+1])
            self._gaussian("final", z[-1])
            self._gaussian("concat", np.concatenate(z, axis=1))
            for l, projected in enumerate(z):
                self._gaussian(f"layer_{l}", projected)
            # Optional class-conditional RMD uses only training intent labels.
            labels = [r.get("label") for r in train.records]
            self.metadata["class_labels"] = []
            if all(isinstance(label, str) and label for label in labels) and len(set(labels)) > 1:
                classes = sorted(set(labels))
                ids = np.asarray([classes.index(label) for label in labels])
                means = np.stack([z[-1][ids == i].mean(0) for i in range(len(classes))])
                shared = LedoitWolf().fit(z[-1]-means[ids])
                self.state["class_means"] = means
                self.state["class_precision"] = shared.precision_
                self.metadata["class_labels"] = classes
            xr, zr = self._project(reference.values)
            self.state["bank_final"] = xr[:, -1]
            self.state["bank_middle"] = xr[:, len(zr)//2]
            residuals = self._residual(zr, reference.lengths)
            self.state["bank_residuals"] = residuals
            self.state["residual_var"] = np.maximum(residuals.var(0, ddof=1), 1e-6)
            self._gaussian("residual", residuals)
            d, _ = self._neighbors(self.state["bank_final"], self.state["bank_final"], exclude_self=True)
            self.state["bank_radii"] = np.maximum(d[:, -1], 1e-6)
            raw_ref = self._raw(reference, reference_mode=True)
            for name in ("local_density", "local_residual", "global_residual", "global_residual_diag"):
                logged = np.log1p(raw_ref[name])
                self.state[f"fusion_center_{name}"] = np.asarray(np.median(logged))
                self.state[f"fusion_scale_{name}"] = np.asarray(max(np.quantile(logged,.75)-np.quantile(logged,.25), .05))
            cal = self._with_fusion(self._raw(calibration))
            self.metadata["methods"] = list(cal)
            for name, scores in cal.items():
                self.state["cal_"+name] = np.sort(scores)
        self.metadata["fit_seconds"] = time.perf_counter()-started
        self.metadata["actual_ranks"] = [len(self.state[f"pca_scale_{i}"]) for i in range(x.shape[1])]
        return self

    def _raw(self, batch, reference_mode=False):
        x, z = self._project(batch.values)
        d, ix = self._neighbors(x[:, -1], self.state["bank_final"], reference_mode)
        _, mid = self._neighbors(x[:, len(z)//2], self.state["bank_middle"], reference_mode)
        residual = self._residual(z, batch.lengths)
        local_scores = np.empty(len(x))
        for start in range(0, len(x), self.config.query_chunk):
            stop = min(len(x), start+self.config.query_chunk)
            neighbors = self.state["bank_residuals"][mid[start:stop]]
            center = neighbors.mean(axis=1)
            local_var = neighbors.var(axis=1, ddof=1)
            shrink = self.config.variance_shrinkage
            variance = (1-shrink)*local_var + shrink*self.state["residual_var"]
            local_scores[start:stop] = np.mean((residual[start:stop]-center)**2 / np.maximum(variance, 1e-6), axis=1)
        result = {"knn": d[:, -1],
                  "local_density": d[:, -1] / np.maximum(self.state["bank_radii"][ix].mean(axis=1), 1e-6),
                  "mahalanobis": self._maha("final", z[-1]),
                  "mahalanobis_concat": self._maha("concat", np.concatenate(z, axis=1)),
                  "global_residual": self._maha("residual", residual),
                  "global_residual_diag": np.mean((residual-self.state["residual_mean"])**2 / self.state["residual_var"],axis=1),
                  "local_residual": local_scores,
                  "length_only": np.log1p(batch.lengths)}
        layer_distances = np.stack([self._maha(f"layer_{l}", zi) for l, zi in enumerate(z)], axis=1)
        result["tv_sparse"] = np.mean(np.abs(np.diff(layer_distances, axis=1)), axis=1)
        result["layer_mahalanobis_mean"] = layer_distances.mean(axis=1)
        discarded = []
        for l, projected in enumerate(z):
            reconstruction = (projected*self.state[f"pca_scale_{l}"]) @ self.state[f"pca_basis_{l}"].T
            discarded.append(np.mean((x[:, l]-self.state[f"pca_mean_{l}"]-reconstruction)**2, axis=1))
        result["pca_residual"] = np.mean(discarded, axis=0)
        if self.metadata.get("class_labels"):
            distances = np.stack([_quadratic(z[-1], mean, self.state["class_precision"])
                                  for mean in self.state["class_means"]], axis=1)
            result["relative_mahalanobis"] = distances.min(axis=1)-result["mahalanobis"]
        if any(not np.isfinite(v).all() for v in result.values()):
            raise ValueError("Non-finite detector score; inspect embeddings and fit conditioning")
        return result

    def _with_fusion(self, raw):
        scaled = {name: (np.log1p(raw[name])-self.state[f"fusion_center_{name}"]) / self.state[f"fusion_scale_{name}"]
                  for name in ("local_density", "local_residual", "global_residual", "global_residual_diag")}
        return {**raw, "global_fusion": np.maximum(scaled["local_density"], scaled["global_residual"]),
                "global_diag_fusion": np.maximum(scaled["local_density"], scaled["global_residual_diag"]),
                "local_transport": np.maximum(scaled["local_density"], scaled["local_residual"])}

    def score(self, batch: EmbeddingBatch, alpha: float = .05, reject_seen: bool = False) -> list[dict]:
        if not self.metadata.get("methods"):
            raise RuntimeError("Fit or load a detector first")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be strictly between 0 and 1")
        if batch.signature != self.metadata["embedding_signature"]:
            raise ValueError("Embedding signature differs from the fitted model/layer/pooling configuration")
        if reject_seen and set(batch.hashes) & set(self.metadata["used_text_hashes"]):
            raise ValueError("Evaluation text overlaps fit, reference or calibration; use a disjoint split")
        with threadpool_limits(limits=self.config.threads):
            scores = self._with_fusion(self._raw(batch))
        result = [{"id": r["id"], "token_count": r["token_count"], "truncated": r.get("truncated", False), "methods": {}} for r in batch.records]
        for method, values in scores.items():
            cal = self.state["cal_"+method]
            # Include ties in the tail for conservative split-conformal ranks.
            p = (1 + len(cal)-np.searchsorted(cal, values, side="left"))/(len(cal)+1)
            for i, value in enumerate(values):
                result[i]["methods"][method] = {"score": float(value), "p_id": float(p[i]), "flag_ood": bool(p[i] <= alpha)}
        return result

    def save(self, path, overwrite=False):
        if not self.metadata.get("methods"):
            raise RuntimeError("Fit the detector before saving")
        save_npz(path, {**self.state, "metadata_json": np.asarray(json.dumps(self.metadata))}, overwrite)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata_json"]))
            if metadata.get("format") != 1:
                raise ValueError("Unsupported detector artifact format")
            result = cls(DetectorConfig(**metadata["config"]))
            result.metadata = metadata
            result.state = {k: saved[k].copy() for k in saved.files if k != "metadata_json"}
        return result

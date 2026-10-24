# LayerOOD

A CPU-first Python package for extracting hidden-state embeddings from a small local language model and detecting requests outside a reference collection.

The package extracts three layers in one frozen model forward pass, caches the vectors, fits OOD scores from your known texts, and returns an ID-calibrated alert for new requests. It includes a Python API, CLI, optional HTTP service and CLINC150 data preparation.

The working hypothesis is **local conditional layer residuals**: an unfamiliar request may finish near familiar requests in embedding space, while reaching that representation differently from its nearest known neighbours. The proposed score checks that difference. This is a research candidate; the package does not establish state-of-the-art performance.

**Executed:** real SmolLM2-135M CPU extraction and a complete 125-text CLINC subset workflow (40 fit / 20 reference / 25 calibration / 40 evaluation). On its 20 ID and 20 OOD evaluation texts, `local_transport` AUROC was **0.7725**, versus **0.8050** for the matched global-diagonal fusion and **0.8950** for PCA residual. This small run does not support the locality hypothesis's superiority; no hyperparameters were changed to improve its result. [Execution record](experiments/000-package-example.md).

Build an installable wheel and source archive with `uv build`; the files are written to `dist/`. The local service implementation is included; no persistent server is left running. [Retained pilot results](results/clinc-mini-heldout-42/report.md) and their provenance are included in this repository.

## Run locally

```bash
git clone https://github.com/RohitPhaniramSakamuri/layerood.git
cd layerood
uv sync --locked

# Extract real SLM hidden states. The first command downloads model weights.
uv run layerood embed --input examples/requests.jsonl --output artifacts/my-embeddings.npz

# A small end-to-end demonstration using the public CLINC data.
uv run layerood prepare-clinc --profile mini --protocol heldout-intents --seed 42 --output data/my-clinc-mini
uv run layerood run --data data/my-clinc-mini --output artifacts/my-clinc-run

# Use the saved detector on new requests, with cached local weights.
uv run layerood predict --detector artifacts/my-clinc-run/detector.npz --input examples/requests.jsonl --output artifacts/my-predictions.json --local-files-only
```

The default model is `HuggingFaceTB/SmolLM2-135M`, CPU float32, batch size 4, 128 tokens and 4 numerical threads. Its model card identifies 135M parameters and an Apache-2.0 license. The model is a causal language model; the package uses its hidden states without generating answers. Model weights live in the Hugging Face cache; extracted vectors live under `.cache/embeddings/`. Internet is needed for first downloads; subsequent inference supports `--local-files-only`. Python 3.11–3.13 is supported, and the lockfile supplies CPU PyTorch wheels for this Linux environment. [Model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M).

For the original workspace checkout, start in `~/Documents/Research/local-llm-ood` instead of cloning. Environments, downloaded datasets, embeddings and build outputs are generated locally and excluded from Git. The `results/` directory retains the small pilot's JSON metrics, predictions and manifests without copying its downloaded text corpus.

Use `--model HuggingFaceTB/SmolLM2-360M` for a larger compatible SLM, or `--model sentence-transformers/all-MiniLM-L6-v2` for a compact sentence-encoder comparison. These alternatives are not automatically downloaded. The default retained layers are one-third, two-thirds and the final transformer output; index 0, the input embedding, is excluded. The final transformer output may include the architecture's final normalization. Pooling defaults to an attention-mask-aware mean including valid special tokens; `--pooling last` is also available.

`mini` and `quick` use small intent subsets. `full` uses the complete configured official pools, with a custom fit/reference partition and duplicate removal. A run is not directly comparable with a published score unless dataset, intents, supervision, encoder and evaluation protocol agree. A random held-out intent is not automatically a semantically near-OOD intent.

## Bring your own texts

One JSON object per line:

```json
{"id":"request-001","text":"Can you freeze my payment card?","label":"freeze_card","is_ood":false,"group":"banking"}
```

`id` and `text` identify the input (`id` is generated when absent). `label` is optional and enables class-relative Mahalanobis. `group` is optional for evaluation breakdowns. `is_ood` must be a JSON boolean when present and is required for evaluation. Labels are retained as metadata and do not enter the encoder. The detector's fit, reference and calibration inputs must contain only ID texts.

Prepare **four text-disjoint files**: `train.jsonl`, `reference.jsonl`, `calibration.jsonl`, `evaluation.jsonl`. Then `layerood run --data your-directory --output a-new-run-directory` executes the whole workflow. At minimum use 8 fit, 4 reference and 20 calibration texts; useful research needs substantially more. The default neighbour count is reduced when a bank is too small and the effective value is recorded.

Alternatively operate on cached vectors:

```bash
uv run layerood fit --train artifacts/my-clinc-run/train.npz --reference artifacts/my-clinc-run/reference.npz --calibration artifacts/my-clinc-run/calibration.npz --output artifacts/my-detector.npz
uv run layerood evaluate --detector artifacts/my-detector.npz --input artifacts/my-clinc-run/evaluation.npz --output artifacts/my-evaluation.json
uv run layerood inspect artifacts/my-detector.npz
```

Files are preserved by default. Individual artifact commands accept `--overwrite`; experiment directories must be new. Failed/interrupted runs preserve their partial artifacts. Model revision, tokenizer behaviour, model configuration, layers, pooling, truncation and dtype enter the embedding identity. A detector refuses incompatible embeddings. Normalized-text fingerprints catch repeats across splits; exact text is used in cache keys because whitespace and case can change embeddings.

## Python API

```python
from layerood import LayerOOD, load_embeddings
from layerood.pipeline import OODPipeline

detector = LayerOOD().fit(
    load_embeddings("train.npz"),
    load_embeddings("reference.npz"),
    load_embeddings("calibration.npz"),
)
detector.save("detector.npz")
pipeline = OODPipeline.load("detector.npz", local_files_only=True)
result = pipeline.predict(["Please cancel my flight reservation."], alpha=0.05)
print(result[0]["methods"]["local_transport"])
```

Each method returns `score` (larger is more anomalous), `p_id` (an upper-tail rank against held-out ID calibration) and `flag_ood` (`p_id <= alpha`). **`p_id` is not the probability that the request is ID.** Under exchangeable future ID data and a fixed scoring rule, split calibration provides a marginal false-alarm interpretation. Changing the ID population, dependent streams or imbalanced benchmark mixtures can break that interpretation. The JSON report always gives observed ID false alarms separately.

## Local service

```bash
uv sync --locked --extra serve
uv run layerood serve --detector artifacts/my-clinc-run/detector.npz --local-files-only
```

The service binds to `127.0.0.1:8765`; `/docs` provides the API interface. `GET /healthz` reports the model and methods. `POST /v1/ood` takes `{"texts":["Freeze my card"],"alpha":0.05}`. `POST /v1/embeddings` takes the same text list and returns the layer vectors and provenance. Requests are limited to 16 texts of at most 8,000 characters. This is a local research service; authentication and public hosting are outside v1.

## What is compared

| Score | Role |
|---|---|
| `knn` | kth-neighbour distance on unit-normalized final embeddings |
| `local_density` | Final-layer distance divided by neighbours' leave-one-out radii |
| `mahalanobis` | Shrinkage Mahalanobis on ID-fitted final-layer PCA coordinates |
| `mahalanobis_concat` | Shrinkage Mahalanobis on concatenated layer coordinates |
| `layer_mahalanobis_mean` | Mean per-layer Gaussian distance |
| `tv_sparse` | Mean absolute successive distance change; sparse-layer TV-score adaptation |
| `pca_residual` | Mean discarded feature energy across retained layers |
| `relative_mahalanobis` | Class-relative distance; available only with complete multi-class fit labels |
| `global_residual`, `global_residual_diag` | Layer-transport residual scored globally with full or diagonal covariance |
| `local_residual` | Residual scored against the middle-layer neighbour distribution |
| `global_fusion`, `global_diag_fusion` | Density combined with global residual scores |
| `local_transport` | Proposed density + locally conditioned residual score |
| `length_only` | Token-length confounding control |

These are declared implementations/adaptations, not reproductions of every cited benchmark. The **matched locality control is `global_diag_fusion`**, because it retains the diagonal covariance structure of `local_transport`. `mahalanobis_concat` is essential: a global linear conditional Gaussian can be equivalent to a joint Gaussian decomposition. Complexity alone does not establish a new signal.

Outputs include vectors, a portable non-pickle detector, per-text scores, `metrics.json`, `report.md` and `run.json`. Metrics cover OOD-positive AUROC/AUPR, observed ID false-alarm rate, OOD recall at the ID-calibrated threshold, and ID FPR at 95% OOD TPR. A fixed-bin token-length-matched analysis reports retained coverage and per-group results. All-method scoring time is recorded separately from extraction/model-load time; it is not a single-method latency benchmark.

## Research records

- [Cross-machine Codex handoff and development brief](HANDOFF.md)
- [Exact hypothesis and method](APPROACH.md)
- [Closest literature and novelty limits](outputs/literature-review.md), [reading provenance](outputs/literature-review.provenance.md)
- [References](REFERENCES.md) and [BibTeX](references.bib)
- [Planned benchmark protocol](experiments/001-local-transport.md)
- [Glossary](GLOSSARY.md) and [research branches](notes/research-branches.md)
- [Execution record](experiments/000-package-example.md)

No results from the separate vision adaptation report belong to this project. Test suites are not added or run without the user's request, following the workspace instructions. The small execution example records ordinary package usage; full multi-seed benchmark claims remain pending.

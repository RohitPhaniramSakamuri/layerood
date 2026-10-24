# Local package execution example

Experiment ID: 000. Status: completed, 24 October 2026.

Purpose: run the real extractor and complete package workflow on a small public-data subset. This is not the planned multi-seed benchmark or evidence of SOTA performance.

Hardware: Intel Core i5-8250U, 8 logical CPUs, approximately 16 GiB installed RAM, CPU execution. Project environment created with `uv sync --python /usr/bin/python`; existing project environments were preserved. CPU PyTorch was downloaded from its declared wheel index. No paid APIs or cloud GPU were used.

Commands initiated:

```bash
uv run layerood prepare-clinc --profile mini --protocol heldout-intents --seed 42 --output data/clinc-mini-heldout
uv run layerood embed --input examples/requests.jsonl --output artifacts/example-embeddings.npz
```

Prepared sizes: 40 fit, 20 reference, 25 calibration and 40 evaluation texts. The evaluation contains 20 ID and 20 held-out-intent texts. Source commit and data hash are in the prepared manifest. The 5 retained known intents are a subset; this is not the complete CLINC150 benchmark.

## Completed execution

The extractor returned shape `(3, 3, 576)` for three request texts at transformer outputs 10, 20 and 30. Resolved SmolLM2 revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. Initial model loading, including the download, took approximately 105 seconds; the three-text extraction took approximately 0.172 seconds. Those first artifact identities preceded a strengthened fingerprint (format 2), which additionally includes tokenizer behaviour and model configuration. The subsequent full package example used the strengthened identity.

```bash
uv run layerood run --data data/clinc-mini-heldout --output artifacts/clinc-mini-heldout-42 --local-files-only
```

The complete 125-text workflow finished in approximately 11.18 seconds with the model weights already downloaded. It generated embeddings, fitted the detector and scored all 15 available methods. This is aggregate workflow time, not a standalone detector latency measurement. Retained versions: Python 3.13.9, torch 2.14.1+cpu, transformers 4.57.6, numpy 2.5.4, scipy 1.18.1 and scikit-learn 1.9.1.

| Method | AUROC, OOD positive | ID false alarms at calibrated alpha=.05 | OOD recall |
|---|---:|---:|---:|
| Final-layer kNN | 0.5425 | 0.15 | 0.10 |
| Concatenated-layer Mahalanobis | 0.7825 | 0.00 | 0.05 |
| Global diagonal residual fusion | 0.8050 | 0.00 | 0.20 |
| Proposed local transport | 0.7725 | 0.00 | 0.20 |
| PCA residual baseline | 0.8950 | 0.05 | 0.55 |

There were 20 ID and 20 OOD evaluation texts. These numbers are too small for a reliable operating-point or multi-condition superiority claim. The proposed score did not outperform the matched global control or the strongest baseline. Its fixed hyperparameters were not tuned after this result. Full per-method, length-matched and group metrics are in the saved JSON.

## Artifacts and interpretation

- [Run manifest](../results/clinc-mini-heldout-42/run.json), [all-method metrics](../results/clinc-mini-heldout-42/metrics.json), [predictions](../results/clinc-mini-heldout-42/predictions.json), [readable results](../results/clinc-mini-heldout-42/report.md). These are byte-identical copies of the original local outputs, retained for repository readers.
- Generated local binary artifacts: `artifacts/example-embeddings.npz` and `artifacts/clinc-mini-heldout-42/detector.npz`. They are excluded from Git and can be regenerated with the recorded commands.
- Package wheel and source archive built with `uv build`.
- Saved-detector prediction on the three request texts completed with `uv run layerood predict --detector artifacts/clinc-mini-heldout-42/detector.npz --input examples/requests.jsonl --output artifacts/example-predictions.json --local-files-only`. These requests are usage examples, not an additional labeled benchmark.
- [Delivery manifest](../outputs/delivery-manifest.json) records current code/artifact hashes and the static review. It is explicitly a post-execution delivery snapshot. Evaluated OOD intent names were also added to the dataset/run manifests from retained inputs without changing results.

The implementation now supports the intended research. The scientific result is a small negative locality comparison, not an improved-SOTA detector. Next: inspect whether the PCA residual result survives the fixed full protocol, and assess local conditioning without altering the completed example's interpretation. No test suite or large benchmark was run.

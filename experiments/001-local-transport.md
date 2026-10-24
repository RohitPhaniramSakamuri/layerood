# Local conditioning for intent OOD

Experiment ID: 001. Status: planned full evaluation. Opened 24 October 2026.

## Question and hypothesis

Does reference-neighbour conditioning improve a frozen SLM's detection of unsupported intents beyond global residuals and final-layer density? The exact method is fixed in [APPROACH.md](../APPROACH.md). The primary locality comparator is global_diag_fusion.

## Setup

- Models: default SmolLM2-135M; follow-up SmolLM2-360M and MiniLM sentence encoder. Store resolved commits and fixed mean pooling. No model fine-tuning or OOD exposure.
- Data: official CLINC full source, separate official-OOS and heldout-intent protocols. Random held-out intent status alone does not prove semantic proximity; a within-domain hard-OOD extension needs separate metadata/protocol work.
- Seeds: 11, 19, 23, 42, 71; `--profile full --protocol heldout-intents --id-fraction 0.75`. Record selected known and evaluated OOD intents. Repeat official-OOS as a separate question.
- Fit/reference: 60/40 per retained training intent before deduplication. Calibration: official ID validation pool. Evaluation: official test texts, preserving train/cal/test separation. No OOD train/validation examples are used.
- Hyperparameters: three retained layers; 128-token limit; PCA rank up to 32; k=10; ridge=10; local variance shrinkage=0.5; fixed max fusion; alpha=0.05. Freeze these before looking at benchmark outcomes. Any later choice requires a separate development protocol and fresh evaluation.

## Commands

```bash
uv run layerood prepare-clinc --profile full --protocol heldout-intents --seed 11 --output data/clinc-heldout-11
uv run layerood run --data data/clinc-heldout-11 --output artifacts/clinc-heldout-11 --seed 11
```

Use fresh paths for other seeds and model conditions. Extraction caches may be reused only when their complete encoder signature matches. Normalized text overlap is rejected before fitting/evaluation.

## Measurements and criteria

Report OOD-positive AUROC/AUPR, OOD recall at the ID-calibrated alert threshold, observed ID false alarms, and ID FPR at 95% OOD TPR. Report raw and fixed-length-bin-matched evaluation with retained coverage. Show every method, seed and label/group breakdown; do not select favourable runs.

The main comparison is local_transport versus global_diag_fusion, with local_density and mahalanobis_concat as essential controls. Aggregate paired seed differences and uncertainty before claiming a reliable benefit. A gain accompanied by more ID false alarms needs to be reported as a trade-off. Use a sufficiently large calibration pool; the mini run cannot substantiate a 5% operating-point claim.

Separately measure cold model load, uncached extraction, cached extraction, each score's standalone latency, and memory on the target hardware. Current all-method scoring time is a workflow measure, not standalone latency. A published SOTA comparison requires matching the original protocol, faithful current baselines and explicit compute/supervision differences; the current command set alone does not establish that comparison.

## Actual results

Full multi-seed/model runs have not been executed. The package demonstration is recorded separately in experiment 000. Negative results will remain in the project records.

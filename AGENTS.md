# Local LLM OOD project

Follow the parent workspace AGENTS.md when one is present. This repository is also usable as a standalone checkout. Preserve existing work, keep code and research records together, verify source metadata, and distinguish planned experiments from actual execution. Do not invent results, novelty or publication claims. This is a separate prospect opened 24 October 2026. Do not import metrics, deadlines, budgets or novelty claims from the vision adaptation or earlier OOD projects.

The product is `layerood`: a local embedding extractor, reusable Python detector API, CLI, optional local HTTP service and benchmark adapters. Use uv and the project environment. Default execution is CPU, float32, SmolLM2-135M, batches of four and bounded token length. Larger models and full benchmarks require deliberate commands; no paid services are part of the package.

The hypothesis concerns locally conditioned cross-layer transport residuals. Layer trajectories, embedding/trajectory fusion and conformal calibration are existing ideas. SOTA is an empirical target, not an implemented feature or a verified contribution.

Keep fit, reference-bank and calibration texts disjoint. Evaluation text must not overlap any of them. Calibration uses ID data only; evaluation OOD labels must not select hyperparameters. Record exact model revision, pooling, layers, truncation and input hashes. Never call conformal p-values a probability that a request is OOD.

Do not add or run test suites unless requested. A documented package example is ordinary usage, not a benchmark replication. Record real execution separately from planned benchmark work. Use uv for Python environments and execution, never pip. Keep README, references, paper notes and experiment records current during substantive work; preserve negative findings.

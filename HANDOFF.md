# Codex handoff: develop LayerOOD into a complete research package

Prepared 24 October 2026. Starting implementation: commit `005cc443628e22b7b7714bc08a229663dad5706d` on `main` in the private repository [RohitPhaniramSakamuri/layerood](https://github.com/RohitPhaniramSakamuri/layerood).

## 1. Objective and working agreement

Continue this existing project on the new machine. Build a useful local embedding/OOD package with substantive experimental evidence, clear documentation, reusable workflows and polished data visualizations. The intended audience is the researcher, teammates, faculty reviewers and future users of the package.

“Substantial work” means implemented capabilities, executed experiments, inspectable outputs and honest interpretation. Preserve negative results. Do not invent runs, confidence intervals, novelty, benchmark wins, user studies or a history of development. Avoid filler files or numerous nearly identical experiments whose only purpose is to inflate counts.

Work autonomously on reversible local implementation and documentation. Read `AGENTS.md`, `README.md`, `APPROACH.md`, the two experiment records and the retained pilot results before making changes. Keep this repository separate from the earlier vision adaptation and faculty OOD projects. Their metrics, deadlines, budgets and cloud credits do not apply here.

Use uv. Preserve the lockfile unless a dependency change is necessary. The repository explicitly says **do not add or run test suites unless the user requests them**; this handoff does not revoke that rule. Document ordinary package usage, builds and static inspection accurately. Do not claim automated test coverage. If the user later requests tests, prioritize leakage, calibration, batching, persistence and API consistency over tests that merely repeat implementation logic.

Keep the GitHub repository private. Work on `codex/package-and-visuals` unless that branch already contains another agent's work, in which case use a fresh descriptive branch. Commit coherent milestones and push the working branch to the existing origin. Do not force-push, publish to PyPI, change visibility or spend on remote APIs/GPUs as part of this task. The user already requested GitHub storage; ordinary commits and branch pushes are part of continuing it.

## 2. Current implementation and actual evidence

The package is version 0.1.0, built with Hatchling, with import name and command `layerood`. It has:

- Local Transformers `AutoModel` extraction with CPU float32 defaults, frozen weights, masked mean or last-token pooling, retained hidden layers and exact-text caching.
- NPZ embeddings with JSON metadata and a portable NPZ detector; no pickle loading. Identity includes model revision, tokenizer behaviour, model configuration, pooling, layer selection and truncation.
- Three disjoint ID-only fitting pools: fit learns PCA/transport maps; reference supplies neighbours and score normalization; calibration ranks the frozen scores.
- Fifteen scores when class labels are supplied: final kNN, local density, final/concatenated/layerwise Mahalanobis, sparse TV-score, PCA residual, class-relative Mahalanobis, global/local residuals and fusions, and length-only control.
- CLI commands `embed`, `fit`, `score`, `predict`, `evaluate`, `prepare-clinc`, `run`, `inspect`, `serve`.
- Python `LocalEncoder`, `LayerOOD` and `OODPipeline` interfaces. Optional FastAPI endpoints `/healthz`, `/v1/embeddings`, `/v1/ood`; service implementation exists, but an HTTP execution was not recorded in the original session.
- CLINC150 source preparation, fixed-bin length-matched evaluation, method/group summaries, installable wheel/sdist, literature notes, a glossary and experiment records.

The original machine was an i5-8250U with about 16 GiB RAM and no GPU. Do not assume the new machine has those capabilities. Python supports 3.11–3.13. The committed uv source configuration selects CPU PyTorch wheels: `--device cuda` alone does not turn it into a CUDA installation. Verify official installation guidance before changing that configuration; keep CPU use working.

### Executed pilot — preserve these facts

Model: SmolLM2-135M revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`; layers 10/20/30, dimension 576, mean pooling, max length 128. CLINC subset, seed 42: 40 fit + 20 reference + 25 calibration + 40 evaluation texts. Evaluation has 20 ID and 20 OOD texts, from five retained ID and five evaluated OOD intents.

| Method | OOD-positive AUROC | ID false-alarm rate | OOD recall at calibrated alpha=.05 |
|---|---:|---:|---:|
| Final-layer kNN | 0.5425 | 0.15 | 0.10 |
| Concatenated-layer Mahalanobis | 0.7825 | 0.00 | 0.05 |
| Matched global-diagonal fusion | 0.8050 | 0.00 | 0.20 |
| Proposed local transport | 0.7725 | 0.00 | 0.20 |
| PCA residual | 0.8950 | 0.05 | 0.55 |

The complete small workflow took about 11.18 seconds after model weights were available. This includes multiple stages and all methods; it is not standalone detector latency. Initial weights loading/download took approximately 105 seconds on the original connection.

The candidate did not beat the strongest controls. This tiny run does not establish that PCA residual is generally best either. Do not redesign the method using this evaluation and subsequently call it an untouched confirmation set. The existing 15 scores are not 15 trained language models, and multiple seeds alter the dataset split/intent selection rather than train new frozen SLMs.

### What transfers through Git

`results/clinc-mini-heldout-42/` contains metrics, predictions, run/data manifests, readable results and copy hashes. Source code, notes, examples and `uv.lock` are committed. `.venv/`, downloaded data, model cache, embedding cache, detector binaries and build outputs are excluded. The historical manifest's original absolute command path is provenance, not a path that will work on the new machine. The delivery manifest is explicitly a post-execution snapshot.

## 3. Start on the new machine

```bash
gh repo clone RohitPhaniramSakamuri/layerood
cd layerood
git status --short
git switch -c codex/package-and-visuals
uv sync --locked
uv run layerood --help
```

If already cloned, inspect local changes and fetch before choosing a branch; do not reset the checkout. Inspect CPU, available RAM, disk and accelerators. Record the environment and a measured small-run throughput before estimating full-run costs. Use a Python version in the declared range; do not immediately regenerate dependencies to accommodate an unrelated system Python.

Run an ordinary fresh-machine example in new output directories. First model access needs network; `--local-files-only` is appropriate after the weights have been downloaded.

```bash
uv run layerood prepare-clinc --profile mini --protocol heldout-intents --seed 42 --output data/new-machine-mini
uv run layerood run --data data/new-machine-mini --output artifacts/new-machine-mini --revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2
uv run layerood predict --detector artifacts/new-machine-mini/detector.npz --input examples/requests.jsonl --output artifacts/new-machine-predictions.json --local-files-only
```

For exact historical data reconstruction, use the URL and SHA256 in the committed dataset manifest, not the current GitHub default branch. The adapter accepts `--source path/to/data_full.json`. Compare the source hash and selected IDs before interpreting small metric differences. Downloads, caches and non-identical platforms are part of the reproduction record.

## 4. Implementation priorities, in order

### A. Make saved runs independently analyzable

This is the first concrete implementation task. Current `predictions.json` includes IDs, token lengths and method scores but **does not include `is_ood` or intent labels**. The evaluated embedding artifact contains those labels but is not committed. Aggregate metrics alone cannot reconstruct ROC/PR curves.

Add a versioned, text-free evaluation export to each new run:

- `evaluation_records.jsonl`: `id`, `text_hash`, `label` when available, `is_ood`, `group`, `token_count`, `original_token_count`, `truncated`.
- `scores.csv`: one row per `(run_id, id, method)` with raw score, p_id, alpha and flag. Store model, protocol, seed and provenance in the run manifest rather than duplicating the full configuration on every row.
- Manifest: schema version, Git revision and dirty status, source/input hashes, encoder identity, split counts, selected intents, methods, hyperparameters, timings and completion/failure status.

Join records by ID and verify uniqueness and one-to-one coverage. Do not rely on row position or infer labels from a score. Existing files remain readable; add fields/artifacts without silently replacing historical measurements. Version new artifact formats if semantics change.

For historical visualizations, reconstruct the retained pilot's labels from its pinned source/splits and saved IDs, checking hashes. If that fails, mark that chart unavailable; the existing aggregate table is still usable. Keep raw user text out of committed exports by default.

### B. Add reusable plotting and aggregation

Implement new commands `layerood plot --run <directory> --output <directory>` and `layerood summarize --runs <directories...> --output <directory>`. These commands should read saved artifacts without loading model weights or rerunning inference. Add a `viz` optional dependency group with Matplotlib for static exports and Plotly for a standalone interactive report. Keep plotting imports out of the base inference path.

Summarization must group compatible dataset/protocol/model/pooling/alpha configurations, preserve per-seed rows, and expose missing/failed runs. Never pool a mini profile with full-profile results or combine different OOD protocols into one unexplained mean. Report mean and sample SD across seeds only when at least two seeds exist. Include paired per-seed differences against `global_diag_fusion` and the strongest declared baselines; do not present seed SD as a confidence interval.

### C. Improve everyday package use

- Add `layerood doctor` for environment/model-cache inspection; no implicit model download. Provide actionable messages for unsupported Python, missing optional dependencies and unavailable devices.
- Add resumable stage orchestration: an unchanged completed extraction stage may be reused only when source and encoder identities match. Failed jobs retain their logs and status; never silently mix configurations.
- Add optional method selection to scoring/prediction while preserving the current default output for backward compatibility. Distinguish the selected production score from computing every baseline for research. This enables meaningful method-specific timing.
- Improve progress and structured logging for downloads, extraction, fitting and evaluation. Explicitly distinguish cached from uncached processing.
- Add user guides for extraction, custom ID data, calibration, interpreting alerts and running the local API. Include one executable walkthrough notebook that calls package APIs instead of duplicating the implementation; save outputs only after executing it.
- Add one Mermaid architecture diagram and a small artifact-format guide. Build wheel/sdist with `uv build` after substantive packaging changes. Preserve existing CLI flags and saved-artifact support or provide a clear migration.

Do not grow this into a generic platform or add external services. The useful deliverable is a dependable local package and a self-contained research report.

## 5. Visualization specification

Build an offline HTML report with embedded assets/Plotly JavaScript and PNG/SVG exports of scientific plots. It should open from disk without a running service or CDN. Use a clean white background, readable labels and a consistent accessible palette. Every chart must show its run/model/protocol, ID/OOD counts and data source. Prefer five representative methods in dense plots, with all methods available in the interactive view/table.

| View | Required content | Interpretation rule |
|---|---|---|
| Overview | Model, protocol, split sizes, seed count, execution status and method table | Label mini results and incomplete matrices visibly. |
| Data overview | Intent counts, ID/OOD counts, token-length distributions and truncation fraction | Separate fit/reference/calibration/evaluation. |
| ROC and precision–recall | Curves from labeled per-text scores; OOD explicitly positive | Show chance/prevalence baselines; no fabricated curves from aggregate AUROC. |
| Score distributions | ID/OOD empirical distributions for chosen methods and calibrated operating points | Avoid smoothing choices that hide small counts. |
| Calibration behaviour | Observed ID false alarms versus alpha, plus OOD recall versus alpha | Do not call p_id a posterior probability. With 25 calibration texts, acknowledge coarse ranks. |
| Method comparison | Seed dots, mean ± SD and paired differences | A single seed gets a point, not a decorative error bar. |
| Length control | Length-only baseline, raw versus matched metrics and retained coverage | Matching is coarse; it does not prove invariance. |
| Error analysis | False-positive/false-negative counts by intent/group and length | Text examples require local source availability; never invent explanations. |
| Efficiency | Uncached extraction, cached processing, individual score latency and memory | Do not reuse all-method workflow time as one method's latency. |
| Geometry, secondary | ID-fitted PCA projection of embeddings with later points transformed into that space | Use for exploration; 2-D separation does not prove OOD performance. |

Make figures reproducible from the saved exports. Include a local interactive table filtered by model, protocol, seed and method. There should be no large custom web frontend in this milestone.

## 6. Evidence-building sequence

The existing full protocol is in `experiments/001-local-transport.md`. Preserve it as the fixed baseline question. Run jobs sequentially by default, cache embeddings and record interrupted/failed jobs. Estimate runtime on the actual machine before launching the larger matrix. No deadline, paid-compute budget or GPU entitlement is assumed by this handoff.

1. Complete the small new-machine execution and plotting pipeline.
2. Run one full held-out-intent seed with SmolLM2-135M using fixed hyperparameters. This first full seed is part of the baseline evaluation, not a tuning set.
3. Complete seeds **11, 19, 23, 42, 71** for that model, and the same five seeds for the MiniLM architecture control. Compare the same actual prepared datasets across models.
4. Repeat the five-seed, two-model matrix under official-OOS as a separately reported protocol. That gives a planned primary matrix of 20 runs, not 20 completed runs until executed.
5. SmolLM2-360M and additional datasets are extensions after this core matrix. Add a second dataset only with verified primary-source provenance, a declared protocol and an actual download/run; do not pad the report with datasets mentioned only in a proposal.

A full held-out-intent run uses `--profile full --protocol heldout-intents --id-fraction 0.75`; the dataset preparation seed controls intent selection and split draws. Pass the matching seed to `run` for the length-matching record. Merely changing `run --seed` against identical prepared files is not a new split repeat. Pin model revisions and data source versions; the current automatic source lookup follows GitHub's default branch.

Keep PCA rank 32, k=10, ridge alpha 10, variance shrinkage 0.5, mean pooling, the declared three layers, 128-token limit and alpha=.05 for the initial fixed matrix. These runs establish what the current approach does. Do not pick the best feature, rank or pooling on their final OOD results and report that choice as an unbiased improvement.

For a subsequent ablation/development study, use a separate declared development protocol and fresh confirmation. Limit initial questions to mean versus last pooling, layer subsets, and locality versus its matched global-diagonal control. If proposing a new method, motivate it with actual failure cases and closest prior art before implementation. If PCA residual or a simpler encoder continues to win, it remains a legitimate project finding; the package need not force the original hypothesis to win.

Strong evidence requires more than AUROC: report actual ID false alarms, OOD recall at the frozen calibration threshold, uncertainty from genuinely executed repeats, class/group breakdowns and cost. Repeated seeds reuse the underlying dataset and are not independent datasets. Random held-out intents are not automatically semantically near-OOD. A SOTA comparison requires a matching protocol and faithful current competitors; the current baseline collection alone does not establish it.

## 7. Completion criteria

The next milestone is complete when a new reader can:

1. Install the package with uv, extract embeddings, fit from three disjoint ID pools and score new text using documented commands.
2. Open an offline report that is generated entirely from saved run artifacts, with no hidden training or missing label join.
3. Inspect a run ledger showing exactly which matrix entries completed, failed or remain planned; see all completed results, including negative findings.
4. Compare the candidate against matched global-diagonal fusion, concatenated Mahalanobis, PCA residual and simple kNN under identical conditions.
5. Read a concise methods/results/limitations report with linked figures, reproducible commands and a clear distinction between published methods and our adaptations.
6. Use an updated wheel/sdist, README, artifact documentation and a recorded local API example if the service has actually been exercised.

Deliver working code and evidence, not just this roadmap. Commit useful increments. Do not mark the full research matrix complete because the package or charts are finished. If hardware or data access blocks a run, preserve partial progress, state the precise blocker and continue independent documentation/analysis work.

## 8. First action for the receiving Codex

Read the current repository and report what is present on the new machine. Then implement the labeled evaluation export and a first offline visualization report from the retained pilot. This creates useful visible progress while the larger runs are prepared. Keep routine work moving without repeatedly asking for confirmation; ask only for missing information or a genuine boundary such as paid resources or changing the explicit test-suite rule.

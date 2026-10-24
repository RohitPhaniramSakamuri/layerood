# Reading and implementation provenance

Date: 24 October 2026. Used the installed literature-review skill's researcher and verifier roles. The skill's `/lit` expansion is not exposed as an executable tool here; primary-source web retrieval, researcher review and local records supplied the workflow. A separate verifier reviewed the package statically without executing tests.

- TV-Score: primary metadata, sections 2–4 and the UMAP appendix passage. The paper's neighbour count in UMAP is not its detector parameter.
- Two-Pathway: primary HTML method, setup, limitations, feature/calibration definitions and selected appendices. No complete replication or code audit; usable public code availability was not verified.
- SNAP-UQ: published metadata plus abstract/introduction and conditional scoring passages from an accessible submission PDF. The final ICLR record supersedes the initially found anonymous submission. Full published-code equivalence is not established.
- PRISM: primary introduction/method overview and author repository usage, not an experimental reproduction.
- kNN, RMD, DkNN and text-DkNN: primary metadata/abstracts; RMD method and experiment overview. No complete reproduction.
- CLINC: official repository README, authoritative ACL metadata, data source and license. Prepared subsets record source commit and file hash.
- SmolLM2 and MiniLM: official model cards. No task-specific performance inferred from their general model evaluations.
- Conformal tutorial: primary metadata/abstract; the package's rank definition and independence requirements are written explicitly.

The implemented baselines are documented adaptations with common retained layers and preprocessing. They must not be described as exact reproductions of published scores. No paper metric was copied into this project's actual-results table.

# Embedding Trajectory for Out-of-Distribution Detection in Mathematical Reasoning

Citation key: `Wang2024TVScore`. [Primary source](https://arxiv.org/abs/2405.14039). Verified author/year/venue metadata is in [references.bib](../references.bib).

Reading progress: Sections 2–4 and selected UMAP appendix passages. No full replication or code audit.

## Research connection and limitations

The method measures variation of layerwise Mahalanobis distances. Our tv_sparse uses three retained layers and is explicitly an adaptation. The UMAP neighbour count is not a detector setting.

## Implementation and next step

See [the exact method](../APPROACH.md) and [the planned experiment](../experiments/001-local-transport.md). Assess comparability before treating a package baseline as a reproduction of this source.

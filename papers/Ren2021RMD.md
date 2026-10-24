# A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection

Citation key: `Ren2021RMD`. [Primary source](https://arxiv.org/abs/2106.09022). Verified author/year/venue metadata is in [references.bib](../references.bib).

Reading progress: Method definition and experiment overview. No full replication or code audit.

## Research connection and limitations

Background-subtracted class-conditional distance is an important near-OOD control. The implementation requires complete, multi-class fit labels and uses source-fitted PCA coordinates.

## Implementation and next step

See [the exact method](../APPROACH.md) and [the planned experiment](../experiments/001-local-transport.md). Assess comparability before treating a package baseline as a reproduction of this source.

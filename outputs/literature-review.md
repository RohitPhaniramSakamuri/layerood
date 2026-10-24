# Closest literature and a bounded research target

Reviewed 24 October 2026. The broad idea is established: hidden layers, nearest neighbours, prediction residuals and calibrated uncertainty have all been used for OOD or uncertainty. We selected a narrower hypothesis rather than assigning a novelty or publication score.

| Prior work | Established mechanism | Consequence for this package |
|---|---|---|
| [Deep kNN, 2018](https://arxiv.org/abs/1803.04765) | Neighbour evidence across layers supports confidence. | Multi-layer nearest neighbours are not new. |
| [Wallace et al., 2018](https://aclanthology.org/W18-5416/) | Nearest-neighbour interpretation extends to text representations. | Applying neighbour analysis to language is also established. |
| [Deep nearest-neighbour OOD, 2022](https://proceedings.mlr.press/v162/sun22d.html) | Normalized feature-distance detection is a strong simple control. | Include final-layer kth-neighbour distance. |
| [Relative Mahalanobis, 2021](https://arxiv.org/abs/2106.09022) | Subtracts background distance from class-conditional distance for near-OOD. | Include the labeled-ID comparator; do not fake it with one Gaussian. |
| [TV-Score, 2024](https://arxiv.org/html/2405.14039v2) | Measures variation of per-layer Gaussian distances in mathematical reasoning. | Include a declared sparse-layer adaptation. |
| [SNAP-UQ, 2026](https://arxiv.org/abs/2508.12907) | Predicts later activation statistics and scores unusual residuals. | Cross-layer prediction error is not a new principle. |
| [Two-Pathway, 2026](https://arxiv.org/html/2605.00269v1) | Combines semantic embeddings and processing trajectories; studies length confounding. | Fusion and length matching already have direct precedent, including small LMs. |
| [PRISM, 2026](https://arxiv.org/html/2609.06785v1) | Combines layers through subspace geometry and residual energy. | Include concatenated-layer Gaussian and discarded-energy controls. |

The concrete candidate is an ID-reference-neighbour distribution over layer-prediction residuals. It differs from scoring a global residual vector or merely concatenating representations. That difference needs an ablation with the same covariance structure: the implementation supplies both global and local diagonal residuals with matched fusion.

There is a mathematical reason to be demanding about the controls. Under a joint Gaussian model, a global linear conditional Gaussian and the corresponding joint Mahalanobis score are closely related through the Schur complement. A more complicated sequence of global Gaussian calculations may therefore add no information. Local conditioning has to demonstrate its own benefit.

The realistic initial target is **frozen small-model intent OOD with no OOD training and a fixed CPU inference budget**. It is not defensible to predict that a 135M causal model will outperform unrestricted fine-tuned models, larger models or hybrid LLM systems. A compact sentence encoder with the same detection controls is a necessary architecture comparison; the package accepts the official [MiniLM model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

The package makes the hypothesis executable. It does not turn a stated limitation into a verified gap, and its initial small data run is not a benchmark SOTA result. Full exact-method replication and more recent competing intent systems remain future literature/benchmark work.

# Working glossary

These explanations introduce the package vocabulary. Understanding has not been confirmed by the user.

## Embedding and pooling

An embedding is a list of numbers describing an input inside a model. A transformer produces one vector per token at each layer; pooling combines them into one vector for the text. For example, averaging the valid token vectors gives a fixed-size representation whether a request has five words or fifteen. It lets us compare requests and store reference vectors. Padding tokens must be excluded. A language-model embedding is not automatically as good for semantic search as a trained sentence embedding. Prerequisite: vectors and averages. [Model documentation](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

## ID and OOD

In-distribution (ID) means drawn from the kind of inputs the detector's reference collection is intended to represent. OOD means outside that scope. For a bot trained on card services, a cooking request is a clear example; an unsupported banking request is harder. OOD is relative to the task, not a universal property of a sentence, and does not automatically mean the model's answer is wrong. Practical use: flag inputs for a fallback workflow. [CLINC](https://aclanthology.org/D19-1131/).

## Nearest neighbours

The reference texts with the most similar embedding vectors. If a request is near several card-freezing requests, those are its neighbours. k is how many neighbours we retain; default 10. Their distance gives a simple anomaly score and their residuals provide a local comparison group. Similarity is model- and layer-dependent; nearby vectors do not prove identical intent. Prerequisite: embeddings and a distance. [Sun2022KNN](REFERENCES.md).

## Layer transport residual

A residual is what a predictor failed to explain. Here, a small regression predicts a later-layer vector from an earlier-layer vector and text length. Actual minus predicted is the residual. We compare it to residuals from familiar nearby texts. A large error may be unusual, but it can also reflect a weak predictor or normal variation. Prerequisites: vector, regression and subtraction. Next-activation prediction has prior work in [SNAP-UQ](REFERENCES.md).

## Covariance and shrinkage

Variance describes how much one number varies; covariance describes how two vary together. With few reference texts, estimated variances can be unstable. Shrinkage mixes a local estimate with a more stable global estimate. In this package each local residual variance is mixed 50:50 with the global variance. It stabilizes division by the expected variation; it does not create additional observations. Prerequisites: mean and variance.

## Calibration p-value

Compare a new anomaly score with scores from separate known-good texts. If only a few calibration scores are as large, the new input is unusual relative to that reference population. With n calibration texts the smallest possible rank p-value is 1/(n+1). The package flags p_id <= 0.05 by default. A value of 0.02 does not mean a 98% probability of OOD. Its false-alarm interpretation assumes exchangeable ID observations. [Conformal introduction](https://arxiv.org/abs/2107.07511).

## AUROC and false alarms

OOD-positive AUROC measures how often an OOD input scores above an ID input as the alert threshold varies. A score of 0.5 is chance ranking. ID false-alarm rate instead counts known inputs incorrectly flagged at the selected threshold. A detector can have good average ranking but poor behaviour at the chosen operating point, so the package reports both. The positive class is explicitly OOD to avoid reversing interpretations.

## Confounding by length

A score may appear to detect unfamiliar intent while mainly detecting longer text. If OOD examples happen to be longer, that shortcut can look successful. We include a length-only score and comparisons within fixed token-length bins. Those controls reduce one shortcut; they do not establish invariance to wording or style. [Two-Pathway](https://arxiv.org/abs/2605.00269).

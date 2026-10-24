# Research branches

## Local conditional transport — selected

Origin: user requested a separate locally deployable SLM embedding/OOD product with a hypothesis aimed at stronger results. Evidence: layer trajectories and global prediction residuals have direct predecessors, but the identified methods do not establish this specific local reference-residual protocol. Question: does neighbourhood conditioning add value after controlling covariance structure, layer information and length? Status: implemented, full benchmark pending. [Approach](../APPROACH.md).

## Representation quality versus detector sophistication

Question: does a compact sentence encoder with simple kNN outperform a causal SLM with a complex detector at equal practical cost? Include MiniLM as an architecture control before attributing gains to language-model reasoning. Status: supported model configuration, unrun comparison.

## Shift versus task failure

This product measures support relative to an ID text collection. OOD status is not automatically an incorrect generated answer, hallucination, prompt injection or harmful content. Predicting those outcomes requires their own labels, evaluation and likely separate project scope. Status: explicitly outside v1.

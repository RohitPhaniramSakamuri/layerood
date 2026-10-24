# Local conditional layer residuals for text OOD detection

Status: implemented hypothesis, 24 October 2026. No verified novelty or SOTA claim.

## Hypothesis

For a frozen small language model, the normal relationship between intermediate and later representations varies across semantic neighbourhoods. Comparing a query's layer-prediction residual with the residuals of nearby known texts may detect unsupported but semantically adjacent intents better than final-layer distance or a global residual score, after controlling for text length.

Example: a banking assistant supports balance enquiries and card freezes but not chargeback appeals. An appeal can resemble its supported banking texts. The candidate signal asks whether the model transforms that text like nearby supported requests. This is an intuition to test, not evidence that layer evolution identifies intent or correctness.

## Representation and fitting

1. Run a frozen local model. Retain pooled transformer outputs at roughly one-third, two-thirds and the final layer. Use the attention mask for mean pooling. Extract no vocabulary logits and generate no tokens.
2. Use three text-disjoint ID pools: **fit**, **reference**, **calibration**. Fit is the only pool that learns PCA, transport maps and primary Gaussian baselines. Reference supplies neighbour geometry, residual distributions and fusion normalization. Calibration supplies only final score ranks.
3. Unit-normalize each layer vector. Fit PCA independently per layer on fit texts, retain up to 32 nondegenerate directions and divide by source component standard deviations. Let these coordinates be z_l.
4. Fit ridge maps `z_(l+1) ≈ W_l [z_l, standardized log(1 + token count), 1]`, alpha 10, unpenalized intercept. Concatenate the errors across transitions into residual vector e. Explicit length input is an attempted nuisance adjustment, not proof of length invariance.

## Query score

Find k=10 nearest reference texts in the **unit-normalized middle-layer embedding**. Compute their residual mean mu_local and sample diagonal variance v_local. The local residual score is the mean coordinate-wise squared difference `(e - mu_local)^2 / (0.5 v_local + 0.5 v_global)`, with a fixed numerical floor. v_global is the reference residual variance.

Independently, compute the final-layer kth-neighbour distance divided by the mean leave-one-out kth-neighbour radius of its neighbours. This local-density ratio retains a signal for far-OOD inputs.

For these two nonnegative components, apply log(1 + score), subtract the reference median, and divide by the reference interquartile range with a fixed 0.05 floor. Reference statistics use self-excluded neighbours. The fused scalar is the maximum of the two standardized components. No fusion weight is selected using OOD examples.

For an independent calibration pool C, `p_id(x) = (1 + count{s in C: score(s) >= score(x)}) / (|C| + 1)`. Flag when p_id <= alpha, default 0.05. Ties are included conservatively. This p-value is not an OOD probability; its false-alarm interpretation requires exchangeability with future ID data.

## What would support or reject the hypothesis

The decisive locality ablation replaces mu_local and v_local with the global reference residual mean and variance, retaining diagonal scoring and identical fusion/calibration machinery (`global_diag_fusion`). Compare also final kNN, density alone, concatenated-layer Mahalanobis, sparse TV-score and class-relative Mahalanobis.

Evidence would require gains on a preregistered held-out-intent protocol across multiple seeds and more than one model, while reporting actual ID false alarms and complete inference cost. Gains must survive token-length matching and strong global controls. An effect confined to a tiny subset, a higher false-alarm rate, or a weak final-layer baseline does not support a SOTA claim. If global diagonal fusion or concatenated-layer Mahalanobis matches the local score, the locality hypothesis is weakened.

Cross-layer residual prediction is already related to SNAP-UQ; trajectory scoring to TV-Score; embedding/trajectory fusion and length controls to Two-Pathway. The possible contribution is specifically the benefit and limits of **reference-neighbour conditioning** for frozen SLM intent OOD under a controlled protocol.

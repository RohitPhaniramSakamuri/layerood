# OOD evaluation

Higher OOD scores indicate more anomalous texts. OOD is the positive class.

| Method | AUROC | OOD AUPR | ID false alarms | OOD recall | ID FPR at 95% OOD TPR |
|---|---:|---:|---:|---:|---:|
| knn | 0.5425 | 0.5564 | 0.1500 | 0.1000 | 1.0000 |
| local_density | 0.5600 | 0.5587 | 0.1500 | 0.0500 | 1.0000 |
| mahalanobis | 0.6450 | 0.6039 | 0.1500 | 0.2000 | 0.8000 |
| mahalanobis_concat | 0.7825 | 0.8264 | 0.0000 | 0.0500 | 0.6500 |
| global_residual | 0.8150 | 0.8562 | 0.0000 | 0.2000 | 0.7000 |
| global_residual_diag | 0.8125 | 0.8505 | 0.0000 | 0.2000 | 0.5500 |
| local_residual | 0.7975 | 0.8493 | 0.0000 | 0.2000 | 0.6000 |
| length_only | 0.6175 | 0.7228 | 0.0000 | 0.1000 | 1.0000 |
| tv_sparse | 0.7900 | 0.8298 | 0.0000 | 0.1000 | 0.6000 |
| layer_mahalanobis_mean | 0.7825 | 0.8264 | 0.0000 | 0.0500 | 0.6500 |
| pca_residual | 0.8950 | 0.8910 | 0.0500 | 0.5500 | 0.3000 |
| relative_mahalanobis | 0.6650 | 0.7265 | 0.0000 | 0.0500 | 0.8500 |
| global_fusion | 0.8150 | 0.8562 | 0.0000 | 0.2000 | 0.7000 |
| global_diag_fusion | 0.8050 | 0.8468 | 0.0000 | 0.2000 | 0.5500 |
| local_transport | 0.7725 | 0.7986 | 0.0000 | 0.2000 | 0.6000 |

ID-calibration alert level: 0.05. Evaluation texts: 40.
Thresholds were fixed using ID calibration; the ROC diagnostic uses evaluation labels only to describe the curve.

The JSON report also contains fixed-bin length-matched metrics, coverage and per-group breakdowns.
This comparison covers implemented baselines under one protocol. It does not establish a state-of-the-art result.

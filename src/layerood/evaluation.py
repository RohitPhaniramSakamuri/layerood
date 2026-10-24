"""Evaluation with OOD explicitly positive and thresholds fixed by ID calibration."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from .artifacts import EmbeddingBatch


def _metrics(y, scores, flags):
    id_mask, ood_mask = ~y, y
    result = {"n_id": int(id_mask.sum()), "n_ood": int(ood_mask.sum()),
              "id_false_alarm_rate": float(flags[id_mask].mean()) if id_mask.any() else None,
              "ood_recall_at_calibrated_threshold": float(flags[ood_mask].mean()) if ood_mask.any() else None,
              "auroc_ood_positive": None, "aupr_ood_positive": None, "id_fpr_at_95pct_ood_tpr": None}
    if id_mask.any() and ood_mask.any():
        fpr, tpr, _ = roc_curve(y, scores)
        result.update(auroc_ood_positive=float(roc_auc_score(y, scores)),
                      aupr_ood_positive=float(average_precision_score(y, scores)),
                      id_fpr_at_95pct_ood_tpr=float(fpr[tpr >= .95].min()))
    return result


def evaluate(batch: EmbeddingBatch, predictions: list[dict], alpha=.05, seed=42):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    if len(predictions) != len(batch.records) or any(p["id"] != r["id"] for p, r in zip(predictions, batch.records)):
        raise ValueError("Predictions must match input records in order and id")
    if any(type(r.get("is_ood")) is not bool for r in batch.records):
        raise ValueError("Evaluation needs a boolean is_ood for every text")
    y = np.asarray([r["is_ood"] for r in batch.records], dtype=bool)
    if len(set(y)) != 2:
        raise ValueError("Overall evaluation requires both ID and OOD texts")
    if len(set(batch.hashes)) != len(batch.hashes):
        raise ValueError("Evaluation contains duplicate normalized text; deduplicate first")
    lengths = batch.lengths
    # Fixed bins and label-only matching. Every method sees the same selected texts.
    edges = [0,8,16,24,32,48,64,96,128,256,10**9]
    bins = np.digitize(lengths, edges, right=True)-1
    rng = np.random.default_rng(seed)
    matched, lengths_report = [], []
    for i in range(len(edges)-1):
        id_ix = np.flatnonzero((bins == i) & ~y)
        ood_ix = np.flatnonzero((bins == i) & y)
        count = min(len(id_ix), len(ood_ix))
        if count:
            matched.extend(rng.choice(id_ix, count, replace=False).tolist())
            matched.extend(rng.choice(ood_ix, count, replace=False).tolist())
        lengths_report.append({"lower_exclusive": edges[i], "upper_inclusive": edges[i+1],
                               "n_id": len(id_ix), "n_ood": len(ood_ix), "selected_each": count})
    matched = np.asarray(sorted(matched), dtype=int)
    groups = np.asarray([str(r.get("group", "unspecified")) for r in batch.records])
    report = {"positive_class": "OOD", "alpha": alpha, "n": len(y),
              "threshold_source": "disjoint ID calibration only; scores frozen before evaluation",
              "length_matching": {"bins": lengths_report, "seed": seed, "n_selected": len(matched),
                                  "fraction_id_retained": float((~y[matched]).sum()/(~y).sum()),
                                  "fraction_ood_retained": float(y[matched].sum()/y.sum()),
                                  "limitation": "Coarse token-length strata; not exact matching or proof of invariance"},
              "truncated_fraction": float(np.mean([r.get("truncated", False) for r in batch.records])),
              "methods": {}, "sota_claim": False}
    for method in predictions[0]["methods"]:
        s = np.asarray([p["methods"][method]["score"] for p in predictions])
        f = np.asarray([p["methods"][method]["p_id"] <= alpha for p in predictions])
        report["methods"][method] = {"all": _metrics(y, s, f),
            "length_matched": _metrics(y[matched], s[matched], f[matched]) if len(matched) else None,
            "groups": {g: _metrics(y[groups == g], s[groups == g], f[groups == g]) for g in sorted(set(groups))}}
    return report


def markdown_report(report):
    lines = ["# OOD evaluation", "", "Higher OOD scores indicate more anomalous texts. OOD is the positive class.", "",
             "| Method | AUROC | OOD AUPR | ID false alarms | OOD recall | ID FPR at 95% OOD TPR |",
             "|---|---:|---:|---:|---:|---:|"]
    for method, data in report["methods"].items():
        a = data["all"]
        cols = [a[k] for k in ("auroc_ood_positive", "aupr_ood_positive", "id_false_alarm_rate", "ood_recall_at_calibrated_threshold", "id_fpr_at_95pct_ood_tpr")]
        lines.append("| "+method+" | "+" | ".join("—" if v is None else f"{v:.4f}" for v in cols)+" |")
    lines += ["", f"ID-calibration alert level: {report['alpha']}. Evaluation texts: {report['n']}.",
              "Thresholds were fixed using ID calibration; the ROC diagnostic uses evaluation labels only to describe the curve.",
              "", "The JSON report also contains fixed-bin length-matched metrics, coverage and per-group breakdowns.",
              "This comparison covers implemented baselines under one protocol. It does not establish a state-of-the-art result.", ""]
    return "\n".join(lines)

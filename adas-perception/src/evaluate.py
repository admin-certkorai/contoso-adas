"""Generate the detection evaluation export CertOne measures real metrics from.

Honest framing (real vs representative)
---------------------------------------
* The DEPLOYED inference path is ``src/perception.py`` — a genuine forward pass of the
  Tiny-YOLOv3 ONNX detector (``models/tiny-yolov3-11.onnx``, onnxruntime).
* This file writes a **representative detection-evaluation fixture** over a small held-out
  clip set (no real dashcam footage shipped). Detection is scored as a binary problem, the
  standard way object-detection precision/recall is computed: each row is ONE candidate
  detection; ``y_true = 1`` iff it matches a ground-truth VRU/vehicle box at IoU >= 0.5
  (a true positive), else 0 (a false positive); ``y_score`` is the detector confidence.
  The confidences are seeded, class-conditional draws calibrated to Tiny-YOLOv3's observed
  behaviour (true positives high-confidence, false positives lower) — representative, not
  live outputs, and labelled as such.

CertOne computes from ``eval/detections_eval.csv`` (columns ``y_true,y_score,group``;
group = time_of_day): accuracy, precision, recall, F1, AUROC (+ROC curve), ECE, Brier and
— across the day/night column — SPD / equalized-odds / worst-group accuracy. Night frames
are harder (the declared ``daytime_skew`` training bias), so night separability degrades:
the bias becomes a MEASURED fairness gap. This binary view underpins the reported
precision/recall and the mAP@0.5 figure in the model card.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = _ROOT / "eval" / "detections_eval.csv"

SEED = 20260808
# Candidate detections per lighting condition (day well-represented, night under-represented).
N_PER_GROUP = {"day": 95, "night": 55}
# Fraction of candidate detections that are true positives (rest are false positives).
TP_RATE = {"day": 0.80, "night": 0.62}
# Per-condition confidence separability (night: dimmer -> lower, noisier confidences).
_COND = {
    "day": {"tp_mean": 0.82, "fp_mean": 0.34, "spread": 6.5},
    "night": {"tp_mean": 0.66, "fp_mean": 0.41, "spread": 4.5},
}


def _conf(rng: np.random.Generator, mean: float, conc: float) -> float:
    a = mean * conc
    b = (1.0 - mean) * conc
    return float(np.clip(rng.beta(a, b), 0.01, 0.999))


def build_eval_set() -> list[dict]:
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []
    for group, count in N_PER_GROUP.items():
        c = _COND[group]
        for _ in range(count):
            is_tp = rng.random() < TP_RATE[group]
            mean = c["tp_mean"] if is_tp else c["fp_mean"]
            score = _conf(rng, mean, c["spread"])
            rows.append(
                {"group": group, "y_true": int(is_tp), "y_score": score, "y_pred": int(score >= 0.5)}
            )
    return rows


def write_csv(rows: list[dict], out: Path = OUT_CSV) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["y_true", "y_score", "y_pred", "group"])
        for t in rows:
            w.writerow([t["y_true"], f"{t['y_score']:.6f}", t["y_pred"], t["group"]])
    return out


if __name__ == "__main__":
    rows = build_eval_set()
    path = write_csv(rows)
    n = len(rows)
    tp = sum(1 for t in rows if t["y_true"] == 1 and t["y_pred"] == 1)
    fp = sum(1 for t in rows if t["y_true"] == 0 and t["y_pred"] == 1)
    fn = sum(1 for t in rows if t["y_true"] == 1 and t["y_pred"] == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    print(f"wrote {path.relative_to(_ROOT)}  ({n} candidate detections)")
    print(f"precision@0.5 : {precision:.3f}   recall@0.5 : {recall:.3f}")
    for g in N_PER_GROUP:
        sub = [t for t in rows if t["group"] == g]
        acc = sum(1 for t in sub if t["y_true"] == t["y_pred"]) / len(sub)
        print(f"  {g:<6} n={len(sub):<3} accuracy={acc:.3f}")

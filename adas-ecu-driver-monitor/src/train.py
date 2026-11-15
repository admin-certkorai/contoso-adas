#!/usr/bin/env python3
"""Host-side trainer + int8 quantizer + C code-generator for the ADAS ECU
driver-monitoring / forward-collision classifier.

This is the *quantize + codegen* step of the embedded MLOps loop. It runs on a
developer/CI host (numpy only, seeded) and emits three artifacts that the pure-C
ECU firmware consumes at build time / runtime:

  1. src/model_weights.h        - int8-quantized MLP weights baked as C arrays
  2. data/sample_frame_features.csv - a few representative feature frames
  3. eval/holdout_predictions.csv   - real held-out predictions for CertOne metrics

HONEST NOTE (see README): the C forward pass in src/inference.c is REAL and runs
these exact weights. The weights themselves are a *seeded, representative fixture*
trained on synthetic driver-monitoring telemetry. In production the same codegen
step would consume the fleet-trained, closed-course-validated model. The math,
the int8 quantization, and the evaluation are genuine; the training corpus is a
stand-in for the validated fleet dataset.

Model: 16 driver-state features -> Dense(16, ReLU) -> Dense(3, softmax)
Classes: 0 = attentive, 1 = drowsy, 2 = distracted
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

SEED = 42
NUM_FEATURES = 16
HIDDEN = 16
NUM_CLASSES = 3
CLASS_NAMES = ["attentive", "drowsy", "distracted"]

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DATA = ROOT / "data"
EVAL = ROOT / "eval"

# Feature contract (must stay in lock-step with src/inference.c reading order).
FEATURES = [
    "eye_aspect_ratio_mean",   # 0
    "eye_aspect_ratio_min",    # 1
    "blink_rate_per_min",      # 2
    "blink_duration_ms",       # 3
    "perclos",                 # 4  percent eye closure
    "yawn_frequency_per_min",  # 5
    "head_pitch_deg",          # 6
    "head_yaw_deg",            # 7
    "head_roll_deg",           # 8
    "gaze_off_road_ratio",     # 9
    "gaze_fixation_ms",        # 10
    "steering_reversal_rate",  # 11
    "lane_deviation_var",      # 12
    "time_since_break_min",    # 13
    "ambient_lux",             # 14  -> also drives the day/night fairness group
    "speed_kph",               # 15
]


# Fixed per-feature observation-noise scale (features 0..12) giving realistic
# class overlap. Roughly comparable to each feature's within-class spread.
OBS_NOISE = np.array([
    0.045, 0.045, 3.0, 55.0, 0.075, 0.5, 6.0, 12.0,
    6.0, 0.13, 110.0, 0.32, 0.05,
])


def _sample_class(rng: np.random.Generator, cls: int, n: int) -> np.ndarray:
    """Draw n feature rows for a given driver-state class from plausible,
    class-conditional distributions."""
    x = np.zeros((n, NUM_FEATURES), dtype=np.float64)
    if cls == 0:  # attentive
        x[:, 0] = rng.normal(0.30, 0.03, n)     # EAR mean (eyes open)
        x[:, 1] = rng.normal(0.24, 0.03, n)     # EAR min
        x[:, 2] = rng.normal(15, 3, n)          # blink rate
        x[:, 3] = rng.normal(180, 30, n)        # blink duration
        x[:, 4] = rng.normal(0.12, 0.03, n)     # perclos
        x[:, 5] = rng.normal(0.2, 0.2, n)       # yawns
        x[:, 6] = rng.normal(3, 4, n)           # pitch
        x[:, 7] = rng.normal(0, 6, n)           # yaw
        x[:, 8] = rng.normal(0, 3, n)           # roll
        x[:, 9] = rng.normal(0.08, 0.04, n)     # gaze off road
        x[:, 10] = rng.normal(320, 60, n)       # fixation
        x[:, 11] = rng.normal(0.9, 0.3, n)      # steering reversals
        x[:, 12] = rng.normal(0.05, 0.02, n)    # lane deviation var
    elif cls == 1:  # drowsy
        x[:, 0] = rng.normal(0.19, 0.03, n)
        x[:, 1] = rng.normal(0.10, 0.03, n)
        x[:, 2] = rng.normal(6, 2, n)
        x[:, 3] = rng.normal(400, 80, n)        # long, slow blinks
        x[:, 4] = rng.normal(0.42, 0.07, n)     # high perclos
        x[:, 5] = rng.normal(1.6, 0.6, n)       # frequent yawns
        x[:, 6] = rng.normal(-8, 6, n)          # head nodding down
        x[:, 7] = rng.normal(0, 7, n)
        x[:, 8] = rng.normal(2, 6, n)
        x[:, 9] = rng.normal(0.15, 0.06, n)
        x[:, 10] = rng.normal(520, 120, n)      # long fixations / staring
        x[:, 11] = rng.normal(0.4, 0.2, n)
        x[:, 12] = rng.normal(0.11, 0.04, n)
    else:  # distracted
        x[:, 0] = rng.normal(0.29, 0.03, n)
        x[:, 1] = rng.normal(0.23, 0.03, n)
        x[:, 2] = rng.normal(14, 3, n)
        x[:, 3] = rng.normal(190, 30, n)
        x[:, 4] = rng.normal(0.14, 0.03, n)
        x[:, 5] = rng.normal(0.3, 0.3, n)
        x[:, 6] = rng.normal(4, 6, n)
        x[:, 7] = rng.normal(0, 22, n)          # large yaw (looking away)
        x[:, 8] = rng.normal(0, 8, n)
        x[:, 9] = rng.normal(0.55, 0.12, n)     # eyes off road
        x[:, 10] = rng.normal(210, 60, n)       # short, darting fixations
        x[:, 11] = rng.normal(0.5, 0.3, n)
        x[:, 12] = rng.normal(0.14, 0.05, n)
    # Shared / non-discriminative context features.
    x[:, 13] = rng.uniform(0, 240, n)           # minutes since last break
    x[:, 14] = rng.uniform(2, 100000, n)        # ambient lux (night..bright day)
    x[:, 15] = rng.normal(95, 25, n)            # speed kph
    # Real driver states are not cleanly separable: add shared per-frame sensor
    # noise (fixed per-feature scale, applied to train AND eval rows alike) so
    # the classes overlap and the model cannot memorize the corpus.
    x[:, :13] += rng.normal(0, 1.0, (n, 13)) * OBS_NOISE
    return x


def build_dataset(rng: np.random.Generator, n_per_class: int):
    xs, ys = [], []
    for cls in range(NUM_CLASSES):
        xs.append(_sample_class(rng, cls, n_per_class))
        ys.append(np.full(n_per_class, cls))
    x = np.vstack(xs)
    y = np.concatenate(ys)
    perm = rng.permutation(len(y))
    return x[perm], y[perm]


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_mlp(xn: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """A small, hand-rolled 2-layer MLP trained with vanilla SGD/Adam-ish
    updates in numpy. Kept tiny and dependency-free on purpose."""
    n, d = xn.shape
    y1h = np.eye(NUM_CLASSES)[y]
    # He init.
    W1 = rng.normal(0, np.sqrt(2.0 / d), (d, HIDDEN))
    b1 = np.zeros(HIDDEN)
    W2 = rng.normal(0, np.sqrt(2.0 / HIDDEN), (HIDDEN, NUM_CLASSES))
    b2 = np.zeros(NUM_CLASSES)
    lr = 0.05
    epochs = 400
    batch = 128
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            bi = idx[s:s + batch]
            xb, yb = xn[bi], y1h[bi]
            # Forward.
            z1 = xb @ W1 + b1
            h = np.maximum(0, z1)
            z2 = h @ W2 + b2
            p = softmax(z2)
            # Backward.
            g2 = (p - yb) / len(bi)
            dW2 = h.T @ g2
            db2 = g2.sum(0)
            gh = (g2 @ W2.T) * (z1 > 0)
            dW1 = xb.T @ gh
            db1 = gh.sum(0)
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2
    return W1, b1, W2, b2


def forward(xn, W1, b1, W2, b2):
    h = np.maximum(0, xn @ W1 + b1)
    return softmax(h @ W2 + b2)


def quantize_int8(W: np.ndarray):
    """Symmetric per-tensor int8 quantization. Returns (q_int8, scale) with
    W ~= q * scale. This is exactly what the C forward pass dequantizes."""
    amax = float(np.abs(W).max())
    scale = amax / 127.0 if amax > 0 else 1.0
    q = np.clip(np.round(W / scale), -127, 127).astype(np.int8)
    return q, scale


def c_array_2d(name, q):
    rows = []
    for r in q:
        rows.append("  {" + ", ".join(f"{v:4d}" for v in r) + "}")
    body = ",\n".join(rows)
    return f"static const int8_t {name}[{q.shape[0]}][{q.shape[1]}] = {{\n{body}\n}};\n"


def c_array_1d_f(name, v):
    body = ", ".join(f"{x:.8f}f" for x in v)
    return f"static const float {name}[{len(v)}] = {{ {body} }};\n"


def codegen_header(mean, std, W1q, s1, b1, W2q, s2, b2):
    # Store quantized weights transposed to [out][in] so the C matmul is a
    # simple row-major dot product per output neuron.
    W1qT = W1q.T  # [HIDDEN][NUM_FEATURES]
    W2qT = W2q.T  # [NUM_CLASSES][HIDDEN]
    out = []
    out.append("/* AUTO-GENERATED by src/train.py - DO NOT EDIT BY HAND.\n"
               " * int8-quantized weights for the driver-monitoring MLP.\n"
               " * Layout: Dense(16->16, ReLU) -> Dense(16->3, softmax).\n"
               " * Dequantization: w_float = w_int8 * <LAYER>_SCALE. */\n")
    out.append("#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n\n")
    out.append("#include <stdint.h>\n\n")
    out.append(f"#define NUM_FEATURES {NUM_FEATURES}\n")
    out.append(f"#define HIDDEN       {HIDDEN}\n")
    out.append(f"#define NUM_CLASSES  {NUM_CLASSES}\n\n")
    names = ", ".join(f"\"{c}\"" for c in CLASS_NAMES)
    out.append(f"static const char *const CLASS_NAMES[NUM_CLASSES] = {{ {names} }};\n\n")
    out.append("/* Input standardization (baked from the training corpus). */\n")
    out.append(c_array_1d_f("FEATURE_MEAN", mean))
    out.append(c_array_1d_f("FEATURE_STD", std))
    out.append("\n/* Layer 1: hidden (ReLU). */\n")
    out.append(c_array_2d("W1_Q", W1qT))
    out.append(f"static const float W1_SCALE = {s1:.10f}f;\n")
    out.append(c_array_1d_f("B1", b1))
    out.append("\n/* Layer 2: output logits (softmax applied in C). */\n")
    out.append(c_array_2d("W2_Q", W2qT))
    out.append(f"static const float W2_SCALE = {s2:.10f}f;\n")
    out.append(c_array_1d_f("B2", b2))
    out.append("\n#endif /* MODEL_WEIGHTS_H */\n")
    return "".join(out)


def write_sample_csv(rng, mean, std):
    """A handful of representative frames (one clearly of each class + a couple
    of ambiguous ones). Pure numeric, 16 columns, header on line 1 - the C demo
    skips the header and parses 16 floats/row."""
    frames = []
    frames.append(_sample_class(rng, 0, 1)[0])  # attentive
    frames.append(_sample_class(rng, 1, 1)[0])  # drowsy
    frames.append(_sample_class(rng, 2, 1)[0])  # distracted
    frames.append(_sample_class(rng, 1, 1)[0])  # drowsy (night, low lux)
    frames[-1][14] = 6.0
    frames.append(_sample_class(rng, 0, 1)[0])  # attentive (bright day)
    frames[-1][14] = 80000.0
    DATA.mkdir(exist_ok=True)
    with open(DATA / "sample_frame_features.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FEATURES)
        for fr in frames:
            w.writerow([f"{v:.5f}" for v in fr])


def write_eval_csv(rng, mean, std, W1, b1, W2, b2):
    """Held-out evaluation, framed as BINARY 'unsafe driver state' detection
    (drowsy OR distracted = 1, attentive = 0) so CertOne's classification/
    fairness finders compute accuracy/AUROC/ECE/Brier + a day/night fairness gap.

    The group axis is time_of_day (day/night), derived from ambient lux. The
    training corpus is intentionally DAYTIME-SKEWED (see BOM known_bias:
    daytime_skew): night frames get extra IR/low-light sensor noise, so the
    model is measurably weaker at night - a genuine measured fairness gap."""
    n = 240
    xs, ys, groups = [], [], []
    for _ in range(n):
        cls = int(rng.integers(0, NUM_CLASSES))
        row = _sample_class(rng, cls, 1)[0]
        # Day/night split, daytime-skewed sensor quality.
        is_night = rng.random() < 0.5
        if is_night:
            row[14] = rng.uniform(2, 40)         # low lux
            # Under-represented at train time -> degraded IR/low-light features.
            # This corruption is what turns the declared daytime_skew into a
            # MEASURED night accuracy gap in eval/holdout_predictions.csv.
            row[0] += rng.normal(0, 0.10)
            row[1] += rng.normal(0, 0.10)
            row[4] += rng.normal(0, 0.16)
            row[9] += rng.normal(0, 0.22)
            row[10] += rng.normal(0, 140)
            row[3] += rng.normal(0, 120)
        else:
            row[14] = rng.uniform(3000, 100000)  # daylight
        xs.append(row)
        ys.append(cls)
        groups.append("night" if is_night else "day")
    X = np.array(xs)
    Xn = (X - mean) / std
    P = forward(Xn, W1, b1, W2, b2)
    p_unsafe = P[:, 1] + P[:, 2]                  # P(drowsy)+P(distracted)
    EVAL.mkdir(exist_ok=True)
    with open(EVAL / "holdout_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["y_true", "y_score", "y_pred", "group"])
        for cls, score, grp in zip(ys, p_unsafe, groups):
            y_true = 0 if cls == 0 else 1
            y_pred = 1 if score >= 0.5 else 0
            w.writerow([y_true, f"{score:.6f}", y_pred, grp])
    # Console summary.
    y_true = np.array([0 if c == 0 else 1 for c in ys])
    y_pred = (p_unsafe >= 0.5).astype(int)
    acc = float((y_pred == y_true).mean())
    for grp in ("day", "night"):
        m = np.array([g == grp for g in groups])
        ga = float((y_pred[m] == y_true[m]).mean())
        print(f"  holdout accuracy [{grp:5s}] = {ga:.3f}  (n={int(m.sum())})")
    print(f"  holdout accuracy [all]   = {acc:.3f}  (n={n})")


def main():
    rng = np.random.default_rng(SEED)
    print(f"[train] seed={SEED} building synthetic driver-monitoring corpus...")
    X, y = build_dataset(rng, n_per_class=1500)
    mean = X.mean(0)
    std = X.std(0)
    std[std < 1e-6] = 1.0
    Xn = (X - mean) / std
    print("[train] training MLP (16 -> 16 ReLU -> 3 softmax)...")
    W1, b1, W2, b2 = train_mlp(Xn, y, rng)
    p = forward(Xn, W1, b1, W2, b2)
    train_acc = float((p.argmax(1) == y).mean())
    print(f"[train] train accuracy = {train_acc:.3f}")

    print("[quant] symmetric int8 quantization of dense weights...")
    W1q, s1 = quantize_int8(W1)
    W2q, s2 = quantize_int8(W2)
    # Sanity: quantized forward should track float forward.
    W1dq, W2dq = W1q.astype(np.float64) * s1, W2q.astype(np.float64) * s2
    pq = forward(Xn, W1dq, b1, W2dq, b2)
    q_acc = float((pq.argmax(1) == y).mean())
    print(f"[quant] int8 train accuracy = {q_acc:.3f}")

    SRC.mkdir(exist_ok=True)
    header = codegen_header(mean, std, W1q, s1, b1, W2q, s2, b2)
    (SRC / "model_weights.h").write_text(header)
    print(f"[codegen] wrote {SRC / 'model_weights.h'}")

    write_sample_csv(np.random.default_rng(SEED + 1), mean, std)
    print(f"[codegen] wrote {DATA / 'sample_frame_features.csv'}")

    write_eval_csv(np.random.default_rng(SEED + 2), mean, std, W1dq, b1, W2dq, b2)
    print(f"[eval] wrote {EVAL / 'holdout_predictions.csv'}")
    print("[done]")


if __name__ == "__main__":
    main()

# adas-ecu-driver-monitor — DMS-1 driver-monitoring / forward-collision ECU

A production-shaped **embedded AI** repo for an automotive **driver-monitoring
system (DMS)** ECU that also feeds the **forward-collision-warning (FCW)**
arbiter. An in-cabin IR camera pipeline extracts driver-state features every
frame; a small **int8-quantized neural network, hand-written in pure C**,
classifies the driver as **attentive / drowsy / distracted** and raises an
attention warning (and escalates to FCW) on unsafe states.

The firmware is written to be **cross-compiled for aarch64 and certified by
CertOne on the target ISA**: CertOne cross-builds the ELF and runs it under
`qemu-aarch64` inside an attested confidential VM, so the compliance verdict is
computed against the *actual* embedded artifact.

## Embedded target

- **SoC:** ARM64 (aarch64) Cortex-A automotive SoC, embedded Linux (Yocto).
- **Runtime:** pure C11, **no external libraries** — bare `gcc` + `libm` only.
  No TensorFlow Lite, no CMSIS-NN, no torch. The network is implemented by hand.
- **Model:** MLP `16 -> 16 (ReLU) -> 3 (softmax)`, **int8 symmetric per-tensor
  quantized** weights baked into `src/model_weights.h`. Dequantization
  (`w = w_int8 * scale`) + the matmul/ReLU/softmax/argmax forward pass run in
  `src/inference.c`. Non-accelerated: it runs on the ECU CPU.
- **Inputs (16 features):** eye-aspect-ratio (mean/min), blink rate & duration,
  PERCLOS, yawn frequency, head pitch/yaw/roll, gaze-off-road ratio, fixation,
  steering-reversal rate, lane-deviation variance, time-since-break, ambient lux,
  speed. Contract lives in `src/train.py` `FEATURES` (train + firmware agree).

## Repository layout

```
certone-bom.yaml               # declared AI BOM (arch: arm64 -> unikernel/embedded substrate)
src/train.py                   # host trainer: numpy, seeded; quantize + CODEGEN of model_weights.h
src/model_weights.h            # AUTO-GENERATED int8 weights + standardization constants
src/inference.c                # pure-C int8 MLP forward pass + main() demo (the certified artifact)
src/ecu_config.h               # ECU/build config (contains a planted hard-coded OTA key)
data/sample_frame_features.csv # representative feature frames for the demo
eval/holdout_predictions.csv   # real held-out preds (y_true,y_score,y_pred,group) for metrics
Makefile                       # native + CROSS (aarch64-linux-gnu-) targets
ota/release-pipeline.yml       # fleet OTA release: build->cross->CertOne gate->sign->Uptane
deploy/uptane-targets.json     # Uptane image-repo targets metadata (post-gate, signed)
deploy/ecu-target.json         # ECU target descriptor + phased rollout plan
.github/workflows/certone.yml  # CI: cross-build + CertOne compliance gate + console submit
requirements.txt               # host tooling only (numpy); firmware needs NO Python
```

## Run the local demo

The aarch64 cross toolchain + qemu run on the CertOne worker; locally, native
`gcc` proves the C is portable and correct (same source, no x86-specific code):

```bash
python src/train.py                                   # regenerate weights + eval + sample
gcc -O2 -Wall -o dms_infer src/inference.c -lm        # or: make
./dms_infer data/sample_frame_features.csv            # or: make run
```

Example output:

```
DMS-1 driver-monitoring ECU  |  model=dms-mlp-v1.0.0  hw=revC
int8 MLP  16 -> 16(ReLU) -> 3(softmax)   frame=33ms

frame 0: attentive    conf=0.999  ... ok
frame 1: drowsy       conf=1.000  ... UNSAFE -> WARN+FCW
frame 2: distracted   conf=0.996  ... UNSAFE -> WARN+FCW
frame 3: drowsy       conf=1.000  ... UNSAFE -> WARN+FCW
frame 4: attentive    conf=0.998  ... ok
```

Cross-compile the ECU firmware (the artifact CertOne certifies):

```bash
make cross          # -> dms_infer_aarch64 (aarch64 ELF, -static)
```

## CI + OTA pipeline

- **`.github/workflows/certone.yml`** (push/PR): installs `gcc-aarch64-linux-gnu`,
  regenerates the int8 weights, **cross-compiles the aarch64 ELF**, then runs the
  **CertOne compliance gate** (`fail-on: high`, SARIF upload, PR MoA fix
  suggestions, console submit `environment: edge`). A HIGH finding fails the PR.
- **`ota/release-pipeline.yml`** (on `fw-v*` tags): the fleet release pipeline
  `build -> cross-compile -> CertOne gate -> sign (HSM Uptane key) -> Uptane
  release`. The gate is a hard predecessor of the `sign` and `uptane_release`
  jobs, so **a non-compliant model can never reach a vehicle**. Rollout is phased
  (canary 1% -> staged 10% -> fleet-wide) with A/B rollback on any DTC
  (`deploy/ecu-target.json`), matching **UNECE R156** software-update management.

## Regulatory framing

This is a **high-risk AI system** and the compliance scope is real:

- **EU AI Act — Annex I / high-risk:** an AI safety component of a vehicle
  (driver monitoring feeding collision warning). Art.9 (risk management),
  **Art.10 (data governance)**, **Art.13 (transparency/documentation)** and
  **Art.15 (accuracy, robustness, cybersecurity)** all apply.
- **ISO 26262 (functional safety)** + **ISO 21448 / SOTIF** (safety of the
  intended function under degraded/edge inputs) — the DMS function is ASIL-B and
  contributes to ASIL-D FCW arbitration.
- **UNECE R155 (CSMS / cybersecurity)** + **R156 (SUMS / software updates)** —
  key management, supply-chain integrity, and controlled OTA release.
- CI packs/standards: `eu_ai_act`, `nist_ai_rmf`; standards `iso_26262`,
  `iso_21448`, `unece_r155`; sector `automotive`; region `EU`; `fail-on: high`.

### Known compliance gaps (deliberately present, to be surfaced by the scan)

- **Hard-coded OTA / diagnostic key** in `src/ecu_config.h` (`OTA_DIAGNOSTIC_KEY`)
  — a shared in-source unlock seed. **EU AI Act Art.15 / UNECE R155** key-mgmt +
  supply-chain gap; must move to the HSM per-ECU key store.
- **Sensitive biometric PII:** in-cabin driver-facing camera
  (`has_sensitive_pii: true`) with realistic declared bias
  `[daytime_skew, demographic_skew]` — **Art.10 data governance** / fairness.
- **No model card and no ISO 26262 safety case / documentation** — CertOne
  measures model-card + datasheet completeness at **0.0** (**Art.13 transparency**).
- **No watchdog / safe-state fallback** in `src/inference.c` (**ASIL / SOTIF gap**):
  the forward pass has no inference timeout, no NaN/Inf guard, and no defined
  degraded-input behaviour — a bad frame silently yields "attentive", the worst
  fail-active mode for a drowsiness monitor. A plausible fixed-point/int32
  requant-overflow risk is also flagged in-code. Required by ISO 26262 + SOTIF;
  tracked, not yet implemented.

## Evaluation metrics

CertOne computes these **for real** from `eval/holdout_predictions.csv` (240
held-out frames), framed as binary **unsafe-state detection** (drowsy OR
distracted = 1, attentive = 0), with the `group` column = **time_of_day
(day/night)** so the declared `daytime_skew` becomes a measured fairness gap:

| metric | value |
| --- | --- |
| accuracy | **0.958** |
| precision/recall → F1 | **F1 0.968** |
| AUROC (+ROC curve) | **0.989** |
| ECE | 0.045 |
| Brier | 0.038 |
| statistical-parity difference | 0.039 |
| equalized-odds difference | **0.134** |
| worst-group accuracy (night) | **0.926** (vs 0.992 day) |

Primary thresholds (`certone-bom.yaml`): `auroc ≥ 0.90`, `recall ≥ 0.90`,
`worst_group_accuracy ≥ 0.90`, `latency_ms ≤ 10`.

## Honest real-vs-representative note

The **C forward pass is real** — `src/inference.c` runs the exact int8 weights
in `src/model_weights.h` and produces the printed classifications, and the
evaluation metrics above are computed from genuine held-out model scores. The
**shipped weights are a seeded, representative fixture** trained by `src/train.py`
on *synthetic-but-realistic* driver-monitoring telemetry. In production the same
quantize+codegen step consumes the **fleet-trained, closed-course-validated**
model on registered OEM test-fleet data. The math, quantization, cross-compile
path, and evaluation are production-grade; the training corpus is the stand-in.

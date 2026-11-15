# ADAS Forward-Perception

Forward-collision and vulnerable-road-user (VRU) perception for an **SAE L2+**
driver-assistance ECU. A single forward-facing camera frame is run through a
real ONNX object detector and the road actors relevant to forward-collision
warning (FCW) are decoded: **pedestrians, cyclists, vehicles, and signs**.

- **Model:** Tiny-YOLOv3 (COCO-80), frozen ONNX graph with in-graph NMS.
  Deployed subset: `person, bicycle, motorcycle, car, bus, truck, traffic light,
  stop sign`.
- **Runtime:** `onnxruntime` on CPU, no network at inference. Designed to run
  inside an **AMD SEV-SNP** confidential VM on the ECU.
- **Input:** 1080p camera frame, aspect-preserving letterbox to 416x416.

## Repository layout

```
certone-bom.yaml              AI bill of materials (model + dataset + build)
src/perception.py             runnable ONNX inference (load/preprocess/detect)
models/tiny-yolov3-11.onnx    the detector (real, ~34 MB)
data/sample_street.jpg        sample forward-camera frame
requirements.txt              lightweight CPU runtime deps
azure-ml/pipeline.yml         Azure ML CI: prep -> train -> eval -> register
.github/workflows/certone.yml CertOne compliance gate on push/PR
deploy/                       Azure IoT Edge manifest + Dockerfile + ECU config
```

## Run the local demo

```bash
pip install -r requirements.txt
python src/perception.py
```

It loads the local model, runs inference on `data/sample_street.jpg`, prints the
decoded ADAS detections (class, confidence, box) plus the FCW summary, in
< 5 s on CPU with no network access.

Public entrypoints (probed by CertOne's dynamic test-scenario generator):
`load_model()`, `preprocess(path)`, `detect(frame)`.

## CI / CD

- **Azure ML pipeline** (`azure-ml/pipeline.yml`): a CLI v2 pipeline job
  `prep -> train -> evaluate -> register` on a CPU compute cluster. A
  `certone check` step runs as a **compliance gate before model registration** —
  a verdict at or above `high` blocks promotion to the registry.
- **GitHub Actions** (`.github/workflows/certone.yml`): on every push / PR the
  `certkor/certone-action` runs the same compliance gate, writes `report.json`
  and `certone.sarif`, uploads the SARIF to code scanning, and (on PRs) posts
  Method-of-Amendment fix suggestions inline. `fail-on: high` gates the merge.

## Deployment target — edge (automotive ECU)

Deployed as an **Azure IoT Edge** module (`deploy/deployment.json`,
`deploy/Dockerfile`) onto the ECU. The perception container runs on CPU inside
an AMD SEV-SNP confidential VM; `deploy/ecu_config.yaml` carries the runtime and
fleet-telemetry configuration. Training/CI runs in Azure ML; only the frozen
ONNX graph ships to the edge.

## Regulatory framing

This is a **vehicle safety component**, so the perception function is
**EU AI Act Annex I high-risk**. Applicable obligations and standards:

- **EU AI Act** — Annex I (safety component of a regulated product); Art.10 data
  governance, Art.13 transparency, Art.15 accuracy/robustness/cybersecurity.
- **NIST AI RMF** — Map/Measure/Manage for the perception function.
- **UNECE R155 / R156** — vehicle cybersecurity management and software-update
  management for the ECU and its OTA lifecycle.
- **ISO 21448 (SOTIF)** — safety of the intended functionality; management of
  perception triggering conditions and performance limitations.

Known limitations affecting SOTIF and fairness are recorded in
`certone-bom.yaml` (`known_bias: [daytime_skew, single_region_roads]`,
`has_sensitive_pii: true`).

## Evaluation metrics

CertOne computes these from `eval/detections_eval.csv` (regenerate: `python src/evaluate.py`).
The deployed path is the real Tiny-YOLOv3 ONNX detector in `src/perception.py`; detection is
scored the standard binary way (`y_true` = IoU>=0.5 match to a VRU/vehicle box, `y_score` =
confidence) over a **representative held-out clip fixture** — see the docstring. Measured over 150
candidate detections: **accuracy 0.85**, F1 0.89, **precision 0.89 / recall 0.88**, **AUROC 0.92**
(+ROC), ECE 0.10, Brier 0.12, and across `time_of_day`: SPD 0.23, equalized-odds 0.12,
**worst-group accuracy 0.78** (night) — the declared `daytime_skew` bias made measurable. These
binary operating points underpin the model card's precision/recall and mAP@0.5.

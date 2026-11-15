# Contoso Mobility — ADAS

- `adas-ecu-driver-monitor/`: driver-monitoring ECU firmware (pure-C int8 model, arm64).
- `adas-perception/`: forward-perception object detector (Tiny-YOLOv3, ONNX).

Every pull request to `main` is certified by CertOne (`.github/workflows/certone.yml`): each AI system
is scanned in a confidential VM and its check fails unless the verdict passes.

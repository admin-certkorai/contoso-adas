# DMS-1 driver-monitoring classifier v1.1 model card

## Intended use

Classifies the driver state (attentive, drowsy, distracted) from in-cabin camera features on the
DMS-1 ECU, to trigger an HMI warning and escalate to the forward-collision-warning arbiter. It
assists the driver; it never takes vehicle control.

## Out of scope

- Identifying or authenticating drivers.
- Use on hardware other than the DMS-1 ECU (aarch64 Cortex-A, embedded Linux).
- Sole basis for any automated driving decision.

## Model

- int8 MLP, 16 inputs, 16 hidden units (ReLU), 3 classes (softmax), symmetric per-tensor
  quantisation, baked into the firmware (`src/model_weights.h`).
- Runs every 33 ms frame on the ECU CPU, with no accelerator.

## Training data

- Registered OEM driver-monitoring test fleet (in-cabin IR camera) and a closed-course drowsiness
  and distraction protocol. The shipped sample is synthetic but realistic feature telemetry.
- Only derived per-frame features are stored; raw face frames and identifiers never leave the ECU.

## Evaluation

240 held-out frames (`eval/holdout_predictions.csv`), unsafe state (drowsy or distracted) against
attentive, grouped by time of day.

| Metric | Value |
| --- | --- |
| Accuracy | 0.958 |
| F1 | 0.968 |
| AUROC | 0.989 |
| Worst-group accuracy (night) | 0.926 |

Release thresholds: AUROC at least 0.90, recall at least 0.90, worst-group accuracy at least 0.90.

## Limitations and risks

- Night and low-light frames perform worse than daylight frames (0.926 against 0.992).
- Eyewear, face geometry and skin-tone coverage is uneven in the fleet data.
- The shipped weights are a representative fixture; production weights come from the
  fleet-trained, closed-course-validated model.

## Safety behaviour

When the camera feature input is missing or carries no valid frame, the monitor raises a fault
DTC, lights the driver-monitoring telltale, suspends its driver-state output, hands back to the
driver and exits with a dedicated safe-state code (ISO 26262, ISO 21448).

## Change log

- v1.1 (`dms-mlp-v1.1.0`): safe state on acquisition faults, feature-only dataset, documented bias
  mitigation, this model card.
- v1.0 (`dms-mlp-v1.0.0`): first release.

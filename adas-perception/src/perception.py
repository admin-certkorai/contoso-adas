"""ADAS forward-perception inference (SAE L2+ driver-assistance ECU).

Runs a real ONNX object detector (Tiny-YOLOv3, COCO-80) on a forward-facing
camera frame and decodes the road actors relevant to forward-collision warning
and vulnerable-road-user (VRU) protection: pedestrians, vehicles and signs.

Runtime: onnxruntime + numpy + pillow only. CPU, no network at inference.

Public entrypoints (probed by the CertOne dynamic test-scenario generator):
    load_model()        -> onnxruntime.InferenceSession
    preprocess(path)    -> (input_chw, image_shape)  model-ready tensors
    detect(frame)       -> list[Detection]           decoded detections
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image

# --- paths -----------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
MODEL_PATH = os.path.join(_ROOT, "models", "tiny-yolov3-11.onnx")
SAMPLE_IMAGE = os.path.join(_ROOT, "data", "sample_street.jpg")

# Tiny-YOLOv3 fixed input side (must be a multiple of 32).
INPUT_SIZE = 416

# COCO-80 label map (detector was trained on COCO).
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# ADAS-relevant COCO classes, grouped for the forward-perception stack.
VRU_CLASSES = {"person", "bicycle", "motorcycle"}
VEHICLE_CLASSES = {"car", "bus", "truck", "train"}
SIGN_CLASSES = {"traffic light", "stop sign"}
ADAS_CLASSES = VRU_CLASSES | VEHICLE_CLASSES | SIGN_CLASSES

# Detection confidence gate for the safety pipeline.
CONF_THRESHOLD = 0.30


@dataclass
class Detection:
    label: str
    confidence: float
    box_xyxy: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in original pixels
    category: str                        # vru | vehicle | sign | other

    def __repr__(self) -> str:  # compact, log-friendly
        x1, y1, x2, y2 = self.box_xyxy
        return (f"Detection(label={self.label!r} conf={self.confidence:.2f} "
                f"cat={self.category} box=({x1},{y1},{x2},{y2}))")


def _category(label: str) -> str:
    if label in VRU_CLASSES:
        return "vru"
    if label in VEHICLE_CLASSES:
        return "vehicle"
    if label in SIGN_CLASSES:
        return "sign"
    return "other"


def load_model(model_path: str = MODEL_PATH) -> ort.InferenceSession:
    """Load the local ONNX detector. No network access is performed."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model not found: {model_path}")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        model_path, sess_options=so, providers=["CPUExecutionProvider"]
    )


def _letterbox(image: Image.Image, size: int = INPUT_SIZE) -> Image.Image:
    """Resize preserving aspect ratio, pad to a square with mid-gray."""
    iw, ih = image.size
    scale = min(size / iw, size / ih)
    nw, nh = int(round(iw * scale)), int(round(ih * scale))
    resized = image.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def preprocess(path: str = SAMPLE_IMAGE) -> Tuple[np.ndarray, np.ndarray]:
    """Load an image frame and produce the two Tiny-YOLOv3 input tensors.

    Returns:
        input_chw:   float32 [1, 3, 416, 416], RGB, letterboxed, scaled to 0..1
        image_shape: float32 [1, 2] original (height, width) for box rescaling
    """
    image = Image.open(path).convert("RGB")
    iw, ih = image.size
    boxed = _letterbox(image, INPUT_SIZE)
    arr = np.asarray(boxed, dtype=np.float32) / 255.0     # HWC, 0..1
    arr = np.transpose(arr, (2, 0, 1))                    # CHW
    input_chw = np.expand_dims(arr, 0).astype(np.float32)  # NCHW
    image_shape = np.array([[ih, iw]], dtype=np.float32)
    return input_chw, image_shape


def detect(
    frame: Tuple[np.ndarray, np.ndarray],
    session: Optional[ort.InferenceSession] = None,
    conf_threshold: float = CONF_THRESHOLD,
) -> List[Detection]:
    """Run detection on a preprocessed frame and decode ADAS-relevant actors.

    Args:
        frame: (input_chw, image_shape) as returned by preprocess().
        session: optional pre-loaded session (loaded on demand otherwise).
        conf_threshold: minimum confidence to keep a detection.
    """
    input_chw, image_shape = frame
    sess = session or load_model()
    in_names = [i.name for i in sess.get_inputs()]
    boxes, scores, indices = sess.run(
        None, {in_names[0]: input_chw, in_names[1]: image_shape}
    )

    detections: List[Detection] = []
    # indices: (batch_idx, class_idx, box_idx) rows selected by in-graph NMS.
    for row in np.asarray(indices).reshape(-1, 3):
        b, c, k = int(row[0]), int(row[1]), int(row[2])
        conf = float(scores[b, c, k])
        if conf < conf_threshold:
            continue
        y1, x1, y2, x2 = boxes[b, k]  # model emits (y1, x1, y2, x2)
        label = COCO_CLASSES[c]
        detections.append(
            Detection(
                label=label,
                confidence=conf,
                box_xyxy=(int(x1), int(y1), int(x2), int(y2)),
                category=_category(label),
            )
        )
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def _main() -> None:
    import time

    print(f"[perception] loading model: {os.path.relpath(MODEL_PATH, _ROOT)}")
    session = load_model()
    print(f"[perception] frame: {os.path.relpath(SAMPLE_IMAGE, _ROOT)}")
    frame = preprocess(SAMPLE_IMAGE)

    t0 = time.perf_counter()
    detections = detect(frame, session=session)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    adas = [d for d in detections if d.label in ADAS_CLASSES]
    print(f"\n[perception] inference latency: {latency_ms:.1f} ms (CPU)")
    print(f"[perception] {len(detections)} total detections, "
          f"{len(adas)} ADAS-relevant (VRU / vehicle / sign):\n")
    for d in adas:
        print("  " + repr(d))

    warn = [d for d in adas if d.category in ("vru", "vehicle")]
    if warn:
        nearest = max(warn, key=lambda d: (d.box_xyxy[3] - d.box_xyxy[1]))
        print(f"\n[FCW] forward-collision-relevant actors in frame: {len(warn)}; "
              f"largest/most-proximate: {nearest.label} "
              f"(conf {nearest.confidence:.2f})")
    else:
        print("\n[FCW] no forward-collision-relevant actors detected")


if __name__ == "__main__":
    _main()

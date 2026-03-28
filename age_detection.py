"""
age_detection.py
----------------
Lightweight age estimation using:
  - OpenCV DNN (Caffe)  →  face detection   (~5 MB model)
  - ONNX Runtime        →  age estimation   (~8 MB model)

No DeepFace, no TensorFlow, no PyTorch.
Works on laptop webcam now; drop-in ready for Pi Camera later.

FIRST-TIME SETUP — run once in your terminal:
---------------------------------------------
    pip install opencv-python onnxruntime
    python age_detection.py --download

Then use normally:
    python age_detection.py          # quick webcam test
    from age_detection import detect_age, get_age_group, modify_response_by_age
"""

import os
import sys
import time
import cv2
import numpy as np

# ── Model paths (kept next to this file, or set env var VEDAM_MODELS_DIR) ──
_MODELS_DIR = os.environ.get(
    "VEDAM_MODELS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
)

FACE_PROTO   = os.path.join(_MODELS_DIR, "deploy.prototxt")
FACE_MODEL   = os.path.join(_MODELS_DIR, "face_detector.caffemodel")
AGE_MODEL    = os.path.join(_MODELS_DIR, "age_gender.onnx")


# ─────────────────────────────────────────────────────────────────────────────
# 1. MODEL DOWNLOAD HELPER  (run once: python age_detection.py --download)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_URLS = {
    "deploy.prototxt": (
        "https://raw.githubusercontent.com/opencv/opencv/master/"
        "samples/dnn/face_detector/deploy.prototxt"
    ),
    "face_detector.caffemodel": (
        "https://github.com/opencv/opencv_3rdparty/raw/"
        "dnn_samples_face_detector_20180205_fp16/"
        "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    ),
    "age_gender.onnx": (
        "https://storage.googleapis.com/ailia-models/"
        "age-gender-recognition-retail/"
        "age-gender-recognition-retail-0013.onnx"
    ),
}


def download_models():
    """Download all required model files (~13 MB total). Run once."""
    import urllib.request

    os.makedirs(_MODELS_DIR, exist_ok=True)
    print(f"[Setup] Downloading models to: {_MODELS_DIR}")

    for filename, url in MODEL_URLS.items():
        dest = os.path.join(_MODELS_DIR, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print(f"  ✓ {filename} already exists, skipping.")
            continue
        print(f"  ↓ Downloading {filename} …", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, dest)
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"done ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"FAILED\n    Error: {e}")
            print(f"    Please download manually from:\n    {url}")
            print(f"    and save to: {dest}")

    print("[Setup] Done.")


def _models_ready():
    """Return True if all model files exist and are non-empty."""
    return all(
        os.path.exists(p) and os.path.getsize(p) > 1000
        for p in [FACE_PROTO, FACE_MODEL, AGE_MODEL]
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL LOADING  (cached at module level — loaded once per process)
# ─────────────────────────────────────────────────────────────────────────────

_face_net = None
_age_sess = None


def _load_models():
    """Load face detector and age ONNX model. Called lazily on first use."""
    global _face_net, _age_sess

    if _face_net is not None and _age_sess is not None:
        return True                         # already loaded

    if not _models_ready():
        print("[AgeDetection] Model files missing. Run: python age_detection.py --download")
        return False

    try:
        import onnxruntime as ort

        _face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
        _age_sess = ort.InferenceSession(AGE_MODEL, providers=["CPUExecutionProvider"])
        print("[AgeDetection] Models loaded successfully.")
        return True

    except Exception as e:
        print(f"[AgeDetection] Failed to load models: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 3. FACE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _detect_faces(frame, conf_threshold=0.7):
    """
    Detect faces in a BGR frame using OpenCV DNN (SSD + ResNet10).
    Returns list of (x, y, w, h) bounding boxes.
    """
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0)
    )
    _face_net.setInput(blob)
    detections = _face_net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < conf_threshold:
            continue
        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)

        # Clamp to frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2 - x1, y2 - y1))

    return boxes


# ─────────────────────────────────────────────────────────────────────────────
# 4. AGE ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

# Age ranges corresponding to the 8 output classes of the ONNX model
_AGE_CLASSES = [
    (0,  2),   # class 0
    (4,  6),   # class 1
    (8,  12),  # class 2
    (15, 20),  # class 3
    (25, 32),  # class 4
    (38, 43),  # class 5
    (48, 53),  # class 6
    (60, 100), # class 7
]

def _get_model_input_format():
    """
    Introspect the ONNX model to determine expected input shape.
    Returns 'NHWC' or 'NCHW'.
    """
    inp = _age_sess.get_inputs()[0]
    shape = inp.shape  # e.g. [1, 62, 62, 3] or [1, 3, 62, 62]
    # shape[1] == 3 means channels-first (NCHW); shape[3] == 3 means NHWC
    if len(shape) == 4:
        if shape[3] == 3:
            return "NHWC"
        elif shape[1] == 3:
            return "NCHW"
    return "NHWC"  # safe default for this model family


def _estimate_age_onnx(face_img):
    """
    Run the ONNX age model on a cropped face image.
    Returns estimated age (int) or None.

    age-gender-recognition-retail-0013 specifics:
      Input  : (1, 62, 62, 3)  NHWC, float32, range [0, 255]
      Output : [fc3_gender (1,2), fc3_age (1,1)]
               fc3_age is a raw scalar — multiply by 100 to get years
    """
    try:
        inp_meta   = _age_sess.get_inputs()[0]
        input_name = inp_meta.name
        fmt        = _get_model_input_format()

        face_resized = cv2.resize(face_img, (62, 62))
        face_rgb     = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        blob         = face_rgb.astype(np.float32)  # keep [0, 255] range

        if fmt == "NHWC":
            # (62, 62, 3) → (1, 62, 62, 3)
            blob = np.expand_dims(blob, axis=0)
        else:
            # (62, 62, 3) → (3, 62, 62) → (1, 3, 62, 62)
            blob = np.transpose(blob, (2, 0, 1))
            blob = np.expand_dims(blob, axis=0)

        outputs = _age_sess.run(None, {input_name: blob})

        # Debug: print output shapes on first run to help diagnose future issues
        # Uncomment if needed:
        # for i, o in enumerate(outputs):
        #     print(f"  output[{i}] shape={o.shape} value={o}")

        # age-gender-recognition-retail-0013 layout:
        #   outputs[0] → gender  (1, 2)   — ignored
        #   outputs[1] → age     (1, 1)   — raw scalar ~[0, 1], multiply by 100
        if len(outputs) >= 2:
            raw_age = float(np.squeeze(outputs[1]))
            age = int(raw_age * 100) if raw_age <= 1.0 else int(raw_age)
            return max(1, min(age, 100))

        # Fallback: if only one output, check if it's a class distribution
        if len(outputs) == 1:
            out = np.squeeze(outputs[0])
            if out.ndim == 1 and len(out) == len(_AGE_CLASSES):
                # Softmax class output → pick highest class → return midpoint
                best_class = int(np.argmax(out))
                lo, hi = _AGE_CLASSES[best_class]
                return (lo + hi) // 2

        return None

    except Exception as e:
        print(f"[AgeDetection] ONNX inference error: {e}")
        # Print model input info to help debug
        try:
            inp = _age_sess.get_inputs()[0]
            print(f"  Model input name : {inp.name}")
            print(f"  Model input shape: {inp.shape}")
            print(f"  Model input type : {inp.type}")
        except Exception:
            pass
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 5. PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def detect_age(camera_index=0, warmup_frames=5):
    """
    Open the webcam, capture a frame, detect the largest face, estimate age.

    Args:
        camera_index  : 0 for laptop webcam; change for external/Pi cam
        warmup_frames : frames to discard so auto-exposure stabilises

    Returns:
        int   — estimated age
        None  — if camera, face, or model unavailable
    """
    if not _load_models():
        return None

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[AgeDetection] Could not open camera index {camera_index}.")
        return None

    # Warm-up: let the sensor auto-adjust
    for _ in range(warmup_frames):
        cap.read()
        time.sleep(0.05)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("[AgeDetection] Failed to capture frame.")
        return None

    faces = _detect_faces(frame)
    if not faces:
        print("[AgeDetection] No face detected in frame.")
        return None

    # Use the largest detected face (closest to camera)
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    face_crop   = frame[y:y+h, x:x+w]

    age = _estimate_age_onnx(face_crop)
    print(f"[AgeDetection] Detected age: {age}")
    return age


def get_age_group(age):
    """Map numeric age to a group label."""
    if age is None:
        return "unknown"
    elif age < 13:
        return "child"
    elif age < 20:
        return "teen"
    elif age < 50:
        return "adult"
    else:
        return "senior"

# ─────────────────────────────────────────────────────────────────────────────
# 6. CLI  — quick test & first-time model download
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--download" in sys.argv:
        download_models()
        sys.exit(0)

    print("=== VedaMitra Age Detection Test ===")
    print(f"Models directory: {_MODELS_DIR}\n")

    if not _models_ready():
        print("Models not found. Run first:\n  python age_detection.py --download\n")
        sys.exit(1)

    age       = detect_age()
    group     = get_age_group(age)
    sample    = "The concept of Dharma is central to Hindu philosophy."
    modified  = modify_response_by_age(sample, group)

    print(f"\nResult:")
    print(f"  Age detected : {age}")
    print(f"  Age group    : {group}")
    print(f"  Sample output: {modified}")

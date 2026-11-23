# input_handler.py
# Camera + MediaPipe wrapper. Outputs HandState only — no game logic.
# Camera capture runs on a background thread so cap.read() never stalls
# the main game loop and keyboard input stays responsive.

import cv2
import mediapipe as mp
import math
import os
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))


@dataclass
class HandState:
    # Normalised wrist coords (0–1 relative to frame)
    left_wrist:       Optional[Tuple[float, float]] = None   # bow hand
    right_wrist:      Optional[Tuple[float, float]] = None   # string hand

    # Pixel wrist coords in the camera frame (for overlay drawing)
    left_wrist_px:    Optional[Tuple[int, int]] = None
    right_wrist_px:   Optional[Tuple[int, int]] = None

    right_palm_open:  bool  = False
    both_fists:       bool  = False   # both hands closed → ready pose

    draw_power:       float = 0.0    # 0.0–1.0, based on distance between wrists
    angle_deg:        float = 0.0    # 0–85°, based on RIGHT hand height

    both_visible:     bool  = False
    finger_count:     int   = 0      # 0–5, used for menu gesture select
    frame_bgr:        object = None  # latest camera frame (BGR, already rotated/flipped)


# ── Tuning ────────────────────────────────────────────────────────────
MAX_PULL_NORM = 0.55   # normalised distance between wrists = full power
MIN_PULL_NORM = 0.08   # below this = no power registered

# Angle: right wrist y normalised.
# When right wrist y_norm = ANGLE_TOP_Y  → 85°  (hand raised high)
# When right wrist y_norm = ANGLE_BOT_Y  → 0°   (hand at waist)
ANGLE_TOP_Y = 0.15     # top of useful range (hand raised)
ANGLE_BOT_Y = 0.80     # bottom of useful range (hand low)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _is_fist(lms) -> bool:
    """True when all 4 fingers are curled (tip below mid-pip)."""
    tips  = [8, 12, 16, 20]
    bases = [6, 10, 14, 18]
    curled = sum(1 for t, b in zip(tips, bases)
                 if lms.landmark[t].y > lms.landmark[b].y)
    return curled >= 3


def _is_palm_open(lms) -> bool:
    """True when at least 3 of 4 fingers are extended."""
    tips  = [8, 12, 16, 20]
    bases = [6, 10, 14, 18]
    count = sum(1 for t, b in zip(tips, bases)
                if lms.landmark[t].y < lms.landmark[b].y)
    return count >= 3


def _count_fingers(lms) -> int:
    """Count extended fingers (0–5) for menu gesture."""
    lm    = lms.landmark
    count = 0
    # Thumb: horizontal spread
    if abs(lm[4].x - lm[3].x) > 0.03:
        count += 1
    # Four fingers: tip above pip
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        if lm[tip].y < lm[pip].y:
            count += 1
    return count


class InputHandler:
    def __init__(self, camera_index: int = 0, rotate_cw: bool = True):
        self._rotate_cw = rotate_cw
        self._cap        = self._open_camera(camera_index)
        self._setup_mediapipe()
        self._start_capture_thread()

    # ── camera ────────────────────────────────────────────────────────
    def _open_camera(self, idx):
        if os.name == "nt":
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)
        else:
            cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap

    # ── background thread ─────────────────────────────────────────────
    def _start_capture_thread(self):
        self._raw_frame  = None
        self._frame_lock = threading.Lock()
        self._running    = True

        def _grab():
            while self._running:
                ok, frame = self._cap.read()
                if ok and frame is not None:
                    with self._frame_lock:
                        self._raw_frame = frame

        threading.Thread(target=_grab, daemon=True).start()

    def _get_latest_frame(self):
        with self._frame_lock:
            return self._raw_frame

    # ── mediapipe ─────────────────────────────────────────────────────
    def _setup_mediapipe(self):
        self._use_tasks = False
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            mpH = mp.solutions.hands
            self._hands = mpH.Hands(
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.6,
            )
            self._connections = mpH.HAND_CONNECTIONS
        else:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
            with open(model_path, "rb") as f:
                buf = f.read()
            opts = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_buffer=buf),
                num_hands=2,
                min_hand_detection_confidence=0.7,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.6,
            )
            self._detector  = vision.HandLandmarker.create_from_options(opts)
            self._use_tasks = True

    # ── public read ───────────────────────────────────────────────────
    def read(self) -> HandState:
        raw = self._get_latest_frame()
        if raw is None:
            return HandState()

        frame = raw.copy()
        if self._rotate_cw:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        lms_list, cls_list   = self._detect(rgb)
        left_lms, right_lms  = self._classify(lms_list, cls_list)

        state = HandState(frame_bgr=frame)

        if left_lms:
            lm = left_lms.landmark[0]
            state.left_wrist    = (lm.x, lm.y)
            state.left_wrist_px = (int(lm.x * fw), int(lm.y * fh))

        if right_lms:
            lm = right_lms.landmark[0]
            state.right_wrist    = (lm.x, lm.y)
            state.right_wrist_px = (int(lm.x * fw), int(lm.y * fh))
            state.right_palm_open = _is_palm_open(right_lms)

        # Fist detection
        left_fist  = _is_fist(left_lms)  if left_lms  else False
        right_fist = _is_fist(right_lms) if right_lms else False
        state.both_fists = left_fist and right_fist

        # Finger count for menu (prefer right hand)
        if right_lms:
            state.finger_count = _count_fingers(right_lms)
        elif left_lms:
            state.finger_count = _count_fingers(left_lms)

        if left_lms and right_lms:
            state.both_visible = True
            lw = state.left_wrist
            rw = state.right_wrist

            # Power = normalised distance between both wrists
            pull_norm        = _dist(lw, rw)
            clamped          = max(0.0, (pull_norm - MIN_PULL_NORM) /
                                         (MAX_PULL_NORM - MIN_PULL_NORM))
            state.draw_power = min(clamped, 1.0)

            # Angle = right wrist height
            # y_norm close to ANGLE_TOP_Y → 85°, close to ANGLE_BOT_Y → 0°
            ry               = rw[1]
            t                = 1.0 - (ry - ANGLE_TOP_Y) / max(ANGLE_BOT_Y - ANGLE_TOP_Y, 0.01)
            state.angle_deg  = max(0.0, min(85.0, t * 85.0))

        return state

    def release(self):
        self._running = False
        self._cap.release()

    # ── internal ──────────────────────────────────────────────────────
    def _detect(self, rgb):
        if not self._use_tasks:
            res = self._hands.process(rgb)
            return (res.multi_hand_landmarks or [],
                    res.multi_handedness      or [])
        else:
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            det = self._detector.detect(img)
            if not det or not det.hand_landmarks:
                return [], []

            class _Compat:
                def __init__(self, lm):
                    self.landmark = lm

            lms = [_Compat(lm) for lm in det.hand_landmarks]
            return lms, (det.handedness if det.handedness else [])

    def _classify(self, lms_list, cls_list):
        left = right = None
        if cls_list:
            try:
                for i, hnd in enumerate(cls_list):
                    cat = (hnd[0].category_name
                           if hasattr(hnd[0], "category_name")
                           else hnd.classification[0].label)
                    if cat == "Left"  and i < len(lms_list):
                        left  = lms_list[i]
                    if cat == "Right" and i < len(lms_list):
                        right = lms_list[i]
            except Exception:
                pass

        # Fallback: assign by x position
        if left is None and right is None and lms_list:
            if len(lms_list) == 1:
                left = lms_list[0]
            else:
                x0 = lms_list[0].landmark[0].x
                x1 = lms_list[1].landmark[0].x
                if x0 < x1:
                    left, right = lms_list[0], lms_list[1]
                else:
                    left, right = lms_list[1], lms_list[0]
        return left, right

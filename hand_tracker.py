"""
hand_tracker.py
---------------
Camera capture + hand detection running in a background thread so the game
stays smooth.

Two interchangeable back-ends:

  1. MediaPipeBackend  (recommended, accurate 21-point hand skeleton)
       - works with the old  mp.solutions.hands  API (mediapipe <= 0.10.14)
       - or the new          mp.tasks HandLandmarker API (mediapipe >= 0.10.15);
         the small model file is downloaded automatically on first run.

  2. SkinBackend  (fallback, pure OpenCV: skin colour + contour + convex hull)
       - used automatically if MediaPipe is not installed / not working.
       - press  C  with your hand inside the green box to calibrate.
"""

import os
import threading
import time
import urllib.request

import cv2
import numpy as np

from gestures import Observation

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 100_000:
        return MODEL_PATH
    print("[tracker] Downloading MediaPipe hand model (one time, ~8 MB) ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as exc:  # no internet, firewall, ...
        raise RuntimeError(
            f"Could not download the hand model ({exc}).\n"
            f"Download it manually from:\n  {MODEL_URL}\n"
            f"and save it next to main.py as 'hand_landmarker.task'.") from exc
    return MODEL_PATH


def _fist_from_points(pts, w, h):
    """A hand is a fist when (almost) all four fingers are folded towards the wrist."""
    p = np.array([(x * w, y * h) for x, y in pts], dtype=np.float32)
    wrist = p[0]
    extended = 0
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        if np.linalg.norm(p[tip] - wrist) > 1.08 * np.linalg.norm(p[pip] - wrist):
            extended += 1
    return extended <= 1


# ======================================================================
class MediaPipeBackend:
    name = "MediaPipe"

    def __init__(self):
        import mediapipe as mp  # noqa: imported lazily so the game runs without it
        self.mp = mp
        self.legacy = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
        self._last_ts = 0
        self._t0 = time.time()
        if self.legacy:
            self.hands = mp.solutions.hands.Hands(
                max_num_hands=1, model_complexity=0,
                min_detection_confidence=0.6, min_tracking_confidence=0.5)
        else:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            opts = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.55,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5)
            self.landmarker = vision.HandLandmarker.create_from_options(opts)

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.legacy:
            res = self.hands.process(rgb)
            if not res.multi_hand_landmarks:
                return Observation()
            pts = [(p.x, p.y) for p in res.multi_hand_landmarks[0].landmark]
        else:
            ts = int((time.time() - self._t0) * 1000)
            ts = max(ts, self._last_ts + 1)          # must strictly increase
            self._last_ts = ts
            img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            res = self.landmarker.detect_for_video(img, ts)
            if not res.hand_landmarks:
                return Observation()
            pts = [(p.x, p.y) for p in res.hand_landmarks[0]]

        palm = [pts[i] for i in (0, 5, 9, 13, 17)]
        cx = sum(p[0] for p in palm) / 5.0
        cy = sum(p[1] for p in palm) / 5.0
        return Observation(True, cx, cy, _fist_from_points(pts, w, h),
                           points=pts, connections=HAND_CONNECTIONS)

    def calibrate(self, bgr):  # nothing to calibrate
        pass


# ======================================================================
class SkinBackend:
    name = "OpenCV skin (fallback)"

    def __init__(self):
        self.lo = np.array([0, 135, 85], np.uint8)     # YCrCb skin range (default)
        self.hi = np.array([255, 175, 130], np.uint8)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.face = cascade if not cascade.empty() else None
        self.faces = []
        self.i = 0
        self.fist_state = False

    @staticmethod
    def calib_box(w, h):
        return int(w * 0.42), int(h * 0.40), int(w * 0.58), int(h * 0.62)

    def calibrate(self, bgr):
        h, w = bgr.shape[:2]
        x0, y0, x1, y1 = self.calib_box(w, h)
        roi = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
        med = np.median(roi, axis=0)
        tol = np.array([80, 20, 20])
        self.lo = np.clip(med - tol, 0, 255).astype(np.uint8)
        self.hi = np.clip(med + tol, 0, 255).astype(np.uint8)
        print("[tracker] skin colour calibrated:", self.lo, self.hi)

    def detect(self, bgr):
        small = cv2.resize(bgr, (320, 240))
        ycc = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycc, self.lo, self.hi)

        # remove the user's face (it is also skin coloured!)
        self.i += 1
        if self.face is not None and self.i % 6 == 1:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            self.faces = self.face.detectMultiScale(gray, 1.2, 5, minSize=(40, 40))
        for (x, y, fw, fh) in self.faces:
            cv2.rectangle(mask, (x - int(.3 * fw), y - int(.4 * fh)),
                          (x + int(1.3 * fw), y + int(2.0 * fh)), 0, -1)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return Observation()
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < 1800:
            return Observation()

        hull = cv2.convexHull(c)
        hull_area = max(cv2.contourArea(hull), 1.0)
        solidity = area / hull_area
        # hysteresis so the fist does not flicker
        if self.fist_state:
            self.fist_state = solidity > 0.80
        else:
            self.fist_state = solidity > 0.88

        m = cv2.moments(c)
        if m["m00"] == 0:
            return Observation()
        cx, cy = m["m10"] / m["m00"] / 320.0, m["m01"] / m["m00"] / 240.0
        outline = [(float(p[0][0]) / 320.0, float(p[0][1]) / 240.0) for p in hull]
        return Observation(True, cx, cy, self.fist_state, outline=outline)


# ======================================================================
def make_backend(choice="auto"):
    if choice in ("auto", "mediapipe"):
        try:
            be = MediaPipeBackend()
            print(f"[tracker] Using MediaPipe ({'legacy solutions' if be.legacy else 'tasks'} API)")
            return be
        except Exception as exc:
            print(f"[tracker] MediaPipe not available: {exc}")
            if choice == "mediapipe":
                raise
    print("[tracker] Using the OpenCV skin-colour fallback (press C to calibrate).")
    return SkinBackend()


class HandController:
    """Grabs frames and runs the hand detector on a background thread."""

    def __init__(self, cam_index=0, backend="auto", size=(480, 360)):
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera #{cam_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.size = size
        self.backend = make_backend(backend)
        self.lock = threading.Lock()
        self.obs = Observation()
        self.frame = None
        self.running = False
        self._calib = False
        self.fps = 0.0

    @property
    def backend_name(self):
        return self.backend.name

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def calibrate(self):
        self._calib = True

    def _loop(self):
        n, t0 = 0, time.time()
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            frame = cv2.flip(frame, 1)                      # mirror: left is left
            frame = cv2.resize(frame, self.size)
            if self._calib:
                self.backend.calibrate(frame)
                self._calib = False
            try:
                obs = self.backend.detect(frame)
            except Exception as exc:                        # never kill the thread
                print("[tracker] detect error:", exc)
                obs = Observation()
            with self.lock:
                self.obs, self.frame = obs, frame
            n += 1
            if time.time() - t0 >= 1.0:
                self.fps, n, t0 = n / (time.time() - t0), 0, time.time()

    def latest(self):
        with self.lock:
            return self.obs, (None if self.frame is None else self.frame.copy())

    def stop(self):
        self.running = False
        time.sleep(0.05)
        self.cap.release()

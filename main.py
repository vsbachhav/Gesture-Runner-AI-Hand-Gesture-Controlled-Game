"""
main.py  -  GESTURE RUNNER  (Subway-Surfers style game controlled by your hand)

Run:
    python main.py                 # webcam + hand gestures
    python main.py --keyboard      # no camera: W A S D + SPACE
    python main.py --camera 1      # use the 2nd camera
    python main.py --backend skin  # force the pure-OpenCV tracker
    python main.py --fullscreen

Gestures (mirror view - move your hand like you see it on screen):
    hand OUT of the centre box  LEFT / RIGHT   -> change lane
    hand OUT of the centre box  UP             -> jump
    hand OUT of the centre box  DOWN           -> slide
    HOLD A FIST (~0.5 s)                       -> start / restart, and
                                                  release the ENERGY BOOST when the bucket is full
"""

import argparse
import sys
import time

import cv2

from game import Game
from gestures import GestureEngine, Observation
from renderer import Renderer, W, H

WINDOW = "GESTURE RUNNER"

# arrow-key codes returned by cv2.waitKeyEx on Windows / Linux / macOS
KEYMAP = {
    2424832: "left", 2490368: "up", 2555904: "right", 2621440: "down",     # Windows
    65361: "left", 65362: "up", 65363: "right", 65364: "down",             # Linux (GTK/Qt)
    81: "left", 82: "up", 83: "right", 84: "down",                         # Linux (some builds)
    63234: "left", 63232: "up", 63235: "right", 63233: "down",             # macOS
}
LETTERS = {ord("a"): "left", ord("d"): "right", ord("w"): "up", ord("s"): "down"}


def parse():
    ap = argparse.ArgumentParser(description="Gesture Runner - hand controlled subway runner")
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--backend", choices=("auto", "mediapipe", "skin"), default="auto",
                    help="hand tracker: auto (default) / mediapipe / skin (pure OpenCV)")
    ap.add_argument("--keyboard", action="store_true", help="play with the keyboard, no camera")
    ap.add_argument("--fullscreen", action="store_true")
    return ap.parse_args()


def main():
    args = parse()
    controller = None
    if not args.keyboard:
        try:
            from hand_tracker import HandController
            controller = HandController(args.camera, args.backend).start()
            print(f"[main] hand tracker: {controller.backend_name}")
        except Exception as exc:
            print(f"[main] Camera / tracker unavailable ({exc}).\n"
                  f"[main] Falling back to KEYBOARD mode (W A S D + SPACE).")
            controller = None

    game, engine, renderer = Game(), GestureEngine(), Renderer()
    backend_name = controller.backend_name if controller else "keyboard mode"

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    if args.fullscreen:
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    last = time.time()
    fps, frames, fps_t = 0.0, 0, time.time()
    empty = Observation()

    while True:
        now = time.time()
        dt = min(now - last, 0.05)
        last = now

        obs, frame = controller.latest() if controller else (empty, None)
        charge = game.state != "playing" or game.boost_ready
        events = engine.update(obs, now, charge)

        # ---- keyboard (always works, also together with the hand) ------------
        key = cv2.waitKeyEx(1)
        if key != -1:
            k = key & 0xFF
            if key in KEYMAP:
                events.append(KEYMAP[key])
            elif k in LETTERS or (k != 0 and chr(k).lower() in "wasd"):
                events.append({"a": "left", "d": "right", "w": "up", "s": "down"}[chr(k).lower()])
            elif k == 32:                                   # SPACE
                events.append("boost")
            elif k in (ord("b"), ord("B")):
                events.append("boost")
            elif k in (ord("r"), ord("R")):
                engine.recenter()
            elif k in (ord("c"), ord("C")) and controller:
                controller.calibrate()
            elif k in (27, ord("q"), ord("Q")):
                break

        # ---- route events ----------------------------------------------------
        game_events = []
        for e in events:
            if e == "boost" and game.state != "playing":
                if game.start():
                    engine.recenter()
            else:
                game_events.append(e)
        game.update(dt, game_events)

        # ---- draw ------------------------------------------------------------
        frames += 1
        if now - fps_t >= 0.5:
            fps, frames, fps_t = frames / (now - fps_t), 0, now
        canvas = renderer.render(game, engine, obs, frame, dt, backend_name, fps)
        cv2.imshow(WINDOW, canvas)

        try:
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    if controller:
        controller.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)

# GESTURE RUNNER – Subway-Surfers-style game controlled by your hand (Python + OpenCV)

Run forward on the subway tracks, dodge trains and barriers, collect coins, fill the
**coin bucket**, then make a **fist** to release a huge **energy boost**
(super speed, coin magnet, smash through everything, +1 life).

## 1. Files

| File              | What it does                                                              |
|-------------------|---------------------------------------------------------------------------|
| `main.py`         | Start here. Game loop, keyboard fallback, command-line options            |
| `game.py`         | Game rules: lanes, jump/slide, obstacles, coins, bucket, boost, lives     |
| `gestures.py`     | Turns hand position / fist into LEFT, RIGHT, UP, DOWN, BOOST events        |
| `hand_tracker.py` | Webcam thread + hand detection (MediaPipe, or pure-OpenCV fallback)       |
| `renderer.py`     | All graphics: 3-D world, boy, HUD, camera panel, gesture pad, coin bucket |
| `requirements.txt`| Python packages                                                            |

## 2. Install (once)

Use **Python 3.9 – 3.12** (MediaPipe does not support newer versions yet).

```bash
# 1) unzip the folder, open a terminal inside it
cd subway_gesture_runner

# 2) (recommended) virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
source venv/bin/activate         # macOS / Linux

# 3) install packages
pip install -r requirements.txt
```

## 3. Run

```bash
python main.py
```

* The **first run downloads a small hand model (~8 MB)** automatically – internet needed once.
  If your network blocks it, download
  `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`
  and save it in this folder as `hand_landmarker.task`.
* Allow camera access if your OS asks (macOS: Terminal / VS Code → Camera).

Other options:

```bash
python main.py --keyboard        # no camera: W A S D (or arrows) + SPACE
python main.py --camera 1        # use a different camera
python main.py --backend skin    # pure OpenCV tracker (no MediaPipe needed, press C to calibrate)
python main.py --fullscreen
```

## 4. How to play (hand gestures)

Sit about an arm's length from the camera with good light. The camera preview on the
right shows a small **green box** in the centre – that is your "joystick" neutral zone.

| Gesture                                   | Action                           |
|-------------------------------------------|----------------------------------|
| Move hand **left / right** out of the box | Change lane (hold = keep moving) |
| Move hand **up** out of the box           | Jump                             |
| Move hand **down** out of the box         | Slide (in the air: fast-drop)    |
| Bring hand **back to the box**            | Re-arm the joystick              |
| **Hold a fist** ~0.5 s                    | Start / restart the game         |
| **Full bucket + hold a fist**             | ENERGY BOOST                     |

Obstacles: **striped barrier = jump**, **yellow/black bar overhead = slide**,
**train = change lane**.

Coins: every coin flies into the bucket (20 coins fill it). When it is full the boy glows
gold and the panel says **MAKE A FIST**. Boost lasts 6+ seconds (longer each time), the
bucket drains while boosting, you run ~2x faster, magnetically pull coins (worth double),
smash through obstacles for bonus points and win a life back.

Keyboard (always active): `W A S D` / arrows = move, `SPACE` = start / boost,
`R` = re-centre the neutral box on your hand, `C` = calibrate skin colour (fallback tracker),
`ESC` / `Q` = quit.

## 5. Tips / troubleshooting

* Hand not detected → more light, plain background, whole hand visible.
* Steering feels too sensitive/slow → edit `dead_x`, `dead_y` (box size) and `repeat_delay`
  in `gestures.py`.
* Wrong camera → `--camera 1`. Black window → close other apps using the webcam.
* Low FPS → close other programs; the game itself draws at ~60–80 FPS.
* No MediaPipe (e.g. unsupported Python) → the game automatically uses the OpenCV
  skin-colour tracker; hold your hand inside the green box and press **C** to calibrate.

"""
gestures.py
-----------
Turns raw hand observations (position + open/fist) into discrete game events:

    'left' | 'right' | 'up' | 'down' | 'boost'

How it works (the "joystick flick"):
  * A small NEUTRAL ZONE is drawn in the middle of the camera preview.
  * Move your hand OUT of the zone  -> one event fires (left/right/up/down).
  * Bring it BACK to the zone       -> the joystick re-arms.
  * Holding the hand out to the left/right keeps repeating lane changes.
  * Closing your hand into a FIST for ~0.45 s fires 'boost'
    (only when the coin bucket is full, or to start / restart the game).
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Observation:
    """What the tracker saw in one camera frame (coordinates are 0..1, mirrored)."""
    present: bool = False
    cx: float = 0.5                 # palm centre x
    cy: float = 0.5                 # palm centre y
    fist: bool = False              # True when the hand is closed
    points: Optional[list] = None   # landmark points (x, y) for drawing
    connections: Optional[list] = None  # index pairs to draw as a skeleton
    outline: Optional[list] = None  # closed outline (fallback tracker)
    extra: dict = field(default_factory=dict)


class GestureEngine:
    def __init__(self, dead_x: float = 0.13, dead_y: float = 0.15,
                 repeat_delay: float = 0.55, fist_hold: float = 0.45,
                 smoothing: float = 0.55):
        self.dead_x = dead_x
        self.dead_y = dead_y
        self.repeat_delay = repeat_delay
        self.fist_hold = fist_hold
        self.smoothing = smoothing

        self.neutral = [0.5, 0.5]
        self.sx = 0.5                # smoothed hand position
        self.sy = 0.5
        self.have_hand = False
        self.armed = False
        self.out_since: Optional[float] = None
        self.direction: Optional[str] = None   # direction currently "held"
        self.fist_since: Optional[float] = None
        self.fist_progress = 0.0     # 0..1 charge ring
        self.fist_fired = False
        self.flash = {}              # direction -> time it last fired (for UI)

    # ------------------------------------------------------------------
    def recenter(self):
        """Use the current hand position as the new neutral point."""
        if self.have_hand:
            self.neutral = [self.sx, self.sy]

    @property
    def offset(self):
        return (self.sx - self.neutral[0], self.sy - self.neutral[1])

    # ------------------------------------------------------------------
    def update(self, obs: Observation, now: float, charge_enabled: bool) -> List[str]:
        events: List[str] = []

        if not obs.present:
            self.have_hand = False
            self.armed = False
            self.direction = None
            self.out_since = None
            self.fist_since = None
            self.fist_progress = 0.0
            self.fist_fired = False
            return events

        if not self.have_hand:            # hand just appeared: snap, don't glide
            self.sx, self.sy = obs.cx, obs.cy
            self.have_hand = True
            self.armed = False            # must visit the neutral zone first
        else:
            a = self.smoothing
            self.sx = a * obs.cx + (1 - a) * self.sx
            self.sy = a * obs.cy + (1 - a) * self.sy

        # ---- fist = charge / boost / start ---------------------------------
        if obs.fist and charge_enabled:
            self.direction = None
            self.armed = False
            if self.fist_since is None:
                self.fist_since = now
            self.fist_progress = min(1.0, (now - self.fist_since) / self.fist_hold)
            if self.fist_progress >= 1.0 and not self.fist_fired:
                self.fist_fired = True
                events.append("boost")
                self.flash["boost"] = now
            return events
        self.fist_since = None
        self.fist_progress = 0.0
        self.fist_fired = False

        # ---- direction flicks ----------------------------------------------
        dx, dy = self.offset
        inside = abs(dx) < self.dead_x and abs(dy) < self.dead_y
        if inside:
            self.armed = True
            self.direction = None
            self.out_since = None
            return events

        if abs(dx) / self.dead_x >= abs(dy) / self.dead_y:
            d = "right" if dx > 0 else "left"
        else:
            d = "down" if dy > 0 else "up"
        self.direction = d

        if self.armed:
            self.armed = False
            self.out_since = now
            events.append(d)
            self.flash[d] = now
        elif d in ("left", "right") and self.out_since is not None \
                and now - self.out_since > self.repeat_delay:
            self.out_since = now
            events.append(d)
            self.flash[d] = now
        return events

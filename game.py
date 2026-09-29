"""
game.py
-------
Pure game logic (no drawing). World layout:

    x  : lateral position, lanes are -1, 0, +1
    z  : distance ahead of the runner (0 = the runner, grows towards horizon)
    h  : height above the ground

The renderer turns (x, h, z) into screen pixels.
"""

import random
from dataclasses import dataclass

LANES = (-1, 0, 1)
BASE_SPEED = 26.0        # world units / second
SPAWN_Z = 170.0          # how far ahead things appear
BUCKET_CAP = 20          # coins needed to fill the bucket
JUMP_V = 9.6
GRAVITY = 27.0
SLIDE_TIME = 0.75
MAX_LIVES = 3


@dataclass
class Entity:
    kind: str               # 'coin' | 'barrier' | 'overhead' | 'train' | 'tower'
    x: float                # lateral centre
    wz: float               # absolute position along the track
    h: float = 0.0          # height (coins)
    length: float = 1.0     # depth along z
    width: float = 0.9
    height: float = 1.0
    color: int = 0
    phase: float = 0.0
    alive: bool = True
    hit: bool = False


class Game:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.best = 0
        self.fx = []                    # events for the renderer (particles, popups...)
        self.reset()
        self.state = "ready"            # ready | playing | over

    # ------------------------------------------------------------------
    def reset(self):
        self.state = "playing"
        self.t = 0.0
        self.distance = 0.0             # world scroll (never resets while running)
        self.run_dist = 0.0
        self.speed = 0.0
        self.crash_slow = 1.0

        self.lane = 0
        self.px = 0.0
        self.jump_y = 0.0
        self.vy = 0.0
        self.on_ground = True
        self.queue_slide = False
        self.slide_t = 0.0
        self.run_phase = 0.0

        self.lives = MAX_LIVES
        self.invuln = 0.0
        self.coins = 0
        self.smashes = 0
        self.bucket = 0.0
        self.boost_t = 0.0
        self.boost_total = 1.0
        self.boost_amt = 0.0            # 0..1 smooth blend for visuals
        self.boost_power = 0
        self.boosts_used = 0

        self.entities = []
        self.next_tower_wz = 10.0
        self.tower_side = 1
        self.next_row_wz = 70.0

        self.shake = 0.0
        self.flash = 0.0
        self.over_t = 0.0
        self.action = ("", -10.0)       # (text, time) last gesture, for the HUD
        self.toast_text, self.toast_kind, self.toast_t = "", "info", 0.0
        self.fx.clear()

    # ------------------------------------------------------------------ helpers
    @property
    def meters(self):
        return int(self.run_dist * 0.5)

    @property
    def score(self):
        return self.meters + self.coins * 10 + self.smashes * 50

    @property
    def boosting(self):
        return self.boost_t > 0

    @property
    def boost_ready(self):
        return self.state == "playing" and self.bucket >= BUCKET_CAP and not self.boosting

    @property
    def energy(self):
        """0..1 energy of the boy (bucket fill, or remaining boost time)."""
        if self.boosting:
            return self.boost_t / self.boost_total
        return min(1.0, self.bucket / BUCKET_CAP)

    def _say(self, text):
        self.action = (text, self.t)

    def toast(self, text, kind="info", seconds=2.2):
        self.toast_text, self.toast_kind, self.toast_t = text, kind, seconds

    def _fx(self, *item):
        if len(self.fx) < 300:
            self.fx.append(item)

    # ------------------------------------------------------------------ control
    def start(self):
        if self.state == "over" and self.over_t < 0.8:
            return False
        if self.state == "playing":
            return False
        best = self.best
        self.reset()
        self.best = best
        self.state = "playing"
        self.toast("GO!  STEER WITH YOUR HAND", "info", 1.6)
        return True

    def try_boost(self):
        if not self.boost_ready:
            return False
        self.boost_power = min(4, self.boosts_used)
        self.boost_total = 6.0 + 1.5 * self.boost_power
        self.boost_t = self.boost_total
        self.boosts_used += 1
        if self.lives < MAX_LIVES:
            self.lives += 1
            self._fx("life")
        self.shake = 0.7
        self._fx("boost")
        self.toast(f"ENERGY BOOST x{self.boosts_used}!", "boost", 2.4)
        return True

    def _handle_events(self, events):
        for e in events:
            if e == "left":
                if self.lane > -1:
                    self.lane -= 1
                self._say("LEFT")
            elif e == "right":
                if self.lane < 1:
                    self.lane += 1
                self._say("RIGHT")
            elif e == "up":
                if self.on_ground:
                    self.vy = JUMP_V
                    self.on_ground = False
                    self.slide_t = 0.0
                    self._fx("jump", self.px)
                self._say("JUMP")
            elif e == "down":
                if self.on_ground:
                    self.slide_t = SLIDE_TIME
                else:
                    self.vy = min(self.vy, -16.0)
                    self.queue_slide = True
                self._say("SLIDE")
            elif e == "boost":
                self.try_boost()

    # ------------------------------------------------------------------ update
    def _target_speed(self):
        if self.state == "ready":
            return 12.0
        if self.state == "over":
            return 0.0
        base = BASE_SPEED + min(30.0, self.run_dist * 0.005)
        mult = 1.0 + self.boost_amt * (0.9 + 0.22 * self.boost_power)
        return base * mult * self.crash_slow

    def update(self, dt, events=()):
        self.t += dt
        self.toast_t = max(0.0, self.toast_t - dt)
        self.shake = max(0.0, self.shake - dt * 2.0)
        self.flash = max(0.0, self.flash - dt * 2.5)

        if self.state == "playing":
            self._handle_events(events)
            self.invuln = max(0.0, self.invuln - dt)
            self.crash_slow = min(1.0, self.crash_slow + dt * 0.8)
        elif self.state == "over":
            self.over_t += dt

        # boost bookkeeping
        if self.boost_t > 0 and self.state == "playing":
            self.boost_t -= dt
            self.bucket = BUCKET_CAP * max(0.0, self.boost_t) / self.boost_total
            if self.boost_t <= 0:
                self.boost_t = 0.0
                self.bucket = 0.0
                self.invuln = 1.2
                self._fx("boost_end")
                self.toast("BOOST OVER - FILL THE BUCKET AGAIN!", "info", 2.0)
        target_amt = 1.0 if self.boost_t > 0 else 0.0
        self.boost_amt += (target_amt - self.boost_amt) * min(1.0, dt * 3.0)

        # speed + scrolling
        self.speed += (self._target_speed() - self.speed) * min(1.0, dt * 4.0)
        dz = self.speed * dt
        self.distance += dz
        if self.state == "playing":
            self.run_dist += dz

        # runner physics
        self.px += (self.lane - self.px) * min(1.0, dt * 13.0)
        if not self.on_ground:
            self.vy -= GRAVITY * dt
            self.jump_y += self.vy * dt
            if self.jump_y <= 0.0:
                self.jump_y, self.vy, self.on_ground = 0.0, 0.0, True
                self._fx("land", self.px)
                if self.queue_slide:
                    self.slide_t, self.queue_slide = SLIDE_TIME, False
        if self.slide_t > 0:
            self.slide_t -= dt
        self.run_phase += dt * (7.0 + self.speed * 0.13)

        self._spawn()
        self._update_entities(dt)

    # ------------------------------------------------------------------ spawning
    def _spawn(self):
        limit = self.distance + SPAWN_Z
        while self.next_tower_wz < limit:
            self._spawn_tower()
        if self.state == "playing":
            while self.next_row_wz < limit:
                self._spawn_row()

    def _spawn_tower(self):
        r = self.rng
        side = self.tower_side
        self.tower_side = -side
        w = r.uniform(1.6, 3.2)
        e = Entity("tower", side * (2.5 + w / 2 + r.uniform(0.0, 0.6)), self.next_tower_wz,
                   length=r.uniform(4.0, 6.5), width=w, height=r.uniform(3.5, 10.0),
                   color=r.randrange(4), phase=r.random() * 6.28)
        self.entities.append(e)
        self.next_tower_wz += r.uniform(4.0, 8.0)

    def _coin(self, lane, wz, h):
        self.entities.append(Entity("coin", lane, wz, h=h, phase=self.rng.random() * 6.28))

    def _obstacle(self, kind, lane, wz):
        r = self.rng
        if kind == "train":
            e = Entity("train", lane, wz, length=r.uniform(10.0, 22.0), height=2.4,
                       width=0.94, color=r.randrange(3))
        elif kind == "barrier":
            e = Entity("barrier", lane, wz, length=1.0, height=0.95, width=0.94)
        else:
            e = Entity("overhead", lane, wz, length=1.0, height=2.0, width=0.98)
        self.entities.append(e)
        return e

    def _spawn_row(self):
        r = self.rng
        wz = self.next_row_wz
        diff = min(1.0, self.run_dist / 2500.0)
        roll = r.random()
        length = 0.0

        if roll < 0.30:                                   # coin trail
            lane = r.choice(LANES)
            n = r.randint(6, 12)
            for i in range(n):
                self._coin(lane, wz + i * 2.6, 0.8)
            length = n * 2.6
        elif roll < 0.45:                                 # coin arc over a barrier
            lane = r.choice(LANES)
            self._obstacle("barrier", lane, wz + 7.0)
            for i, h in enumerate((0.8, 1.6, 2.2, 2.45, 2.2, 1.6, 0.8)):
                self._coin(lane, wz + 2.0 + i * 1.6, h)
            length = 13.0
        else:                                             # obstacle row
            if r.random() < 0.10 + 0.12 * diff:           # full-width jump/slide wall
                kind = r.choice(("barrier", "overhead"))
                for ln in LANES:
                    self._obstacle(kind, ln, wz)
                length = 1.0
            else:
                lanes = r.sample(LANES, 1 if r.random() < 0.5 else 2)
                for ln in lanes:
                    kind = r.choices(("barrier", "overhead", "train"),
                                     weights=(4, 3, 3 if diff < 0.3 else 4))[0]
                    e = self._obstacle(kind, ln, wz)
                    length = max(length, e.length)
                free = [ln for ln in LANES if ln not in lanes]
                if r.random() < 0.6:
                    ln = r.choice(free)
                    for i in range(5):
                        self._coin(ln, wz + i * 2.4, 0.8)
        gap = 14.0 + self.speed * 0.55 + r.uniform(0.0, 10.0)
        self.next_row_wz = wz + length + gap

    # ------------------------------------------------------------------ collisions
    def _update_entities(self, dt):
        body = self.jump_y + (0.4 if self.slide_t > 0 else 0.85)   # body-centre height
        playing = self.state == "playing"
        boosting = self.boosting

        for e in self.entities:
            z = e.wz - self.distance
            if z < -7.0:
                e.alive = False
                continue
            if not playing or e.kind == "tower" or not e.alive:
                continue

            if e.kind == "coin":
                if boosting and -1.0 < z < 16.0:                    # coin magnet
                    k = min(1.0, dt * 9.0)
                    e.x += (self.px - e.x) * k
                    e.h += (body - e.h) * k
                if -1.8 < z < 1.0 and abs(e.x - self.px) < 0.6 \
                        and abs(e.h - body) < (1.8 if boosting else 1.05):
                    self._collect(e)
                continue

            if e.hit or abs(e.x - self.px) > 0.62:
                continue
            if z > 0.6 or z + e.length < -0.6:
                continue

            if e.kind == "barrier":
                hit = self.jump_y < 0.85
            elif e.kind == "overhead":
                hit = not (self.slide_t > 0 and self.on_ground)
            else:
                hit = True
            if not hit:
                continue

            if boosting:
                self._smash(e)
            elif self.invuln <= 0:
                self._crash(e)

        self.entities = [e for e in self.entities if e.alive]

    def _collect(self, e):
        e.alive = False
        val = 2 if self.boosting else 1
        self.coins += val
        if not self.boosting and self.bucket < BUCKET_CAP:
            self.bucket = min(BUCKET_CAP, self.bucket + 1)
            if self.bucket >= BUCKET_CAP:
                self.toast("BUCKET FULL!  MAKE A FIST TO BOOST", "ready", 3.0)
                self._fx("bucket_full")
        self._fx("coin", e.x, e.h, val)

    def _smash(self, e):
        e.alive = False
        e.hit = True
        self.smashes += 1
        self.shake = max(self.shake, 0.35)
        self._fx("smash", e.x, e.kind)

    def _crash(self, e):
        e.alive = False
        e.hit = True
        self.lives -= 1
        self.shake = 1.0
        self.flash = 1.0
        self.crash_slow = 0.35
        self.invuln = 1.8
        self._fx("crash", self.px, e.kind)
        if self.lives <= 0:
            self.state = "over"
            self.over_t = 0.0
            self.best = max(self.best, self.score)
            self.toast("", "info", 0.0)

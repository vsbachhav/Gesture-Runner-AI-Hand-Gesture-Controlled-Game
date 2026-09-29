"""
renderer.py
-----------
Everything you SEE. Pure OpenCV drawing (no images / sounds needed).

Window layout (1120 x 640):

    +--------------------------------+-----------------+
    |                                |  CAMERA + hand  |
    |      3-D SUBWAY RUNNER         |  GESTURE PAD    |
    |      (pseudo perspective)      |  + instructions |
    |                                |  COIN BUCKET    |
    |                                |  ENERGY BAR     |
    +--------------------------------+-----------------+
"""

import math
import random

import cv2
import numpy as np

from game import BUCKET_CAP, MAX_LIVES

GW, GH, PW = 800, 640, 320          # game view, panel width
W, H = GW + PW, GH
HY = 180                            # horizon line (y)
K1 = 1140.0                         # perspective strength
CAMH = 1.9                          # camera height
CAM_Z = 6.0                         # camera is 6 units behind the runner
FAR = 175.0
NEAR_Z = -4.8

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT2 = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA


def bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


CYAN, MAG, GOLD = bgr("#22e6ff"), bgr("#ff3df0"), bgr("#ffc928")
ORANGE, WHITE, RED = bgr("#ff8a1e"), (255, 255, 255), bgr("#ff3b4d")
GREEN, PURPLE, DARK = bgr("#39ff88"), bgr("#7a3cff"), bgr("#0b0a1a")
FOG_N, FOG_B = bgr("#3b1d63"), bgr("#1a4aa0")

TOWER_PAL = [  # front, side, top, window
    (bgr("#2b2454"), bgr("#3a2f73"), bgr("#4b3f8f"), bgr("#ffd25a")),
    (bgr("#1f2f5e"), bgr("#2b4485"), bgr("#3b58a8"), bgr("#5af0ff")),
    (bgr("#4a1f4f"), bgr("#6b2e73"), bgr("#8a3f92"), bgr("#ff7ad9")),
    (bgr("#233b3f"), bgr("#2f5559"), bgr("#3f7378"), bgr("#7dffb0")),
]
TRAIN_PAL = [  # front, side, accent
    (bgr("#e0424f"), bgr("#a82f4a"), bgr("#ffd76a")),
    (bgr("#1fb5a8"), bgr("#148a91"), bgr("#ff5aa5")),
    (bgr("#f2b632"), bgr("#c98a1c"), bgr("#2b2454")),
]


# ---------------------------------------------------------------- small helpers
def lerp_col(a, b, f):
    return (int(a[0] + (b[0] - a[0]) * f), int(a[1] + (b[1] - a[1]) * f),
            int(a[2] + (b[2] - a[2]) * f))


def shade(col, k):
    return tuple(int(min(255, max(0, c * k))) for c in col)


def poly(img, pts, col):
    cv2.fillConvexPoly(img, np.asarray(pts, np.int32), col, AA)


def text(img, s, org, scale=0.6, col=WHITE, th=1, font=FONT, anchor="l", outline=True):
    (tw, _), _ = cv2.getTextSize(s, font, scale, th)
    x, y = int(org[0]), int(org[1])
    if anchor == "c":
        x -= tw // 2
    elif anchor == "r":
        x -= tw
    if outline:
        cv2.putText(img, s, (x, y), font, scale, (12, 8, 24), th + 3, AA)
    cv2.putText(img, s, (x, y), font, scale, col, th, AA)


def alpha_rect(img, x0, y0, x1, y1, col, a):
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), int(x1), int(y1)
    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return
    img[y0:y1, x0:x1] = cv2.addWeighted(np.full(roi.shape, col, np.uint8), a, roi, 1 - a, 0)


_GLOW_CACHE = {}


def _glow_map(r):
    r = max(8, int(math.ceil(r / 8.0) * 8))
    m = _GLOW_CACHE.get(r)
    if m is None:
        yy, xx = np.mgrid[-r:r, -r:r].astype(np.float32)
        d = np.sqrt(xx * xx + yy * yy) / r
        m = (np.clip(1.0 - d, 0, 1) ** 2 * 255).astype(np.uint8)
        _GLOW_CACHE[r] = m
    return m


def glow(img, cx, cy, r, col, a=0.22):
    """soft radial glow (additive, smooth falloff). `a` ~ strength 0..1"""
    m = _glow_map(r)
    rr = m.shape[0] // 2
    cx, cy = int(cx), int(cy)
    x0, y0, x1, y1 = cx - rr, cy - rr, cx + rr, cy + rr
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return
    mm = m[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
    k = min(1.0, a * 3.2)
    add = cv2.merge([cv2.convertScaleAbs(mm, alpha=(c / 255.0) * k) for c in col])
    img[y0:y1, x0:x1] = cv2.add(img[y0:y1, x0:x1], add)


def tri(img, cx, cy, sz, d, col, filled=True):
    pts = {"up": [(cx, cy - sz), (cx - sz, cy + sz * .7), (cx + sz, cy + sz * .7)],
           "down": [(cx, cy + sz), (cx - sz, cy - sz * .7), (cx + sz, cy - sz * .7)],
           "left": [(cx - sz, cy), (cx + sz * .7, cy - sz), (cx + sz * .7, cy + sz)],
           "right": [(cx + sz, cy), (cx - sz * .7, cy - sz), (cx - sz * .7, cy + sz)]}[d]
    p = np.asarray(pts, np.int32)
    if filled:
        cv2.fillConvexPoly(img, p, col, AA)
    else:
        cv2.polylines(img, [p], True, col, 2, AA)


def heart(img, cx, cy, r, col):
    cv2.circle(img, (int(cx - r * .55), int(cy - r * .3)), int(r * .68), col, -1, AA)
    cv2.circle(img, (int(cx + r * .55), int(cy - r * .3)), int(r * .68), col, -1, AA)
    cv2.fillConvexPoly(img, np.asarray([(cx, cy + r * 1.05), (cx - r * 1.18, cy - r * .1),
                                        (cx + r * 1.18, cy - r * .1)], np.int32), col, AA)


def bolt(img, cx, cy, s, col):
    pts = [(0.15, -1), (-0.55, 0.15), (-0.05, 0.15), (-0.2, 1), (0.6, -0.2), (0.05, -0.2)]
    cv2.fillConvexPoly(img, np.asarray([(cx + x * s, cy + y * s) for x, y in pts], np.int32), col, AA)


def coin_icon(img, cx, cy, r, phase=0.0):
    rx = max(1, int(r * (0.35 + 0.65 * abs(math.cos(phase)))))
    cv2.ellipse(img, (int(cx), int(cy)), (rx, int(r)), 0, 0, 360, bgr("#b8790a"), -1, AA)
    cv2.ellipse(img, (int(cx), int(cy)), (max(1, rx - 2), max(1, int(r) - 2)), 0, 0, 360, GOLD, -1, AA)
    if r > 7:
        cv2.ellipse(img, (int(cx), int(cy)), (max(1, int(rx * .55)), max(1, int(r * .55))), 0, 0, 360,
                    bgr("#ffe98a"), 1, AA)


class Part:
    __slots__ = ("x", "y", "vx", "vy", "life", "max", "size", "col", "grav", "kind", "txt")

    def __init__(self, x, y, vx, vy, life, size, col, grav=0.0, kind="dot", txt=""):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.life = self.max = life
        self.size, self.col, self.grav, self.kind, self.txt = size, col, grav, kind, txt


# ======================================================================
class Renderer:
    def __init__(self):
        self.t = 0.0
        self.cam_x = 0.0
        self.sx = self.sy = 0.0
        self.fogc = FOG_N
        self.crouch = 0.0
        self.parts = []
        self.cparts = []           # particles in full-window coordinates
        self.flyers = []
        self.bump = 0.0
        self.disp = 0.0            # bucket coins shown
        self.white = 0.0
        self.rng = random.Random(3)
        self.img = np.zeros((GH, GW, 3), np.uint8)
        self.panel = np.zeros((H, PW, 3), np.uint8)
        self.bg_n = self._build_bg(False)
        self.bg_b = self._build_bg(True)
        self.panel_bg = self._build_panel_bg()
        self.vig_boost = self._vignette((255, 190, 30))
        self.vig_red = self._vignette((30, 30, 255))

    # ------------------------------------------------------------ static art
    def _build_bg(self, boost):
        rng = random.Random(11)
        if boost:
            top, mid, hor = bgr("#040a2e"), bgr("#1a38b0"), bgr("#4fe8ff")
            g0, g1, sun, bl = bgr("#13306e"), bgr("#050a22"), bgr("#eaffff"), bgr("#0a1a55")
        else:
            top, mid, hor = bgr("#10082e"), bgr("#5a1d78"), bgr("#ff7a5c")
            g0, g1, sun, bl = bgr("#3b1d63"), bgr("#0d0820"), bgr("#ffd35a"), bgr("#26104a")

        def lerp(a, b, u):
            return np.array(a, np.float32) * (1 - u) + np.array(b, np.float32) * u

        t = np.linspace(0, 1, HY, dtype=np.float32)[:, None]
        sky = np.where(t < 0.6, lerp(top, mid, np.clip(t / 0.6, 0, 1)),
                       lerp(mid, hor, np.clip((t - 0.6) / 0.4, 0, 1)))
        tg = (np.linspace(0, 1, GH - HY, dtype=np.float32)[:, None]) ** 0.55
        gnd = lerp(g0, g1, tg)
        img = np.zeros((GH, GW, 3), np.uint8)
        img[:HY] = sky[:, None, :].astype(np.uint8)
        ground = gnd[:, None, :].astype(np.uint8)

        # sun with synthwave stripes + glow
        cx, cy, r = GW // 2, HY - 34, 90
        ov = img.copy()
        cv2.circle(ov, (cx, cy), 175, sun, -1, AA)
        img = cv2.addWeighted(ov, 0.10, img, 0.90, 0)
        ov = img.copy()
        cv2.circle(ov, (cx, cy), 125, sun, -1, AA)
        img = cv2.addWeighted(ov, 0.14, img, 0.86, 0)
        cv2.circle(img, (cx, cy), r, sun, -1, AA)
        for i in range(6):
            y = cy + 8 + i * 13
            c = tuple(int(v) for v in sky[min(HY - 1, y)])
            cv2.rectangle(img, (cx - r, y), (cx + r, y + 2 + i), c, -1)

        # skyline
        x = -10
        while x < GW:
            w, h = rng.randint(22, 62), rng.randint(20, 100)
            cv2.rectangle(img, (x, HY - h), (x + w, HY + 2), bl, -1)
            for _ in range(w * h // 260):
                wx, wy = x + rng.randint(3, max(4, w - 5)), HY - h + rng.randint(4, max(5, h - 4))
                if rng.random() < 0.55:
                    cv2.rectangle(img, (wx, wy), (wx + 1, wy + 1), CYAN if boost else bgr("#ffd25a"), -1)
            x += w + rng.randint(0, 6)
        cv2.line(img, (0, HY), (GW, HY), hor, 2, AA)
        img[HY:] = ground
        return img

    def _build_panel_bg(self):
        t = np.linspace(0, 1, H, dtype=np.float32)[:, None]
        col = np.array(bgr("#160c34"), np.float32) * (1 - t) + np.array(bgr("#07041a"), np.float32) * t
        img = np.repeat(col[:, None, :], PW, axis=1).astype(np.uint8)
        for y in range(0, H, 22):
            img[y] = np.clip(img[y].astype(np.int16) + 8, 0, 255).astype(np.uint8)
        cv2.line(img, (1, 0), (1, H), CYAN, 2, AA)
        return img

    def _vignette(self, col):
        yy, xx = np.mgrid[0:GH, 0:GW].astype(np.float32)
        d = np.sqrt(((xx - GW / 2) / (GW / 2)) ** 2 + ((yy - GH / 2) / (GH / 2)) ** 2) / 1.4142
        s = np.clip((d - 0.42) / 0.58, 0, 1) ** 2
        return (s[..., None] * np.array(col, np.float32)).astype(np.uint8)

    # ------------------------------------------------------------ projection
    def proj(self, x, h, z):
        d = z + CAM_Z
        if d < 0.6:
            d = 0.6
        s = K1 / d
        return GW / 2 + (x - self.cam_x) * s + self.sx, HY + (CAMH - h) * s + self.sy, s

    def projv(self, x, h, z):
        d = np.maximum(z + CAM_Z, 0.6)
        s = K1 / d
        return np.stack([GW / 2 + (x - self.cam_x) * s + self.sx, HY + (CAMH - h) * s + self.sy], axis=-1)

    def P(self, x, h, z):
        a = self.proj(x, h, z)
        return a[0], a[1]

    def fog(self, col, z):
        f = min(1.0, max(0.0, z) / FAR) ** 1.4 * 0.92
        return lerp_col(col, self.fogc, f)

    # ------------------------------------------------------------ 3-D primitives
    def box(self, img, x0, x1, h0, h1, z0, z1, top, front, side):
        if z1 <= NEAR_Z + 0.2:
            return None, None
        z0c = max(z0, NEAR_Z)
        P = self.P
        cx = self.cam_x
        xf = x0 if x0 > cx else (x1 if x1 < cx else None)
        if xf is not None:
            poly(img, [P(xf, h0, z0c), P(xf, h0, z1), P(xf, h1, z1), P(xf, h1, z0c)], side)
        if CAMH > h1:
            poly(img, [P(x0, h1, z0c), P(x1, h1, z0c), P(x1, h1, z1), P(x0, h1, z1)], top)
        fq = None
        if z0 > NEAR_Z + 0.3:
            fq = [P(x0, h0, z0), P(x1, h0, z0), P(x1, h1, z0), P(x0, h1, z0)]
            poly(img, fq, front)
        return fq, xf

    @staticmethod
    def qp(q, u, v):
        bx, by = q[0][0] + u * (q[1][0] - q[0][0]), q[0][1] + u * (q[1][1] - q[0][1])
        tx, ty = q[3][0] + u * (q[2][0] - q[3][0]), q[3][1] + u * (q[2][1] - q[3][1])
        return bx + v * (tx - bx), by + v * (ty - by)

    def qpoly(self, img, q, u0, u1, v0, v1, col):
        poly(img, [self.qp(q, u0, v0), self.qp(q, u1, v0), self.qp(q, u1, v1), self.qp(q, u0, v1)], col)

    # ------------------------------------------------------------ world
    def _band_quad(self, img, x0, x1, z0, z1, col):
        poly(img, [self.P(x0, 0, z0), self.P(x1, 0, z0), self.P(x1, 0, z1), self.P(x0, 0, z1)], col)

    def _draw_track(self, img, g):
        bands = [-5, 3, 8, 14, 22, 32, 44, 58, 76, 98, 124, 150, 178]
        for a, b in zip(bands[:-1], bands[1:]):
            m = (a + b) * 0.5
            self._band_quad(img, -1.55, 1.55, a, b, self.fog(bgr("#1b1638"), m))
            for sgn in (-1, 1):
                self._band_quad(img, sgn * 1.55, sgn * 2.7, a, b, self.fog(bgr("#4a3d78"), m))
                self._band_quad(img, sgn * 1.5, sgn * 1.62, a, b, self.fog(bgr("#ffc21a"), m))
                self._band_quad(img, sgn * 2.7, sgn * 4.5, a, b, self.fog(bgr("#22184a"), m))

        # sleepers (scroll with distance)
        sp = 2.4
        off = -(g.distance % sp)
        for k in range(-3, 74):
            z = off + k * sp
            if z < NEAR_Z + 0.4 or z > FAR:
                continue
            a, b = self.P(-1.5, 0, z), self.P(1.5, 0, z)
            s = K1 / (z + CAM_Z)
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                     self.fog(bgr("#5b4a8c"), z), max(1, int(0.09 * s)), AA)

        # rails
        for x in (-1.3, -0.7, -0.3, 0.3, 0.7, 1.3):
            for a, b in ((NEAR_Z, 22), (22, 70), (70, FAR)):
                self._band_quad(img, x - 0.035, x + 0.035, a, b, self.fog(bgr("#cfd8ff"), (a + b) / 2))

        # neon dashed lane lines
        sp = 7.0
        off = -(g.distance % sp)
        for k in range(-1, 26):
            z = off + k * sp
            for x, c in ((-0.5, CYAN), (0.5, MAG)):
                za, zb = max(NEAR_Z, z), z + 3.0
                if zb < NEAR_Z + 0.4 or za > FAR:
                    continue
                self._band_quad(img, x - 0.03, x + 0.03, za, zb, self.fog(c, za))

    def _draw_tower(self, img, e, z):
        front, side, top, win = TOWER_PAL[e.color]
        x0, x1 = e.x - e.width / 2, e.x + e.width / 2
        z1 = z + e.length
        fq, xf = self.box(img, x0, x1, 0, e.height, z, z1, self.fog(top, z), self.fog(front, z),
                          self.fog(side, z))
        if xf is not None and z < 85:
            rows = int((e.height - 0.5) // 1.1)
            cols = int((e.length - 0.4) // 1.3)
            if rows > 0 and cols > 0:
                hh, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
                lit = ((hh * 7 + cc * 13 + int(e.phase * 100)) % 5) != 0
                hh, cc = hh[lit].astype(np.float32), cc[lit].astype(np.float32)
                if hh.size:
                    ha, hb = 0.6 + hh * 1.1, 0.6 + hh * 1.1 + 0.6
                    za = np.maximum(z + 0.4 + cc * 1.3, NEAR_Z + 0.3)
                    zb = z + 0.4 + cc * 1.3 + 0.7
                    ok = zb > NEAR_Z + 0.5
                    xs = np.full_like(ha, xf)
                    pts = np.stack([self.projv(xs, ha, za), self.projv(xs, ha, zb),
                                    self.projv(xs, hb, zb), self.projv(xs, hb, za)], axis=1)[ok]
                    if len(pts):
                        cv2.fillPoly(img, list(pts.astype(np.int32)), self.fog(win, z + 10), AA)
        if fq is not None and z < 60:
            cv2.line(img, (int(fq[3][0]), int(fq[3][1])), (int(fq[2][0]), int(fq[2][1])),
                     self.fog(CYAN if e.color % 2 else MAG, z), 1, AA)

    def _draw_train(self, img, e, z, t):
        front, side, accent = TRAIN_PAL[e.color % 3]
        x0, x1 = e.x - 0.47, e.x + 0.47
        z1 = z + e.length
        fq, xf = self.box(img, x0, x1, 0.0, 2.4, z, z1, self.fog(shade(front, 1.2), z),
                          self.fog(front, z), self.fog(side, z))
        if xf is not None and z < 95:
            n = int(e.length // 2.4)
            if n > 0:
                ks = np.arange(n, dtype=np.float32)
                za = np.maximum(z + 0.9 + ks * 2.4, NEAR_Z + 0.3)
                zb = z + 0.9 + ks * 2.4 + 1.5
                ok = zb > NEAR_Z + 0.5
                xs = np.full_like(ks, xf)
                pts = np.stack([self.projv(xs, xs * 0 + 1.15, za), self.projv(xs, xs * 0 + 1.15, zb),
                                self.projv(xs, xs * 0 + 2.0, zb), self.projv(xs, xs * 0 + 2.0, za)],
                               axis=1)[ok]
                if len(pts):
                    cv2.fillPoly(img, list(pts.astype(np.int32)), self.fog(bgr("#ffe9a0"), z), AA)
            a = np.array([self.P(xf, 0.5, max(z, NEAR_Z + 0.3)), self.P(xf, 0.5, z1),
                          self.P(xf, 0.68, z1), self.P(xf, 0.68, max(z, NEAR_Z + 0.3))])
            poly(img, a, self.fog(accent, z))
        if fq is not None:
            wpx = fq[1][0] - fq[0][0]
            if wpx > 14:
                self.qpoly(img, fq, 0.10, 0.90, 0.50, 0.90, self.fog(bgr("#132247"), z))
                self.qpoly(img, fq, 0.0, 1.0, 0.34, 0.41, self.fog(accent, z))
                for u in (0.2, 0.8):
                    px, py = self.qp(fq, u, 0.20)
                    r = max(1, int(wpx * 0.06))
                    cv2.circle(img, (int(px), int(py)), r, self.fog(bgr("#fff6c0"), z), -1, AA)
                    if wpx > 40:
                        glow(img, px, py, r * 3, bgr("#fff6c0"), 0.12)
            for i in range(4):
                pass
            cv2.polylines(img, [np.asarray(fq, np.int32)], True, self.fog(bgr("#120c26"), z), 1, AA)

    def _draw_barrier(self, img, e, z, t):
        x0, x1 = e.x - 0.47, e.x + 0.47
        fq, xf = self.box(img, x0, x1, 0, 0.95, z, z + 1.0, self.fog(bgr("#ff8f8f"), z),
                          self.fog(bgr("#e8e8f5"), z), self.fog(bgr("#b8b8d0"), z))
        if fq is not None:
            for i in range(4):
                if i % 2 == 0:
                    self.qpoly(img, fq, 0, 1, i / 4, (i + 1) / 4, self.fog(RED, z))
            wpx = fq[1][0] - fq[0][0]
            cv2.polylines(img, [np.asarray(fq, np.int32)], True, self.fog(bgr("#120c26"), z), 1, AA)
            if wpx > 8:
                px, py = self.qp(fq, 0.5, 1.12)
                on = int(t * 4) % 2 == 0
                r = max(1, int(wpx * 0.06))
                cv2.circle(img, (int(px), int(py)), r, ORANGE if on else bgr("#663310"), -1, AA)
                if on and wpx > 30:
                    glow(img, px, py, r * 4, ORANGE, 0.16)

    def _draw_overhead(self, img, e, z, t):
        x0, x1 = e.x - 0.47, e.x + 0.47
        for a, b in ((x0, x0 + 0.08), (x1 - 0.08, x1)):
            self.box(img, a, b, 0, 1.0, z, z + 1.0, self.fog(bgr("#9a9ab8"), z),
                     self.fog(bgr("#7f7fa0"), z), self.fog(bgr("#5c5c7c"), z))
        fq, xf = self.box(img, x0, x1, 1.0, 2.0, z, z + 1.0, self.fog(bgr("#ffe066"), z),
                          self.fog(bgr("#1a1a24"), z), self.fog(bgr("#8a7a20"), z))
        if fq is not None:
            for i in range(0, 8, 2):
                self.qpoly(img, fq, i / 8, (i + 1) / 8, 0, 1, self.fog(bgr("#ffcc1a"), z))
            wpx = fq[1][0] - fq[0][0]
            if wpx > 30:  # "slide" arrow
                a = self.qp(fq, 0.5, 0.15)
                b, c = self.qp(fq, 0.32, 0.55), self.qp(fq, 0.68, 0.55)
                poly(img, [a, b, c], WHITE)
            cv2.polylines(img, [np.asarray(fq, np.int32)], True, self.fog(bgr("#120c26"), z), 1, AA)

    def _draw_coin(self, img, e, z, g):
        x, y, s = self.proj(e.x, e.h, z)
        r = 0.27 * s
        if r < 1.6:
            cv2.circle(img, (int(x), int(y)), 1, self.fog(GOLD, z), -1)
            return
        wob = abs(math.cos(e.phase + self.t * 4.0))
        rx, ry = max(1, int(r * (0.25 + 0.75 * wob))), int(r)
        if g.boosting and r > 6:
            glow(img, x, y, r * 2.2, GOLD, 0.10)
        cv2.ellipse(img, (int(x), int(y)), (rx, ry), 0, 0, 360, self.fog(bgr("#b8790a"), z), -1, AA)
        cv2.ellipse(img, (int(x), int(y)), (max(1, rx - 2), max(1, ry - 2)), 0, 0, 360,
                    self.fog(GOLD, z), -1, AA)
        if ry > 8:
            cv2.ellipse(img, (int(x), int(y)), (max(1, int(rx * .55)), max(1, int(ry * .55))), 0, 0, 360,
                        self.fog(bgr("#ffe98a"), z), 1, AA)

    # ------------------------------------------------------------ runner
    def _draw_runner(self, img, g):
        fx, fy, s = self.proj(g.px, g.jump_y, 0.0)
        gx, gy, _ = self.proj(g.px, 0.0, 0.0)
        u = 0.62 * s
        # shadow
        k = 1.0 / (1.0 + g.jump_y * 0.7)
        cv2.ellipse(img, (int(gx), int(gy)), (int(0.34 * u * 1.6 * k), max(2, int(0.09 * u * 1.6 * k))),
                    0, 0, 360, bgr("#08051a"), -1, AA)
        if g.invuln > 0 and not g.boosting and g.state == "playing" and int(self.t * 14) % 2 == 0:
            return

        Y = lambda h: fy - h * u
        X = lambda x: fx + x * u
        c, ph = self.crouch, g.run_phase
        air = not g.on_ground
        hip, sho, hd = 0.88 - 0.46 * c, 1.42 - 0.62 * c, 1.66 - 0.66 * c

        # energy aura
        cy_mid = Y(0.95 - 0.3 * c)
        if g.boosting:
            mix = 0.5 + 0.5 * math.sin(self.t * 9)
            col = lerp_col(ORANGE, CYAN, mix)
            glow(img, fx, cy_mid, u * (2.3 + 0.25 * g.boost_power), col, 0.34)
            glow(img, fx, cy_mid, u * 1.2, WHITE, 0.22)
        elif g.boost_ready:
            pul = 0.5 + 0.5 * math.sin(self.t * 8)
            glow(img, fx, cy_mid, u * (1.5 + 0.2 * pul), GOLD, 0.26)
            cv2.ellipse(img, (int(gx), int(gy)), (int(0.7 * u * (1 + .1 * pul)), int(0.2 * u)), 0, 0, 360,
                        GOLD, 2, AA)

        JEANS, SHOE, SKIN = bgr("#2c4fa8"), bgr("#f3f3ff"), bgr("#e0a878")
        HOOD, HOOD2, PACK = bgr("#ff5a3c"), bgr("#c93a2a"), bgr("#ffd23a")
        HAIR, CAP = bgr("#2a1810"), bgr("#26d7ff")
        if g.boosting:
            HOOD, HOOD2 = bgr("#ffb020"), bgr("#e07a10")

        sw = math.sin(ph)
        if air:
            lf, rf, lx, rx = 0.44, 0.20, -0.22, 0.17
        else:
            lf = 0.07 + max(0.0, sw) * 0.36 * (1 - c) + c * 0.10
            rf = 0.07 + max(0.0, -sw) * 0.36 * (1 - c) + c * 0.10
            lx, rx = -0.13, 0.13
        tl = max(2, int(0.22 * u))
        for sx_, fh, fxx in ((-0.11, lf, lx), (0.11, rf, rx)):
            cv2.line(img, (int(X(sx_)), int(Y(hip))), (int(X(fxx)), int(Y(fh + 0.05))), JEANS, tl, AA)
            cv2.ellipse(img, (int(X(fxx)), int(Y(fh))), (max(2, int(0.16 * u)), max(1, int(0.075 * u))),
                        0, 0, 360, SHOE, -1, AA)

        torso = [(X(-0.28), Y(sho)), (X(0.28), Y(sho)), (X(0.23), Y(hip)), (X(-0.23), Y(hip))]
        poly(img, torso, HOOD)
        poly(img, [(X(0.0), Y(sho)), (X(0.28), Y(sho)), (X(0.23), Y(hip)), (X(0.0), Y(hip))], HOOD2)
        poly(img, [(X(-0.19), Y(sho - 0.05)), (X(0.19), Y(sho - 0.05)), (X(0.17), Y(hip + 0.06)),
                   (X(-0.17), Y(hip + 0.06))], PACK)
        cv2.line(img, (int(X(-0.19)), int(Y(sho - 0.12))), (int(X(-0.14)), int(Y(sho + 0.0))), DARK, 2, AA)
        cv2.line(img, (int(X(0.19)), int(Y(sho - 0.12))), (int(X(0.14)), int(Y(sho + 0.0))), DARK, 2, AA)

        for side, a in ((-1, sw), (1, -sw)):
            if air:
                hx, hh = side * 0.50, sho + 0.02
            elif c > 0.5:
                hx, hh = side * 0.32, sho + 0.04
            else:
                hx, hh = side * (0.40 + 0.03 * a), sho - 0.48 + a * 0.16
            cv2.line(img, (int(X(side * 0.30)), int(Y(sho - 0.05))), (int(X(hx)), int(Y(hh))), HOOD2,
                     max(2, int(0.14 * u)), AA)
            cv2.circle(img, (int(X(hx)), int(Y(hh))), max(2, int(0.075 * u)), SKIN, -1, AA)

        cv2.rectangle(img, (int(X(-0.06)), int(Y(sho + 0.08))), (int(X(0.06)), int(Y(sho - 0.02))), SKIN, -1)
        hc, hr = (int(X(0)), int(Y(hd))), max(3, int(0.215 * u))
        cv2.circle(img, hc, hr, HAIR, -1, AA)
        cv2.ellipse(img, hc, (hr, hr), 0, 180, 360, CAP, -1, AA)
        cv2.ellipse(img, (hc[0], hc[1] - int(hr * .2)), (int(hr * .95), int(hr * .35)), 0, 180, 360,
                    bgr("#ffffff"), 1, AA)
        for side in (-1, 1):
            cv2.circle(img, (int(X(side * 0.215)), int(Y(hd - 0.02))), max(1, int(0.045 * u)), SKIN, -1, AA)
        if g.state == "over":  # dizzy stars
            for i in range(3):
                a = self.t * 4 + i * 2.09
                cv2.circle(img, (int(hc[0] + math.cos(a) * hr * 1.6), int(hc[1] - hr * 1.3 + math.sin(a) * hr * .45)),
                           max(2, int(hr * .18)), GOLD, -1, AA)

    # ------------------------------------------------------------ effects
    def _consume_fx(self, g):
        fx, fy, s = self.proj(g.px, g.jump_y, 0.0)
        gx, gy, _ = self.proj(g.px, 0.0, 0.0)
        u = 0.62 * s
        R = self.rng
        for f in g.fx:
            k = f[0]
            if k == "coin":
                _, x, h, val = f
                sx, sy, _ = self.proj(x, h, 0.6)
                for _ in range(5):
                    a = R.random() * 6.28
                    self.parts.append(Part(sx, sy, math.cos(a) * 140, math.sin(a) * 140 - 40, 0.35, 3,
                                           lerp_col(GOLD, WHITE, R.random() * .6), 260))
                self.parts.append(Part(sx, sy - 20, 0, -70, 0.6, 0, GOLD, 0, "text", f"+{val}"))
                if not g.boosting:
                    self.flyers.append([sx, sy, 0.0, R.uniform(-60, 60)])
            elif k == "crash":
                for _ in range(40):
                    a = R.random() * 6.28
                    v = R.uniform(120, 480)
                    self.parts.append(Part(fx, fy - 0.9 * u, math.cos(a) * v, math.sin(a) * v - 120,
                                           R.uniform(.4, .9), R.uniform(3, 7),
                                           R.choice((RED, ORANGE, WHITE, GOLD)), 700))
                self.parts.append(Part(fx, fy - 0.9 * u, 0, 0, 0.5, 260, RED, 0, "ring"))
                self.parts.append(Part(GW / 2, 300, 0, -30, 0.9, 0, RED, 0, "text", "OUCH!"))
            elif k == "smash":
                _, x, kind = f
                sx, sy, _ = self.proj(x, 1.0, 3.0)
                for _ in range(45):
                    a = R.random() * 6.28
                    v = R.uniform(150, 600)
                    self.parts.append(Part(sx, sy, math.cos(a) * v, math.sin(a) * v - 100, R.uniform(.4, .9),
                                           R.uniform(3, 8), R.choice((CYAN, WHITE, ORANGE, MAG, GOLD)), 600))
                self.parts.append(Part(sx, sy, 0, -40, 0.7, 0, CYAN, 0, "text", "SMASH! +50"))
            elif k == "boost":
                self.white = 1.0
                for i in range(70):
                    a = R.random() * 6.28
                    v = R.uniform(200, 700)
                    self.parts.append(Part(fx, fy - 0.9 * u, math.cos(a) * v, math.sin(a) * v, R.uniform(.5, 1.1),
                                           R.uniform(3, 8), R.choice((GOLD, ORANGE, CYAN, WHITE)), 100))
                for d in (0, 0.1, 0.2):
                    self.parts.append(Part(fx, fy - 0.9 * u, 0, 0, 0.7 + d, 380 + d * 400, GOLD if d == 0 else CYAN,
                                           0, "ring"))
            elif k in ("jump", "land"):
                for _ in range(8):
                    self.parts.append(Part(gx + R.uniform(-30, 30), gy, R.uniform(-90, 90), R.uniform(-60, -10),
                                           0.4, R.uniform(3, 6), bgr("#9c8fc8"), 120))
            elif k == "life":
                self.parts.append(Part(GW - 120, 70, 0, -40, 1.0, 0, RED, 0, "text", "+1 LIFE"))
            elif k == "bucket_full":
                for _ in range(30):
                    a = R.random() * 6.28
                    v = R.uniform(60, 240)
                    self.cparts.append(Part(GW + 96, 520, math.cos(a) * v, math.sin(a) * v, 0.8, 4,
                                            R.choice((GOLD, WHITE, CYAN)), 100))
            elif k == "boost_end":
                for _ in range(30):
                    a = R.random() * 6.28
                    v = R.uniform(100, 300)
                    self.parts.append(Part(fx, fy - 0.9 * u, math.cos(a) * v, math.sin(a) * v, 0.6, 4,
                                           R.choice((CYAN, WHITE)), 200))
        g.fx.clear()

    @staticmethod
    def _step_particles(parts, dt):
        alive = []
        for p in parts:
            p.life -= dt
            if p.life <= 0:
                continue
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.vy += p.grav * dt
            alive.append(p)
        return alive[-500:]

    def _update_particles(self, dt):
        self.parts = self._step_particles(self.parts, dt)
        self.cparts = self._step_particles(self.cparts, dt)

    def _draw_particles(self, img, parts=None):
        for p in (self.parts if parts is None else parts):
            f = p.life / p.max
            if p.kind == "dot":
                cv2.circle(img, (int(p.x), int(p.y)), max(1, int(p.size * (0.3 + 0.7 * f))), p.col, -1, AA)
            elif p.kind == "ring":
                cv2.circle(img, (int(p.x), int(p.y)), max(1, int(p.size * (1 - f) ** 0.6)), p.col,
                           max(1, int(6 * f)), AA)
            else:
                text(img, p.txt, (p.x, p.y), 0.8 if len(p.txt) < 4 else 0.9, p.col, 2, anchor="c")

    def _update_flyers(self, dt, g):
        tx, ty = GW + 96, 512
        keep = []
        for fl in self.flyers:
            fl[2] += dt / 0.65
            if fl[2] >= 1.0:
                self.disp = min(self.disp + 1.0, g.bucket)
                self.bump = 1.0
                for _ in range(4):
                    a = self.rng.random() * 6.28
                    self.cparts.append(Part(tx, ty, math.cos(a) * 90, math.sin(a) * 90 - 40, 0.3, 3, GOLD, 200))
            else:
                keep.append(fl)
        self.flyers = keep

    def _draw_flyers(self, canvas):
        tx, ty = GW + 96, 512
        for x0, y0, t, wob in self.flyers:
            e = t * t * (3 - 2 * t)
            mx, my = (x0 + tx) / 2 + wob, min(y0, ty) - 90
            x = (1 - e) ** 2 * x0 + 2 * (1 - e) * e * mx + e * e * tx
            y = (1 - e) ** 2 * y0 + 2 * (1 - e) * e * my + e * e * ty
            glow(canvas, x, y, 16, GOLD, 0.14)
            coin_icon(canvas, x, y, 9 * (1 - 0.3 * e), t * 14)

    # ------------------------------------------------------------ HUD (game view)
    def _draw_hud(self, img, g):
        # score card
        alpha_rect(img, 12, 12, 236, 96, DARK, 0.55)
        cv2.rectangle(img, (12, 12), (236, 96), CYAN, 1, AA)
        text(img, "SCORE", (24, 34), 0.45, CYAN, 1, FONT2)
        text(img, f"{g.score:,}", (24, 70), 1.15, WHITE, 2)
        text(img, f"{g.meters} m", (140, 34), 0.5, (220, 200, 255), 1, FONT2)
        coin_icon(img, 148, 84, 7, self.t * 3)
        text(img, f"x {g.coins}", (162, 89), 0.55, GOLD, 1, FONT2)

        # lives + best
        for i in range(MAX_LIVES):
            heart(img, GW - 34 - i * 34, 34, 11, RED if i < g.lives else bgr("#3a2a4a"))
        text(img, f"BEST {g.best:,}", (GW - 16, 74), 0.5, (220, 200, 255), 1, FONT2, anchor="r")

        # boost banner / timer
        if g.boosting:
            w = 260
            x0 = GW // 2 - w // 2
            alpha_rect(img, x0 - 8, 10, x0 + w + 8, 52, DARK, 0.55)
            text(img, f"ENERGY BOOST x{g.boosts_used}", (GW // 2, 30), 0.6, ORANGE, 1, anchor="c")
            cv2.rectangle(img, (x0, 38), (x0 + w, 46), bgr("#3a2a4a"), -1)
            cv2.rectangle(img, (x0, 38), (x0 + int(w * g.energy), 46), lerp_col(ORANGE, CYAN, 0.5 + 0.5 * math.sin(self.t * 9)), -1)
        elif g.boost_ready and int(self.t * 3) % 2 == 0:
            alpha_rect(img, GW // 2 - 190, 12, GW // 2 + 190, 50, DARK, 0.6)
            text(img, "BUCKET FULL!  MAKE A FIST", (GW // 2, 39), 0.7, GOLD, 2, anchor="c")

        # last gesture
        txt, t0 = g.action
        age = g.t - t0
        if txt and age < 0.5 and g.state == "playing":
            sc = 1.5 - age * 0.8
            col = {"LEFT": CYAN, "RIGHT": CYAN, "JUMP": GREEN, "SLIDE": MAG}.get(txt, WHITE)
            text(img, txt, (GW // 2, 118), sc, col, 3, anchor="c")

        if g.toast_t > 0 and g.toast_text:
            col = {"boost": ORANGE, "ready": GOLD}.get(g.toast_kind, WHITE)
            (tw, _), _ = cv2.getTextSize(g.toast_text, FONT, 0.95, 2)
            alpha_rect(img, GW // 2 - tw // 2 - 20, 250, GW // 2 + tw // 2 + 20, 296, DARK, 0.6)
            text(img, g.toast_text, (GW // 2, 284), 0.95, col, 2, anchor="c")

    def _draw_overlay(self, img, g, eng):
        if g.state == "playing":
            return
        alpha_rect(img, 0, 0, GW, GH, (10, 5, 25), 0.55)
        pul = 0.5 + 0.5 * math.sin(self.t * 5)
        if g.state == "ready":
            text(img, "GESTURE RUNNER", (GW // 2, 190), 1.9, CYAN, 6, anchor="c", outline=False)
            text(img, "GESTURE RUNNER", (GW // 2, 190), 1.9, WHITE, 2, anchor="c", outline=False)
            text(img, "Subway dash powered by your hand", (GW // 2, 232), 0.7, (220, 200, 255), 1, anchor="c")
            lines = ["1  Show your hand to the camera",
                     "2  Flick out of the centre box to steer / jump / slide",
                     "3  Collect coins to fill the bucket",
                     "4  Full bucket + FIST = huge energy boost"]
            for i, l in enumerate(lines):
                text(img, l, (GW // 2 - 250, 300 + i * 34), 0.62, WHITE, 1, FONT2)
            col = lerp_col(GOLD, WHITE, pul)
            text(img, "HOLD A FIST TO START   (or press SPACE)", (GW // 2, 520), 0.85, col, 2, anchor="c")
            if eng.fist_progress > 0:
                cv2.rectangle(img, (GW // 2 - 150, 540), (GW // 2 + 150, 552), bgr("#3a2a4a"), -1)
                cv2.rectangle(img, (GW // 2 - 150, 540), (GW // 2 - 150 + int(300 * eng.fist_progress), 552), GOLD, -1)
        else:
            text(img, "GAME OVER", (GW // 2, 200), 2.0, RED, 6, anchor="c", outline=False)
            text(img, "GAME OVER", (GW // 2, 200), 2.0, WHITE, 2, anchor="c", outline=False)
            text(img, f"SCORE  {g.score:,}", (GW // 2, 270), 1.0, WHITE, 2, anchor="c")
            text(img, f"BEST  {g.best:,}", (GW // 2, 312), 0.75, GOLD, 1, anchor="c")
            text(img, f"{g.meters} m    {g.coins} coins    {g.smashes} smashes", (GW // 2, 352), 0.65,
                 (220, 200, 255), 1, FONT2, anchor="c")
            if g.over_t > 0.8:
                text(img, "HOLD A FIST TO RUN AGAIN", (GW // 2, 470), 0.85, lerp_col(GOLD, WHITE, pul), 2, anchor="c")
                if eng.fist_progress > 0:
                    cv2.rectangle(img, (GW // 2 - 150, 490), (GW // 2 + 150, 502), bgr("#3a2a4a"), -1)
                    cv2.rectangle(img, (GW // 2 - 150, 490), (GW // 2 - 150 + int(300 * eng.fist_progress), 502), GOLD, -1)

    # ------------------------------------------------------------ side panel
    def _draw_camera(self, p, frame, obs, eng):
        x0, y0, w, h = 16, 40, 288, 216
        if frame is None:
            cam = np.full((h, w, 3), bgr("#120c26"), np.uint8)
            text(cam, "NO CAMERA", (w // 2, h // 2 - 6), 0.8, RED, 2, anchor="c")
            text(cam, "keyboard mode: W A S D + SPACE", (w // 2, h // 2 + 22), 0.45, WHITE, 1, FONT2, anchor="c")
        else:
            cam = cv2.resize(frame, (w, h))
            cam = cv2.addWeighted(cam, 0.82, np.full_like(cam, bgr("#1a0f3c")), 0.18, 0)

        # neutral zone
        nx, ny = eng.neutral
        zx0, zy0 = int((nx - eng.dead_x) * w), int((ny - eng.dead_y) * h)
        zx1, zy1 = int((nx + eng.dead_x) * w), int((ny + eng.dead_y) * h)
        zc = GREEN if obs.present and eng.armed else lerp_col(GREEN, DARK, 0.4)
        cv2.rectangle(cam, (zx0, zy0), (zx1, zy1), zc, 1, AA)
        L = 9
        for (ax, ay, dx, dy) in ((zx0, zy0, 1, 1), (zx1, zy0, -1, 1), (zx0, zy1, 1, -1), (zx1, zy1, -1, -1)):
            cv2.line(cam, (ax, ay), (ax + dx * L, ay), zc, 2, AA)
            cv2.line(cam, (ax, ay), (ax, ay + dy * L), zc, 2, AA)
        cxm, cym = (zx0 + zx1) // 2, (zy0 + zy1) // 2
        now = self.t
        for d, (px_, py_) in {"left": (zx0 - 16, cym), "right": (zx1 + 16, cym),
                              "up": (cxm, zy0 - 14), "down": (cxm, zy1 + 14)}.items():
            lit = eng.direction == d or (eng.flash.get(d, -9) > eng.flash.get("_", -9) and False)
            tri(cam, px_, py_, 7, d, CYAN if lit else bgr("#4a3d78"))

        if obs.present:
            col = ORANGE if obs.fist else CYAN
            if obs.points:
                pts = [(int(x * w), int(y * h)) for x, y in obs.points]
                for a, b in obs.connections or []:
                    cv2.line(cam, pts[a], pts[b], col, 2, AA)
                for q in pts:
                    cv2.circle(cam, q, 3, WHITE, -1, AA)
            elif obs.outline:
                cv2.polylines(cam, [np.asarray([(x * w, y * h) for x, y in obs.outline], np.int32)], True, col, 2, AA)
            hx, hy = int(eng.sx * w), int(eng.sy * h)
            cv2.line(cam, (cxm, cym), (hx, hy), (255, 255, 255), 1, AA)
            cv2.circle(cam, (hx, hy), 7, col, 2, AA)
            cv2.circle(cam, (hx, hy), 2, WHITE, -1, AA)
            tag = "FIST" if obs.fist else "HAND OK"
            text(cam, tag, (8, h - 10), 0.5, col, 1, FONT2)
        elif frame is not None:
            if int(self.t * 2.5) % 2 == 0:
                text(cam, "SHOW YOUR HAND", (w // 2, h - 12), 0.6, GOLD, 1, anchor="c")

        p[y0:y0 + h, x0:x0 + w] = cam
        cv2.rectangle(p, (x0 - 1, y0 - 1), (x0 + w, y0 + h), CYAN, 2, AA)

    def _draw_pad(self, p, g, eng):
        cx, cy = 72, 334
        cv2.circle(p, (cx, cy), 58, bgr("#100a28"), -1, AA)
        cv2.circle(p, (cx, cy), 58, bgr("#4a3d78"), 2, AA)
        for d, (dx, dy) in {"up": (0, -38), "down": (0, 38), "left": (-38, 0), "right": (38, 0)}.items():
            lit = self.t - eng.flash.get(d, -9) < 0.30 and g.state == "playing"
            if lit:
                glow(p, cx + dx, cy + dy, 22, CYAN, 0.25)
            tri(p, cx + dx, cy + dy, 11, d, CYAN if lit else bgr("#4a3d78"))
        prog = eng.fist_progress
        cv2.circle(p, (cx, cy), 21, bgr("#1c1240"), -1, AA)
        cv2.circle(p, (cx, cy), 21, bgr("#4a3d78"), 2, AA)
        if prog > 0:
            cv2.ellipse(p, (cx, cy), (21, 21), -90, 0, int(360 * prog), ORANGE, 4, AA)
        text(p, "FIST", (cx, cy + 5), 0.36, ORANGE if prog > 0 else (180, 170, 220), 1, FONT2, anchor="c", outline=False)

        text(p, "GESTURES", (140, 286), 0.42, CYAN, 1, FONT2)
        cv2.line(p, (140, 291), (306, 291), bgr("#4a3d78"), 1)
        rows = [(("left", "right"), "SWITCH LANE"), (("up",), "JUMP"), (("down",), "SLIDE"), (("fist",), "HOLD = BOOST")]
        for i, (ic, label) in enumerate(rows):
            y = 314 + i * 26
            for j, d in enumerate(ic):
                if d == "fist":
                    cv2.circle(p, (150, y - 4), 8, ORANGE, -1, AA)
                    for kx in (-4, 0, 4):
                        cv2.circle(p, (150 + kx, y - 8), 2, bgr("#ffd0a0"), -1, AA)
                else:
                    tri(p, 150 + j * 20, y - 4, 7, d, CYAN)
            text(p, label, (186 if len(ic) == 1 else 190, y), 0.45, WHITE, 1, FONT2)

    def _draw_bucket(self, p, g):
        cx, top, bot, tw, bw = 96, 450, 588, 66, 50
        wob = math.sin(self.t * 34) * 4 * self.bump
        cxw = cx + wob
        full = g.bucket >= BUCKET_CAP - 0.01
        boosting = g.boosting
        text(p, "COIN BUCKET", (16, 428), 0.45, CYAN, 1, FONT2)
        cv2.line(p, (16, 434), (306, 434), bgr("#4a3d78"), 1)

        if full and not boosting:
            glow(p, cxw, (top + bot) / 2, 110, GOLD, 0.16 + 0.10 * math.sin(self.t * 8))
        elif boosting:
            glow(p, cxw, (top + bot) / 2, 110, lerp_col(ORANGE, CYAN, 0.5 + 0.5 * math.sin(self.t * 9)), 0.18)

        body = np.asarray([(cxw - tw, top), (cxw + tw, top), (cxw + bw, bot), (cxw - bw, bot)], np.int32)
        cv2.fillConvexPoly(p, body, bgr("#241640"), AA)
        n_full = int(self.disp)
        frac = self.disp - n_full
        for i in range(BUCKET_CAP):
            if i > n_full or (i == n_full and frac < 0.05):
                break
            row, col = divmod(i, 4)
            x = cxw + (col - 1.5) * 25 + (row % 2) * 8 - 4
            y = bot - 16 - row * 22
            r = 11 if i < n_full else 11 * frac
            coin_icon(p, x, y, r, 0.0 if i % 3 else 0.6)
        # glass front + neon outline
        alpha_rect(p, int(cxw - tw), top, int(cxw + tw), bot, CYAN, 0.07)
        cv2.polylines(p, [body], True, CYAN, 2, AA)
        cv2.ellipse(p, (int(cxw), top), (tw, 11), 0, 0, 360, CYAN, 2, AA)
        cv2.ellipse(p, (int(cxw), top), (tw + 2, 16), 0, 195, 345, bgr("#6a5aa8"), 3, AA)   # handle
        cv2.rectangle(p, (int(cxw - bw), bot - 8), (int(cxw + bw), bot), bgr("#4a3d78"), -1)
        cv2.polylines(p, [np.asarray([(cxw - bw, bot - 8), (cxw + bw, bot - 8), (cxw + bw, bot), (cxw - bw, bot)], np.int32)],
                      True, CYAN, 1, AA)

        # numbers + energy
        text(p, "COINS", (184, 462), 0.4, (200, 190, 230), 1, FONT2)
        text(p, f"{int(round(self.disp))}/{BUCKET_CAP}", (184, 496), 1.05, GOLD, 2)
        text(p, "ENERGY", (184, 528), 0.42, CYAN, 1, FONT2)
        bx, by, bw2, bh = 184, 536, 122, 22
        e = g.energy
        segs = 10
        for i in range(segs):
            sx0 = bx + int(i * bw2 / segs)
            sx1 = bx + int((i + 1) * bw2 / segs) - 3
            on = e > (i + 0.5) / segs
            base = lerp_col(CYAN, ORANGE, i / (segs - 1))
            if boosting:
                base = lerp_col(ORANGE, WHITE, 0.4 + 0.4 * math.sin(self.t * 12 + i))
            cv2.rectangle(p, (sx0, by), (sx1, by + bh), base if on else bgr("#2a1f4a"), -1)
        bolt(p, bx - 12, by + 11, 9, GOLD if e > 0.05 else bgr("#4a3d78"))

        pul = 0.5 + 0.5 * math.sin(self.t * 8)
        if boosting:
            text(p, "BOOSTING!", (184, 592), 0.62, lerp_col(ORANGE, WHITE, pul), 2)
            text(p, f"{g.boost_t:0.1f}s left", (184, 612), 0.42, WHITE, 1, FONT2)
        elif full:
            text(p, "MAKE A FIST!", (184, 592), 0.62, lerp_col(GOLD, WHITE, pul), 2)
            text(p, "to release energy", (184, 612), 0.42, WHITE, 1, FONT2)
        else:
            text(p, "COLLECT COINS", (184, 592), 0.5, (200, 190, 230), 1, FONT2)
            text(p, f"{BUCKET_CAP - int(g.bucket)} more to boost", (184, 612), 0.42, (200, 190, 230), 1, FONT2)

    def _draw_panel(self, g, eng, obs, frame, backend, fps):
        p = self.panel
        p[:] = self.panel_bg
        text(p, "HAND CONTROL", (16, 26), 0.6, WHITE, 1)
        text(p, backend[:18], (306, 26), 0.36, (170, 160, 210), 1, FONT2, anchor="r")
        self._draw_camera(p, frame, obs, eng)
        self._draw_pad(p, g, eng)
        self._draw_bucket(p, g)
        text(p, "SPACE start   R recenter   C calibrate   ESC quit", (160, 632), 0.34, (150, 140, 190), 1, FONT2,
             anchor="c", outline=False)

    # ------------------------------------------------------------ main entry
    def render(self, g, eng, obs, frame, dt, backend="", fps=0.0):
        dt = min(dt, 0.05)
        self.t += dt
        self.cam_x += (g.px * 0.45 - self.cam_x) * min(1.0, dt * 8)
        self.crouch += ((1.0 if g.slide_t > 0 else 0.0) - self.crouch) * min(1.0, dt * 16)
        self.bump = max(0.0, self.bump - dt * 3.0)
        self.white = max(0.0, self.white - dt * 2.2)
        sh = g.shake
        self.sx, self.sy = self.rng.uniform(-1, 1) * sh * 9, self.rng.uniform(-1, 1) * sh * 7
        a = g.boost_amt
        self.fogc = lerp_col(FOG_N, FOG_B, a)

        # bucket display follows the real value when it drains / resets
        if g.boosting or g.state != "playing" or g.bucket < self.disp:
            self.disp += (g.bucket - self.disp) * min(1.0, dt * 12) if g.bucket < self.disp else 0.0
            if abs(self.disp - g.bucket) < 0.02:
                self.disp = g.bucket
        self.disp = min(self.disp, g.bucket)

        self._consume_fx(g)
        self._update_flyers(dt, g)
        self._update_particles(dt)

        img = self.img
        if a < 0.02:
            img[:] = self.bg_n
        elif a > 0.98:
            img[:] = self.bg_b
        else:
            img[:] = cv2.addWeighted(self.bg_n, 1 - a, self.bg_b, a, 0)

        self._draw_track(img, g)

        ents = []
        for e in g.entities:
            z = e.wz - g.distance
            if z > FAR or z + e.length < NEAR_Z + 0.2:
                continue
            ents.append((z, e))
        ents.sort(key=lambda it: -it[0])
        runner_done = False
        for z, e in ents:
            if not runner_done and z <= 0.0:
                self._draw_runner(img, g)
                runner_done = True
            k = e.kind
            if k == "tower":
                self._draw_tower(img, e, z)
            elif k == "train":
                self._draw_train(img, e, z, self.t)
            elif k == "barrier":
                self._draw_barrier(img, e, z, self.t)
            elif k == "overhead":
                self._draw_overhead(img, e, z, self.t)
            else:
                self._draw_coin(img, e, z, g)
        if not runner_done:
            self._draw_runner(img, g)

        if a > 0.05:                                    # speed lines
            rng = self.rng
            for _ in range(int(10 + 34 * a)):
                ang = rng.uniform(0, 6.283)
                r0 = rng.uniform(140, 300)
                r1 = r0 + rng.uniform(60, 220) * a
                c = (GW // 2, HY + 60)
                p0 = (int(c[0] + math.cos(ang) * r0 * 1.5), int(c[1] + math.sin(ang) * r0))
                p1 = (int(c[0] + math.cos(ang) * r1 * 1.5), int(c[1] + math.sin(ang) * r1))
                cv2.line(img, p0, p1, rng.choice((WHITE, CYAN, GOLD)), 1, AA)

        self._draw_particles(img)
        if a > 0.02:
            img[:] = cv2.add(img, cv2.convertScaleAbs(self.vig_boost, alpha=0.75 * a))
        if g.flash > 0.02:
            img[:] = cv2.add(img, cv2.convertScaleAbs(self.vig_red, alpha=1.1 * g.flash))
        if self.white > 0.02:
            v = int(200 * self.white)
            img[:] = cv2.add(img, (v, v, v, 0))

        self._draw_hud(img, g)
        self._draw_overlay(img, g, eng)

        self._draw_panel(g, eng, obs, frame, backend, fps)
        canvas = np.empty((H, W, 3), np.uint8)
        canvas[:, :GW] = img
        canvas[:, GW:] = self.panel
        self._draw_flyers(canvas)
        self._draw_particles(canvas, self.cparts)
        text(canvas, f"{fps:0.0f} fps", (GW - 10, GH - 10), 0.4, (180, 170, 220), 1, FONT2, anchor="r", outline=False)
        return canvas

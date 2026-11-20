"""A 3D bouquet rendered as coloured ASCII, right in your terminal.

    python -m bouquet.ascii3d                 live, fills your terminal window
    python -m bouquet.ascii3d --gif out.gif   write a shareable GIF instead

How it works
------------
1. The bouquet is a cloud of tens of thousands of 3D points (petals, stems, leaves,
   filler flowers, paper wrapping, ribbon). Every point also knows which way its
   surface faces.
2. Every frame the points are animated (stems grow, petals spiral open), rotated
   around the bouquet, lit by a fixed light, and projected onto a grid of terminal cells.
3. A depth buffer keeps only the nearest point in each cell, and the point's lit
   brightness picks a character from a ramp: dim ones like "." up to bright "@".
4. Each character is coloured with the point's colour.
"""
from __future__ import annotations

import argparse
import functools
import os
import shutil
import sys
import time
from dataclasses import dataclass

import numpy as np

GOLDEN = 2.399963229728653  # the golden angle in radians: nature's favourite spiral
RAMP = " .:-=+*#%@"  # index 0 is blank; brighter points use later characters
BG = (20, 20, 43)
CAMERA_DISTANCE = 30.0
BUILD_SECONDS = 5.0
DEFAULT_FLOWERS = 13
CAPTION_COLOR = (255, 182, 205)

LIGHT = np.array([-0.45, 0.65, 0.75])
LIGHT = LIGHT / np.linalg.norm(LIGHT)

PALETTE = {
    "rose": ["#e63e6d", "#ff6f91", "#f58fa9", "#d62839", "#ff9bb3", "#ffc2c7"],
    "cosmos": ["#b794f6", "#ff8a65", "#ffb08a", "#ff5fa2", "#8f8fff"],
    "daisy": ["#fffaf0", "#fff1a8", "#ffd6e8"],
    "sunflower": ["#ffc83d", "#ffb000", "#ff9f1c"],
}
KIND_CYCLE = ["rose", "cosmos", "daisy", "rose", "sunflower", "cosmos"]
BASE_SIZE = {"rose": 1.55, "cosmos": 1.4, "daisy": 1.25, "sunflower": 1.5}
LEAVES = ["#3fae6e", "#2e8b57", "#5cc28a"]
RIBBONS = ["#e63e6d", "#7c5cff", "#ff9f1c", "#2ec4b6"]
SPARKLE_COLORS = ["#fff2cc", "#ffd1e3", "#cfe8ff", "#ffffff", "#e8d5ff"]


# ----------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------
def _hex(color: str) -> np.ndarray:
    color = color.lstrip("#")
    return np.array([int(color[i : i + 2], 16) / 255 for i in (0, 2, 4)])


def _grad(base: np.ndarray, u: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Per-point colours that run from `lo` (deep) to `hi` (light) as u goes 0 to 1.
    Factors below 1 darken the colour; above 1 they lighten it toward white."""
    f = (lo + (hi - lo) * u)[:, None]
    return np.clip(np.where(f <= 1, base * f, base + (1 - base) * np.minimum(f - 1, 1)), 0, 1)


def _ease_out_back(x):
    """Overshoots slightly before settling, which makes blooms pop. Works on arrays."""
    c1 = 1.70158
    c3 = c1 + 1
    x = np.clip(x, 0.0, 1.0)
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _rotation_to(normal: np.ndarray, spin: float) -> np.ndarray:
    """Rotation matrix that turns +y into `normal`, then spins around it."""
    n = normal / np.linalg.norm(normal)
    up = np.array([0.0, 1.0, 0.0])
    v, c = np.cross(up, n), float(up @ n)
    if np.linalg.norm(v) < 1e-9:
        align = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = _skew(v)
        align = np.eye(3) + vx + vx @ vx / (1 + c)
    k = _skew(n)
    spin_m = np.eye(3) + np.sin(spin) * k + (1 - np.cos(spin)) * (k @ k)
    return spin_m @ align


def _rot_y(v: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]])


def _bezier3(p0, p1, p2, s) -> np.ndarray:
    s = np.asarray(s, dtype=float)[:, None]
    return (1 - s) ** 2 * p0 + 2 * (1 - s) * s * p1 + s**2 * p2


# ----------------------------------------------------------------------------------------
# Building the point cloud
# ----------------------------------------------------------------------------------------
@dataclass
class Cloud:
    """Every point has a final position plus an animation: it grows out of `anchor`
    starting at time `t0` (0 to 1 of the build) and takes `dur` to finish."""

    pos: np.ndarray  # (N, 3) final positions
    anchor: np.ndarray  # (N, 3) the point each part grows out of
    rgb: np.ndarray  # (N, 3) colours, 0 to 1
    lum: np.ndarray  # (N,) base brightness, 0 to 1
    t0: np.ndarray  # (N,)
    dur: np.ndarray  # (N,)
    sparkles: np.ndarray  # (S, 9): radius, start angle, start height, speed, phase, rise, r, g, b
    normal: np.ndarray | None = None  # (N, 3) which way each surface faces; zeros = unlit
    extent: tuple = (5.5, 13.5, -0.4)  # (widest radius, total height, middle height): used to fit the screen


class _Parts:
    def __init__(self):
        self.items = {k: [] for k in ("pos", "anchor", "rgb", "lum", "t0", "dur", "normal")}

    def add(self, pos, anchor, rgb, lum, t0, dur, normal=None):
        pos = np.asarray(pos, dtype=float)
        n = len(pos)
        normal = np.zeros((n, 3)) if normal is None else normal
        for key, value, shape in (
            ("pos", pos, (n, 3)),
            ("anchor", anchor, (n, 3)),
            ("rgb", rgb, (n, 3)),
            ("lum", lum, (n,)),
            ("t0", t0, (n,)),
            ("dur", dur, (n,)),
            ("normal", normal, (n, 3)),
        ):
            self.items[key].append(np.broadcast_to(np.asarray(value, dtype=float), shape))

    def cloud(self, sparkles) -> Cloud:
        data = {k: np.concatenate(v) for k, v in self.items.items()}
        radius = np.percentile(np.hypot(data["pos"][:, 0], data["pos"][:, 2]), 99.7)
        low, high = np.percentile(data["pos"][:, 1], [0.1, 99.9])
        return Cloud(**data, sparkles=sparkles, extent=(float(radius), float(high - low), float((high + low) / 2)))


def _petal_points(count, length, width, curl, lift=0.0, rot=0.0, nu=9, nv=6):
    """Points on `count` petals fanned around the vertical axis.

    Each petal is a teardrop surface that starts at the centre, curls upward
    by `curl`, and sits `lift` above the flower's base.
    Returns (points, distance_along_petal 0..1, petal_index_fraction 0..1, surface normals).
    """
    u = (np.arange(nu) + 0.5) / nu
    v = np.linspace(-1.0, 1.0, nv)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    half = 0.5 * width * np.sin(np.pi * uu**0.7)
    r, lat = uu * length, vv * half
    y = lift + curl * length * uu**2
    slope = 2 * curl * uu  # how steeply the petal rises per unit outward
    norm = np.sqrt(slope**2 + 1)
    chunks, us, angs, normals = [], [], [], []
    for k in range(count):
        a = rot + 2 * np.pi * k / count
        x = r * np.cos(a) - lat * np.sin(a)
        z = r * np.sin(a) + lat * np.cos(a)
        chunks.append(np.stack([x, y, z], -1).reshape(-1, 3))
        normals.append(np.stack([-slope * np.cos(a) / norm, 1 / norm, -slope * np.sin(a) / norm], -1).reshape(-1, 3))
        us.append(uu.reshape(-1))
        angs.append(np.full(uu.size, k / count))
    return np.concatenate(chunks), np.concatenate(us), np.concatenate(angs), np.concatenate(normals)


def _disc_points(radius, n, height):
    """A sunflower-style spiral of points: the centre of a daisy or sunflower."""
    k = np.arange(n)
    r = radius * np.sqrt((k + 0.5) / n)
    theta = k * GOLDEN
    pts = np.stack([r * np.cos(theta), np.full(n, height), r * np.sin(theta)], -1)
    return pts, r / radius, np.tile([0.0, 1.0, 0.0], (n, 1))


YELLOW = np.array([1.0, 0.78, 0.24])


def _flower_local(kind: str, size: float, base: np.ndarray):
    """A flower head around the origin, facing +y.
    Returns (points, colours, brightness, order, normals) where `order` (0..1) is when each
    point blooms within the flower: inner petals first, then spiralling outward."""
    pos, rgb, lum, order, nrm = [], [], [], [], []

    def put(p, color, brightness, o, n):
        pos.append(p)
        rgb.append(np.broadcast_to(color, (len(p), 3)))
        lum.append(np.broadcast_to(brightness, (len(p),)))
        order.append(np.broadcast_to(o, (len(p),)))
        nrm.append(n)

    if kind == "rose":
        layers = 6
        for i in range(layers):
            frac = i / (layers - 1)
            count, length = 9 - i, size * (1 - 0.14 * i)
            p, u, a, n = _petal_points(count, length, length * 0.85, curl=0.5, lift=size * 0.17 * i, rot=i * GOLDEN)
            put(p, _grad(base, u, 0.7 - 0.15 * frac, 1.3 - 0.45 * frac), 0.5 + 0.45 * u * (1 - 0.3 * frac),
                0.5 * (1 - frac) + 0.3 * u + 0.2 * a, n)
    elif kind == "daisy":
        p, u, a, n = _petal_points(21, size, size * 0.22, curl=0.1)
        put(p, _grad(base, u, 0.85, 1.05), 0.6 + 0.4 * u, 0.3 + 0.5 * u + 0.2 * a, n)
        d, r, dn = _disc_points(size * 0.28, 90, size * 0.06)
        put(d, YELLOW, 0.8 + 0.15 * np.sin(r * 14), r * 0.2, dn)
    elif kind == "sunflower":
        p, u, a, n = _petal_points(21, size, size * 0.2, curl=0.15)
        put(p, _grad(base, u, 0.8, 1.1), 0.6 + 0.4 * u, 0.3 + 0.5 * u + 0.2 * a, n)
        p, u, a, n = _petal_points(21, size * 0.8, size * 0.18, curl=0.2, lift=size * 0.04, rot=np.pi / 21)
        put(p, _grad(base, u, 0.95, 1.2), 0.7 + 0.3 * u, 0.2 + 0.4 * u + 0.2 * a, n)
        d, r, dn = _disc_points(size * 0.45, 240, size * 0.08)
        seeds = 0.5 + 0.5 * np.sin(r * 22)
        put(d, _grad(np.array([0.5, 0.3, 0.1]), r, 0.5, 1.0), 0.45 + 0.4 * seeds, r * 0.2, dn)
    else:  # cosmos
        p, u, a, n = _petal_points(8, size, size * 0.62, curl=0.3)
        put(p, _grad(base, u, 0.75, 1.15), 0.5 + 0.5 * u, 0.3 + 0.5 * u + 0.2 * a, n)
        p, u, a, n = _petal_points(8, size * 0.68, size * 0.4, curl=0.35, lift=size * 0.05, rot=np.pi / 8)
        put(p, _grad(base, u, 1.1, 1.45), 0.8 + 0.2 * u, 0.1 + 0.3 * u, n)
        d, r, dn = _disc_points(size * 0.13, 30, size * 0.12)
        put(d, YELLOW, 0.95, r * 0.1, dn)
    return tuple(np.concatenate(x) for x in (pos, rgb, lum, order, nrm))


def build_cloud(seed: int = 7, flowers: int = DEFAULT_FLOWERS, size: float = 1.0) -> Cloud:
    """Build the whole bouquet. The same seed always gives the same bouquet.
    `size` makes every flower bigger or smaller."""
    rng = np.random.default_rng(seed)
    parts = _Parts()
    bundle = np.array([0.0, -2.4, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    leaf_color = _hex(LEAVES[0])
    stem_color = np.array([0.30, 0.78, 0.45])
    order = rng.permutation(flowers)

    for i in range(flowers):
        # Heads are placed on a golden-angle spiral, tallest in the middle.
        psi = i * GOLDEN
        rho = 0.9 + 4.4 * np.sqrt((i + 0.5) / flowers)
        height = 5.2 - 0.4 * rho**1.45 + rng.uniform(-0.3, 0.3)
        head = np.array([rho * np.cos(psi), height, rho * np.sin(psi)])
        ctrl = bundle + (head - bundle) * np.array([0.15, 0.6, 0.15])

        stem_start = 0.02 + 0.15 * i / max(flowers - 1, 1)
        stem_dur = 0.22
        stem_end = stem_start + stem_dur

        # Stem: a curve, thickened with a couple of offset copies.
        s = np.linspace(0, 1, 60)
        curve = _bezier3(bundle, ctrl, head, s)
        thick = np.concatenate([curve, curve + (0.06, 0, 0), curve + (0, 0, 0.06)])
        parts.add(thick, bundle, stem_color, np.tile(0.55 + 0.35 * s, 3), stem_start, stem_dur)

        # Leaves: two per stem, drooping outward. They grow with the stem.
        outward = np.array([head[0], 0.0, head[2]])
        outward = outward / (np.linalg.norm(outward) + 1e-9)
        leaf_color = _hex(str(rng.choice(LEAVES)))
        for j in range(2):
            s_leaf = rng.uniform(0.35, 0.55) + 0.12 * j
            base_pt = _bezier3(bundle, ctrl, head, [s_leaf])[0]
            tangent = 2 * (1 - s_leaf) * (ctrl - bundle) + 2 * s_leaf * (head - ctrl)
            d = 0.55 * tangent / np.linalg.norm(tangent) + 0.85 * outward + 0.15 * up
            d = _rot_y(d, (0.9 if j else 0.0) * (1 if i % 2 else -1))
            d /= np.linalg.norm(d)
            ez = np.cross(d, up)
            ez /= np.linalg.norm(ez) + 1e-9
            basis = np.stack([d, np.cross(ez, d), ez], axis=1)
            length = 1.9 if j == 0 else 1.3
            p, u, _, n = _petal_points(1, length, length * 0.45, curl=-0.2)
            parts.add(p @ basis.T + base_pt, bundle, _grad(leaf_color, u, 0.65, 1.15), 0.5 + 0.45 * u,
                      stem_start, stem_dur, n @ basis.T)

        # Flower head, facing along the stem and tilted up a little.
        kind = KIND_CYCLE[i % len(KIND_CYCLE)]
        color = _hex(str(rng.choice(PALETTE[kind])))
        head_size = (BASE_SIZE[kind] + rng.uniform(-0.1, 0.1)) * size
        tip_dir = 2 * (head - ctrl)
        normal = tip_dir / np.linalg.norm(tip_dir) + 0.5 * up
        rot = _rotation_to(normal, rng.uniform(0, 2 * np.pi))
        local, rgb, lum, bloom_order, nrm = _flower_local(kind, head_size, color)
        bloom_start = max(stem_end + 0.02, 0.34 + 0.045 * order[i] * 7 / flowers)
        parts.add(local @ rot.T + head, head, rgb, lum, bloom_start + 0.12 * bloom_order, 0.14, nrm @ rot.T)

    # Filler: baby's breath (clouds of tiny white blossoms) fills gaps between the big flowers.
    for _ in range(16):
        psi, rho, top = rng.uniform(0, 2 * np.pi), rng.uniform(2.5, 7.2), rng.uniform(1.5, 5.2)
        tip = np.array([rho * np.cos(psi), top, rho * np.sin(psi)])
        ctrl = bundle + (tip - bundle) * np.array([0.15, 0.6, 0.15])
        start = 0.08 + 0.25 * rng.random()
        parts.add(_bezier3(bundle, ctrl, tip, np.linspace(0, 1, 36)), bundle, _hex("#7fcf9c"), 0.5, start, 0.2)
        blossoms = tip + rng.normal(0, 0.3, (36, 3))
        parts.add(blossoms, tip, np.clip(rng.uniform(0.88, 1.0, (36, 1)) * np.ones((36, 3)), 0, 1),
                  rng.uniform(0.7, 1.0, 36), start + 0.22, 0.12)

    # Filler: lavender, tall purple spikes.
    for _ in range(6):
        psi, rho, top = rng.uniform(0, 2 * np.pi), rng.uniform(1.0, 4.5), rng.uniform(4.0, 6.0)
        tip = np.array([rho * np.cos(psi), top, rho * np.sin(psi)])
        ctrl = bundle + (tip - bundle) * np.array([0.15, 0.6, 0.15])
        start = 0.1 + 0.2 * rng.random()
        parts.add(_bezier3(bundle, ctrl, tip, np.linspace(0, 1, 40)), bundle, _hex("#6fbf8a"), 0.5, start, 0.2)
        t = rng.uniform(0, 1, 70)
        spike = tip + np.stack([rng.normal(0, 0.1, 70), 1.7 * t, rng.normal(0, 0.1, 70)], -1)
        parts.add(spike, tip, _grad(_hex("#9b6ff0"), t, 0.8, 1.2), 0.65 + 0.3 * t, start + 0.2, 0.16)

    # Paper cone wrapping. It expands up from its tip first.
    apex_y, rim_y, rim_r = -7.6, -2.4, 3.3
    n = 7000
    h = np.sqrt(rng.uniform(0, 1, n))  # more points where the cone is wider
    theta = rng.uniform(0, 2 * np.pi, n)
    pts = np.stack([rim_r * h * np.cos(theta), apex_y + (rim_y - apex_y) * h, rim_r * h * np.sin(theta)], -1)
    kraft = np.array([0.95, 0.87, 0.72])
    ribbon = _hex(str(rng.choice(RIBBONS)))
    folds = 0.5 + 0.5 * np.sin(5 * theta + 3 * h)
    is_ribbon = (h > 0.5) & (h < 0.58)
    rgb = np.where(is_ribbon[:, None], ribbon, kraft)
    lum = np.clip(np.where(is_ribbon, 0.9, 0.5 + 0.3 * folds + 0.2 * (h > 0.97)) + rng.normal(0, 0.04, n), 0, 1)
    slope = rim_y - apex_y
    cone_normal = np.stack([np.cos(theta) * slope, np.full(n, -rim_r), np.sin(theta) * slope], -1)
    cone_normal /= np.linalg.norm(cone_normal, axis=1, keepdims=True)
    parts.add(pts, np.array([0.0, apex_y, 0.0]), rgb, lum, 0.0, 0.18, cone_normal)

    count = 110
    sparkle_rgb = np.array([_hex(c) for c in SPARKLE_COLORS])[rng.integers(0, len(SPARKLE_COLORS), count)]
    sparkles = np.column_stack(
        [
            rng.uniform(3.0, 9.0, count),
            rng.uniform(0, 2 * np.pi, count),
            rng.uniform(-7, 8, count),
            rng.uniform(0.3, 0.9, count),
            rng.uniform(0, 1, count),
            rng.uniform(0.3, 0.9, count),
            sparkle_rgb,
        ]
    )
    return parts.cloud(sparkles)


# ----------------------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------------------
@dataclass
class Frame:
    chars: np.ndarray  # (rows, cols) single characters
    rgb: np.ndarray  # (rows, cols, 3) uint8


def render(cloud: Cloud, time_s: float, cols: int, rows: int, build_seconds: float = BUILD_SECONDS,
           yaw: float | None = None, pitch: float | None = None, spin: float = 2 * np.pi / 8,
           zoom: float = 1.0, caption: str | None = None) -> Frame:
    """Draw the bouquet at `time_s` seconds onto a grid of `cols` x `rows` characters.
    The bouquet is scaled to fill the grid. A `caption` takes the bottom lines."""
    rows_eff = max(1, rows - 3) if caption else rows
    build = min(max(time_s / build_seconds, 0.0), 1.0)
    scale = _ease_out_back((build - cloud.t0) / cloud.dur)
    pos = cloud.anchor + (cloud.pos - cloud.anchor) * scale[:, None]
    visible = scale > 0.02
    rgb, lum = cloud.rgb, cloud.lum
    normal = np.zeros_like(pos) if cloud.normal is None else cloud.normal

    # Sparkles orbit the bouquet and drift upward, twinkling.
    radius, a0, y0, speed, phase, rise = cloud.sparkles.T[:6]
    angle = a0 + speed * time_s * 0.8
    height = ((y0 + 8 + rise * time_s) % 17) - 8
    sparkle_pos = np.stack([radius * np.cos(angle), height, radius * np.sin(angle)], -1)
    twinkle = 0.5 + 0.5 * np.sin(2 * np.pi * (time_s * speed * 0.7 + phase))
    sparkle_visible = (build > 0.5 + 0.25 * phase) & (twinkle > 0.25)
    sparkle_rgb = cloud.sparkles[:, 6:9] if cloud.sparkles.shape[1] >= 9 else np.tile([1.0, 0.95, 0.8], (len(radius), 1))

    pos = np.vstack([pos, sparkle_pos])
    visible = np.concatenate([visible, sparkle_visible])
    rgb = np.vstack([rgb, sparkle_rgb])
    lum = np.concatenate([lum, 0.12 + 0.45 * twinkle])  # dim, so they read as specks, not flowers
    normal = np.vstack([normal, np.zeros((len(radius), 3))])

    # Camera orbit: a steady spin with a gentle nodding tilt.
    yaw = 0.4 + spin * time_s if yaw is None else yaw
    pitch = 0.30 + 0.10 * np.sin(0.7 * time_s) if pitch is None else pitch
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rot = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    q = pos @ rot.T

    # Lighting: surfaces facing the light are bright. Petals are lit from both sides.
    lit = np.abs(normal @ rot.T @ LIGHT)
    has_normal = np.abs(normal).sum(axis=1) > 0.5
    shade = np.where(has_normal, 0.25 + 0.95 * lit, 0.8)

    # Fit the bouquet to the grid. Terminal cells are about twice as tall as wide, hence the 2.
    extent_r, extent_h, extent_y = cloud.extent
    near = CAMERA_DISTANCE / (CAMERA_DISTANCE - extent_r)  # how much bigger the closest parts look
    unit = min(
        rows_eff * 0.95 / ((0.95 * extent_h + 0.4 * extent_r) * near**0.5),
        cols * 0.97 / (4 * extent_r * near**0.7),
    ) * zoom
    depth = CAMERA_DISTANCE - q[:, 2]
    persp = CAMERA_DISTANCE / np.maximum(depth, 1e-3)
    ix = np.floor(cols / 2 + 2 * unit * persp * q[:, 0]).astype(np.int64)
    iy = np.floor(rows_eff / 2 - unit * persp * (q[:, 1] - extent_y)).astype(np.int64)
    ok = visible & (depth > 1) & (ix >= 0) & (ix < cols) & (iy >= 0) & (iy < rows_eff)

    # Depth buffer: write far-to-near so the nearest point in each cell wins.
    idx = np.flatnonzero(ok)
    idx = idx[np.argsort(q[idx, 2])]
    buf = np.full((rows_eff, cols), -1, dtype=np.int64)
    buf[iy[idx], ix[idx]] = idx

    chars = np.full((rows, cols), " ")
    image = np.zeros((rows, cols, 3), dtype=np.uint8)
    filled = buf >= 0
    sel = buf[filled]
    nearness = np.clip(0.5 + 0.5 * (q[sel, 2] + 7) / 14, 0.3, 1.0)  # far things fade
    brightness = np.clip(lum[sel] * shade[sel] * nearness * 1.1, 0, 1)
    colour = rgb[sel] * (0.55 + 0.45 * nearness[:, None]) * (0.7 + 0.3 * np.minimum(shade[sel], 1.1)[:, None])
    chars[:rows_eff][filled] = np.array(list(RAMP))[1 + (brightness * 8.99).astype(int)]
    image[:rows_eff][filled] = np.clip(colour * 255, 0, 255).astype(np.uint8)

    if caption:
        text = caption[:cols]
        row, start = rows - 2 if rows >= 3 else rows - 1, (cols - len(text)) // 2
        for k, ch in enumerate(text):
            chars[row, start + k] = ch
            image[row, start + k] = CAPTION_COLOR
    return Frame(chars, image)


# ----------------------------------------------------------------------------------------
# Output: terminal text and GIF images
# ----------------------------------------------------------------------------------------
_CUBE = np.array([0, 95, 135, 175, 215, 255])


@functools.lru_cache(maxsize=1 << 16)
def xterm256(rgb) -> int:
    """The closest of the 256 colours every terminal supports: a 6x6x6 colour cube
    plus 24 grays. Picks whichever is nearer to the real colour."""
    c = np.array(rgb, dtype=float)
    cube_idx = np.abs(_CUBE[None, :] - c[:, None]).argmin(axis=1)
    cube_err = ((_CUBE[cube_idx] - c) ** 2).sum()
    cube = 16 + 36 * int(cube_idx[0]) + 6 * int(cube_idx[1]) + int(cube_idx[2])
    gray_level = min(23, max(0, int(round((c.mean() - 8) / 10))))
    gray_err = ((8 + 10 * gray_level - c) ** 2).sum()
    return 232 + gray_level if gray_err < cube_err else cube


def supports_truecolor(env=None) -> bool:
    """Terminals like iTerm2 and VS Code announce 24-bit colour support. macOS Terminal does not."""
    env = os.environ if env is None else env
    return env.get("COLORTERM", "").lower() in ("truecolor", "24bit")


def frame_to_ansi(frame: Frame, truecolor: bool = False) -> str:
    rows, cols = frame.chars.shape
    lines = []
    for r in range(rows):
        out, last = [], None
        for c in range(cols):
            ch = frame.chars[r, c]
            if ch == " ":
                out.append(" ")
                continue
            color = tuple(int(v) for v in frame.rgb[r, c])
            key = color if truecolor else xterm256(color)
            if key != last:
                out.append(f"\x1b[38;2;{color[0]};{color[1]};{color[2]}m" if truecolor else f"\x1b[38;5;{key}m")
                last = key
            out.append(ch)
        out.append("\x1b[0m")
        lines.append("".join(out))
    return "\n".join(lines)


@functools.lru_cache(maxsize=2)
def _font(size: int):
    from matplotlib import font_manager
    from PIL import ImageFont

    return ImageFont.truetype(font_manager.findfont("DejaVu Sans Mono"), size)


def frame_to_image(frame: Frame, cell_w: int = 8, cell_h: int = 16, font_size: int = 13):
    from PIL import Image, ImageDraw

    rows, cols = frame.chars.shape
    image = Image.new("RGB", (cols * cell_w, rows * cell_h), BG)
    draw = ImageDraw.Draw(image)
    font = _font(font_size)
    for r, c in zip(*np.nonzero(frame.chars != " ")):
        draw.text((c * cell_w, r * cell_h), frame.chars[r, c], font=font,
                  fill=tuple(int(v) for v in frame.rgb[r, c]))
    return image


def save_gif(cloud: Cloud, path, cols: int, rows: int, seconds: float, fps: int,
             caption: str | None = None, zoom: float = 1.0) -> None:
    frames = [frame_to_image(render(cloud, i / fps, cols, rows, caption=caption, zoom=zoom))
              for i in range(int(seconds * fps))]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)


def run_live(cloud: Cloud, fps: int = 20, truecolor: bool = False, cols=None, rows=None,
             out=None, max_frames: int | None = None, caption: str | None = None, zoom: float = 1.0) -> None:
    """Animate in the terminal until Ctrl+C."""
    out = out or sys.stdout
    size = shutil.get_terminal_size((100, 40))
    cols, rows = cols or size.columns, rows or size.lines - 1  # leave the last line free
    out.write("\x1b[?1049h\x1b[?25l\x1b[2J")  # alternate screen, hide the cursor
    start, shown = time.perf_counter(), 0
    try:
        while max_frames is None or shown < max_frames:
            frame = render(cloud, time.perf_counter() - start, cols, rows, caption=caption, zoom=zoom)
            out.write("\x1b[H" + frame_to_ansi(frame, truecolor))
            out.flush()
            shown += 1
            delay = start + shown / fps - time.perf_counter()  # wait for this frame's slot
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        out.write("\x1b[0m\x1b[?25h\x1b[?1049l")  # restore the terminal
        out.flush()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="A 3D bouquet drawn with coloured ASCII characters.")
    parser.add_argument("--gif", help="write a GIF to this path instead of animating in the terminal")
    parser.add_argument("--seed", type=int, default=7, help="different layout and colours (default 7)")
    parser.add_argument("--flowers", type=int, default=DEFAULT_FLOWERS, help=f"number of flowers (default {DEFAULT_FLOWERS})")
    parser.add_argument("--size", type=float, default=1.0, help="flower size multiplier (default 1.0)")
    parser.add_argument("--zoom", type=float, default=1.0, help="zoom the whole bouquet in or out (default 1.0)")
    parser.add_argument("--message", help="text shown under the bouquet, e.g. --message \"for you\"")
    parser.add_argument("--cols", type=int, help="width in characters (default: terminal width; GIF 100)")
    parser.add_argument("--rows", type=int, help="height in characters (default: terminal height; GIF 56)")
    parser.add_argument("--fps", type=int, help="frames per second (default 20 live, 15 GIF)")
    parser.add_argument("--seconds", type=float, default=8.0, help="GIF length in seconds (default 8)")
    parser.add_argument("--color", choices=["auto", "256", "truecolor"], default="auto",
                        help="auto uses 24-bit colour only where the terminal says it supports it")
    args = parser.parse_args(argv)

    cloud = build_cloud(args.seed, args.flowers, args.size)
    if args.gif:
        save_gif(cloud, args.gif, args.cols or 100, args.rows or 56, args.seconds, args.fps or 15,
                 args.message, args.zoom)
        print(f"Saved {args.gif}")
    else:
        if not sys.stdout.isatty():
            sys.exit("This needs a real terminal. To make a file instead, use --gif out.gif")
        truecolor = args.color == "truecolor" or (args.color == "auto" and supports_truecolor())
        run_live(cloud, args.fps or 20, truecolor, args.cols, args.rows, caption=args.message, zoom=args.zoom)


if __name__ == "__main__":
    main()

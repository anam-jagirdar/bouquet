"""The bouquet itself: layout, timeline and per-frame drawing.

Everything is a pure function of time `t` between 0 and 1, so any frame can be
rendered on its own. That makes it easy to test and to export as a GIF.

Timeline (fractions of the whole animation):
    0.00 - 0.35  stems grow up out of the wrapping, leaves unfurl
    0.36 - 0.82  flowers bloom one after another
    0.60 - 1.00  the bouquet sways, sparkles drift up
"""
from dataclasses import dataclass

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Polygon

from .flowers import DRAWERS, shade
from .geometry import (
    bezier,
    bezier_angle,
    bezier_point,
    ease_out_back,
    ease_out_cubic,
    petal,
    rotate,
    segment,
)

XLIM = (-8.0, 8.0)
YLIM = (-9.5, 8.5)
FIGSIZE = (6.0, 6.75)  # same ratio as the axes limits, so nothing is squashed

BG_TOP = "#1b1a3a"
BG_BOTTOM = "#54406f"
STEM = "#4fae78"
LEAF_FACE, LEAF_EDGE = "#43a56b", "#2d7a4c"
PAPER, PAPER_EDGE = "#f1dfbd", "#c9b38a"
RIBBON = "#e63e6d"

PALETTE = {
    "rose": ["#e63e6d", "#ff6f91", "#f58fa9"],
    "cosmos": ["#b794f6", "#ff8a65", "#ffb08a"],
    "daisy": ["#fffaf0"],
}
KIND_CYCLE = ["cosmos", "rose", "daisy", "rose"]
BASE_SIZE = {"rose": 1.75, "cosmos": 1.6, "daisy": 1.45}

STEM_GROW = 0.20
BLOOM_TIME = 0.22


@dataclass(frozen=True)
class Flower:
    kind: str
    color: str
    size: float
    head: tuple
    ctrl: tuple
    phase: float
    stem_start: float
    bloom_start: float
    spin: float
    leaf_side: int  # +1 leaf points left of the stem, -1 points right
    leaf_at: float  # how far up the stem (0 to 1) the leaf grows


@dataclass(frozen=True, eq=False)
class Scene:
    flowers: tuple
    sparkles: np.ndarray  # columns: x, y, size, phase, speed
    bundle: tuple


def build_scene(seed: int = 7, count: int = 7, scale: float = 1.0) -> Scene:
    """Fan the flowers out from one point. The same seed always gives the same bouquet.

    `scale` makes every flower bigger or smaller. Flowers are kept inside the frame.
    """
    rng = np.random.default_rng(seed)
    bundle = (0.0, -4.2)
    spread = 52 if count <= 8 else 58
    angles = np.linspace(-spread, spread, count) + rng.uniform(-3, 3, count)
    order = rng.permutation(count)
    # Staggered heights so neighbours overlap like a real bouquet. Bigger bunches get a third row.
    row_offsets = (-0.9, 1.1) if count <= 8 else (-1.2, 1.3, 0.1)

    flowers = []
    for i, deg in enumerate(angles):
        rad = np.radians(deg)
        kind = KIND_CYCLE[i % len(KIND_CYCLE)]
        size = float((BASE_SIZE[kind] + rng.uniform(-0.1, 0.1)) * scale)
        reach = 8.6 - 0.035 * abs(deg) + row_offsets[i % len(row_offsets)] + rng.uniform(-0.3, 0.3)
        margin = 0.4 + 1.05 * size  # keep the whole flower head inside the picture
        hx = float(np.clip(bundle[0] + reach * np.sin(rad), XLIM[0] + margin, XLIM[1] - margin))
        hy = float(min(bundle[1] + reach * np.cos(rad), YLIM[1] - margin))
        head = (hx, hy)
        ctrl = (bundle[0] + (hx - bundle[0]) * 0.12, bundle[1] + (hy - bundle[1]) * 0.62)
        # Leaves splay outward, away from the middle, so they never poke into a neighbouring stem.
        if abs(hx - bundle[0]) < 0.5:
            leaf_side = 1 if i % 2 else -1
        else:
            leaf_side = -1 if hx > bundle[0] else 1
        flowers.append(
            Flower(
                kind=kind,
                color=str(rng.choice(PALETTE[kind])),
                size=size,
                head=head,
                ctrl=ctrl,
                phase=float(rng.uniform(0, 2 * np.pi)),
                stem_start=0.02 + 0.02 * i * (7 / max(count, 7)),
                bloom_start=0.36 + 0.04 * int(order[i]) * (7 / max(count, 7)),
                spin=float(rng.uniform(0, 2 * np.pi)),
                leaf_side=leaf_side,
                leaf_at=float(rng.uniform(0.36, 0.52)),
            )
        )

    n = 30
    sparkles = np.column_stack(
        [
            rng.uniform(-7.5, 7.5, n),
            rng.uniform(-9, 9, n),
            rng.uniform(8, 26, n),
            rng.uniform(0, 1, n),
            rng.uniform(0.6, 1.6, n),
        ]
    )
    return Scene(tuple(flowers), sparkles, bundle)


def make_axes(fig):
    """A static gradient background plus a transparent axes that gets redrawn every frame."""
    fig.set_facecolor(BG_BOTTOM)
    bg = fig.add_axes([0, 0, 1, 1])
    bg.axis("off")
    grad = np.linspace(0, 1, 256)[:, None]
    top, bottom = np.array(to_rgb(BG_TOP)), np.array(to_rgb(BG_BOTTOM))
    bg.imshow((top * (1 - grad) + bottom * grad).reshape(256, 1, 3), aspect="auto", extent=(0, 1, 0, 1))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.patch.set_alpha(0)
    return ax


def _cone_half_width(y: float) -> float:
    return 3.2 * (y + 8.6) / 6.8


def _draw_wrap(ax):
    """Paper cone, ribbon and bow covering the bottom of the stems."""
    cone = [(-3.2, -1.8), (0.0, -2.6), (3.2, -1.8), (0.0, -8.6)]
    ax.add_patch(Polygon(cone, closed=True, facecolor=PAPER, edgecolor=PAPER_EDGE, linewidth=1.0, zorder=10))
    fold = [(-3.2, -1.8), (0.0, -2.6), (0.2, -8.0), (-0.6, -6.9)]  # stays inside the cone's edges
    ax.add_patch(Polygon(fold, closed=True, facecolor=shade(PAPER, 1.25), edgecolor="none", alpha=0.7, zorder=10.1))

    y1, y2 = -4.6, -5.4
    band = [(-_cone_half_width(y1), y1), (_cone_half_width(y1), y1),
            (_cone_half_width(y2), y2), (-_cone_half_width(y2), y2)]
    ax.add_patch(Polygon(band, closed=True, facecolor=RIBBON, edgecolor=shade(RIBBON, 0.65), linewidth=0.8, zorder=11))

    for angle in (np.radians(150), np.radians(30)):
        loop = rotate(petal(1.7, 1.0), angle) + (0.0, -5.0)
        ax.add_patch(Polygon(loop, closed=True, facecolor=shade(RIBBON, 1.1), edgecolor=shade(RIBBON, 0.6),
                             linewidth=0.8, zorder=11.2))
    ax.add_patch(Circle((0.0, -5.0), 0.28, facecolor=shade(RIBBON, 0.85), edgecolor=shade(RIBBON, 0.55),
                        linewidth=0.8, zorder=11.4))


def draw_frame(ax, scene: Scene, t: float) -> None:
    """Render the bouquet at time `t` (0 to 1) onto `ax`."""
    ax.cla()
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("none")

    sway_gain = 0.12 + 0.88 * segment(t, 0.6, 0.85)
    bundle = np.array(scene.bundle)

    for f in scene.flowers:
        g = ease_out_cubic(segment(t, f.stem_start, f.stem_start + STEM_GROW))
        if g <= 0:
            continue

        dx = sway_gain * 0.3 * np.sin(2 * np.pi * 2 * t + f.phase)
        head = np.array(f.head) + (dx, 0.0)
        ctrl = np.array(f.ctrl) + (dx * 0.4, 0.0)

        curve = bezier(bundle, ctrl, head, upto=g)
        ax.plot(curve[:, 0], curve[:, 1], color=STEM, linewidth=2.6, solid_capstyle="round", zorder=2)

        if g >= f.leaf_at:
            grow = ease_out_cubic(segment(g, f.leaf_at + 0.03, f.leaf_at + 0.4))
            if grow > 0:
                base = bezier_point(bundle, ctrl, head, f.leaf_at)
                angle = bezier_angle(bundle, ctrl, head, f.leaf_at) + f.leaf_side * np.radians(38)
                leaf = rotate(petal(1.9 * grow, 0.8 * grow), angle) + base
                ax.add_patch(Polygon(leaf, closed=True, facecolor=LEAF_FACE, edgecolor=LEAF_EDGE,
                                     linewidth=0.6, zorder=2.5))

        tip = curve[-1]
        z = 4 + (10 - tip[1]) * 0.01  # lower flowers sit in front of higher ones
        ax.add_patch(Circle(tip, 0.3 * f.size * g, facecolor=shade(f.color, 0.8), edgecolor="none", zorder=z))
        openness = ease_out_back(segment(t, f.bloom_start, f.bloom_start + BLOOM_TIME))
        DRAWERS[f.kind](ax, tip[0], tip[1], f.size, openness, f.spin + dx * 0.3, f.color, z)

    _draw_wrap(ax)

    fade = segment(t, 0.55, 0.8)
    if fade > 0:
        x, y0, size, phase, speed = scene.sparkles.T
        y = ((y0 + 9 + speed * t * 6) % 18) - 9
        twinkle = 0.5 + 0.5 * np.sin(2 * np.pi * (2 * t * speed + phase))
        alpha = fade * (0.25 + 0.6 * twinkle)
        colors = np.column_stack([np.ones_like(x), np.full_like(x, 0.93), np.full_like(x, 0.75), alpha])
        ax.scatter(x, y, s=size, c=colors, linewidths=0, zorder=20)

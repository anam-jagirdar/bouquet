"""Drawing functions for each flower type.

Every drawer has the same signature, so the scene can pick one by name:
    draw(ax, cx, cy, size, openness, angle, color, z)

`openness` runs from 0 (closed bud) to about 1 (fully open). It can briefly
exceed 1 during the bloom animation, which makes the flower pop open.
"""
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Polygon

from .geometry import petal, rotate

MIN_OPENNESS = 0.03


def shade(color, factor: float):
    """factor < 1 darkens the colour, factor > 1 lightens it toward white."""
    r, g, b = to_rgb(color)
    if factor <= 1:
        return (r * factor, g * factor, b * factor)
    t = min(factor - 1, 1.0)
    return (r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t)


def _petal_patch(cx, cy, length, width, angle, face, edge, z):
    points = rotate(petal(length, width), angle) + (cx, cy)
    return Polygon(points, closed=True, facecolor=face, edgecolor=edge, linewidth=0.5, zorder=z)


def draw_rose(ax, cx, cy, size, openness, angle, color, z):
    """Layered petals, lighter on the outside and darker toward the centre."""
    if openness < MIN_OPENNESS:
        return
    layers = 5
    for i in range(layers):
        frac = i / (layers - 1)
        count = 8 - i
        length = size * (1 - 0.17 * i) * openness
        face = shade(color, 1.18 - 0.5 * frac)
        edge = shade(color, 0.6)
        offset = angle + i * np.pi / count * 0.5
        for k in range(count):
            ax.add_patch(_petal_patch(cx, cy, length, length * 0.8, offset + 2 * np.pi * k / count, face, edge, z))
    ax.add_patch(Circle((cx, cy), 0.12 * size * openness, facecolor=shade(color, 0.55), edgecolor="none", zorder=z))


def draw_daisy(ax, cx, cy, size, openness, angle, color, z):
    """Many narrow petals around a golden centre."""
    if openness < MIN_OPENNESS:
        return
    count = 16
    face, edge = shade(color, 1.0), "#cfc7b6"
    for k in range(count):
        ax.add_patch(
            _petal_patch(cx, cy, size * openness, size * 0.24, angle + 2 * np.pi * k / count, face, edge, z)
        )
    ax.add_patch(
        Circle((cx, cy), size * 0.26 * min(1.0, openness * 1.2), facecolor="#ffc83d", edgecolor="#e0a21b",
               linewidth=0.5, zorder=z)
    )


def draw_cosmos(ax, cx, cy, size, openness, angle, color, z):
    """Eight broad petals with a lighter inner ring."""
    if openness < MIN_OPENNESS:
        return
    count = 8
    outer_face, edge = shade(color, 1.0), shade(color, 0.65)
    inner_face = shade(color, 1.3)
    for k in range(count):
        ax.add_patch(
            _petal_patch(cx, cy, size * openness, size * 0.62, angle + 2 * np.pi * k / count, outer_face, edge, z)
        )
    for k in range(count):
        ax.add_patch(
            _petal_patch(cx, cy, size * 0.68 * openness, size * 0.4,
                         angle + np.pi / count + 2 * np.pi * k / count, inner_face, edge, z)
        )
    ax.add_patch(
        Circle((cx, cy), size * 0.13 * openness, facecolor="#ffc83d", edgecolor="#e0a21b", linewidth=0.5, zorder=z)
    )


DRAWERS = {"rose": draw_rose, "daisy": draw_daisy, "cosmos": draw_cosmos}

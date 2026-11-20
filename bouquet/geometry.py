"""Small math helpers: easing curves, Bezier curves and petal shapes."""
import numpy as np


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def segment(t: float, start: float, end: float) -> float:
    """Progress (0 to 1) of `t` through the window [start, end]."""
    return clamp01((t - start) / (end - start))


def ease_out_cubic(t: float) -> float:
    """Fast start, gentle stop. Good for things growing into place."""
    t = clamp01(t)
    return 1 - (1 - t) ** 3


def ease_out_back(t: float) -> float:
    """Overshoots 1 slightly before settling, which gives blooms a soft 'pop'."""
    c1 = 1.70158
    c3 = c1 + 1
    t = clamp01(t)
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def bezier(p0, p1, p2, upto: float = 1.0, n: int = 40) -> np.ndarray:
    """Points along a quadratic Bezier curve, from the start up to fraction `upto`."""
    s = np.linspace(0.0, clamp01(upto), n)[:, None]
    p0, p1, p2 = (np.asarray(p, dtype=float) for p in (p0, p1, p2))
    return (1 - s) ** 2 * p0 + 2 * (1 - s) * s * p1 + s**2 * p2


def bezier_point(p0, p1, p2, s: float) -> np.ndarray:
    return bezier(p0, p1, p2, upto=s, n=2)[-1]


def bezier_angle(p0, p1, p2, s: float) -> float:
    """Direction (radians) the curve is heading at parameter `s`."""
    p0, p1, p2 = (np.asarray(p, dtype=float) for p in (p0, p1, p2))
    tangent = 2 * (1 - s) * (p1 - p0) + 2 * s * (p2 - p1)
    return float(np.arctan2(tangent[1], tangent[0]))


def rotate(points, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(points, dtype=float) @ np.array([[c, -s], [s, c]]).T


def petal(length: float, width: float, n: int = 18) -> np.ndarray:
    """A teardrop-shaped petal polygon. It starts at the origin and points along +x."""
    s = np.linspace(0.0, 1.0, n)
    half = 0.5 * width * np.sin(np.pi * s**0.7)
    upper = np.column_stack([s * length, half])
    lower = np.column_stack([s[::-1] * length, -half[::-1]])
    return np.vstack([upper, lower[1:-1]])

import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # tests never open a window

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from bouquet.cli import main
from bouquet.geometry import bezier, ease_out_back, ease_out_cubic, petal, rotate, segment
from bouquet.scene import FIGSIZE, XLIM, YLIM, build_scene, draw_frame, make_axes


def new_axes():
    fig = Figure(figsize=FIGSIZE)
    FigureCanvasAgg(fig)
    return fig, make_axes(fig)


class GeometryTests(unittest.TestCase):
    def test_bezier_hits_its_endpoints(self):
        curve = bezier((0, 0), (1, 3), (4, 0))
        np.testing.assert_allclose(curve[0], (0, 0))
        np.testing.assert_allclose(curve[-1], (4, 0))

    def test_bezier_partial_stops_early(self):
        full = bezier((0, 0), (0, 4), (0, 8))
        half = bezier((0, 0), (0, 4), (0, 8), upto=0.5)
        self.assertAlmostEqual(half[-1][1], 4.0)  # a straight line, so halfway is y = 4
        self.assertLess(half[-1][1], full[-1][1])

    def test_easing_stays_in_range_and_hits_the_ends(self):
        self.assertEqual(ease_out_cubic(0), 0)
        self.assertEqual(ease_out_cubic(1), 1)
        self.assertEqual(ease_out_cubic(5), 1)  # clamped
        self.assertAlmostEqual(ease_out_back(0), 0)
        self.assertAlmostEqual(ease_out_back(1), 1)
        self.assertGreater(max(ease_out_back(x / 100) for x in range(101)), 1.0)  # overshoots

    def test_segment_progress(self):
        self.assertEqual(segment(0.1, 0.2, 0.4), 0)
        self.assertAlmostEqual(segment(0.3, 0.2, 0.4), 0.5)
        self.assertEqual(segment(0.9, 0.2, 0.4), 1)

    def test_petal_shape(self):
        shape = petal(2.0, 0.8)
        np.testing.assert_allclose(shape[0], (0, 0))
        self.assertAlmostEqual(shape[:, 0].max(), 2.0)
        self.assertLessEqual(np.abs(shape[:, 1]).max(), 0.4 + 1e-9)

    def test_rotate_quarter_turn(self):
        np.testing.assert_allclose(rotate([(1, 0)], np.pi / 2), [(0, 1)], atol=1e-12)


class SceneTests(unittest.TestCase):
    def test_same_seed_gives_same_bouquet(self):
        a, b = build_scene(seed=3), build_scene(seed=3)
        self.assertEqual([f.head for f in a.flowers], [f.head for f in b.flowers])
        self.assertEqual([f.color for f in a.flowers], [f.color for f in b.flowers])

    def test_different_seeds_differ(self):
        a, b = build_scene(seed=1), build_scene(seed=2)
        self.assertNotEqual([f.head for f in a.flowers], [f.head for f in b.flowers])

    def test_flower_count_and_variety(self):
        scene = build_scene(count=7)
        self.assertEqual(len(scene.flowers), 7)
        self.assertEqual({f.kind for f in scene.flowers}, {"rose", "daisy", "cosmos"})

    def test_leaves_point_away_from_the_middle(self):
        for count in (7, 11):
            scene = build_scene(count=count)
            for f in scene.flowers:
                if abs(f.head[0] - scene.bundle[0]) >= 0.5:
                    expected = -1 if f.head[0] > scene.bundle[0] else 1  # +1 points left, -1 points right
                    self.assertEqual(f.leaf_side, expected)

    def test_size_option_scales_the_flowers(self):
        small = np.mean([f.size for f in build_scene(scale=1.0).flowers])
        big = np.mean([f.size for f in build_scene(scale=1.5).flowers])
        self.assertAlmostEqual(big / small, 1.5, places=1)

    def test_flowers_stay_inside_the_frame(self):
        for count in (7, 11, 14):
            for scale in (1.0, 1.5, 2.0):
                for f in build_scene(count=count, scale=scale).flowers:
                    reach = 1.05 * f.size  # petals extend about one size from the centre
                    self.assertGreaterEqual(f.head[0] - reach, XLIM[0], (count, scale))
                    self.assertLessEqual(f.head[0] + reach, XLIM[1], (count, scale))
                    self.assertLessEqual(f.head[1] + reach, YLIM[1], (count, scale))

    def test_everything_blooms_before_the_end(self):
        for count in (7, 14):
            for f in build_scene(count=count).flowers:
                self.assertLess(f.bloom_start + 0.22, 1.0)
                self.assertGreaterEqual(f.bloom_start + 1e-9, f.stem_start + 0.2)

    def test_everything_blooms_before_the_end_default(self):
        for f in build_scene().flowers:
            self.assertLess(f.bloom_start + 0.22, 1.0)
            self.assertGreaterEqual(f.bloom_start, f.stem_start + 0.2)  # bloom only after the stem is up

    def test_bouquet_fills_in_over_time(self):
        scene = build_scene()
        _, ax = new_axes()
        counts = []
        for t in (0.0, 0.3, 0.6, 0.9):
            draw_frame(ax, scene, t)
            counts.append(len(ax.patches))
        self.assertEqual(counts, sorted(counts))
        self.assertGreater(counts[-1], counts[0] * 5)

    def test_every_frame_renders(self):
        scene = build_scene()
        fig, ax = new_axes()
        for i in range(0, 30):
            draw_frame(ax, scene, i / 30)
        fig.canvas.draw()  # raises if any frame produced invalid geometry (NaN, bad colour)


class CliTests(unittest.TestCase):
    def test_saves_a_gif(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "b.gif"
            main(["--save", str(out), "--frames", "4", "--fps", "10"])
            self.assertTrue(out.read_bytes().startswith(b"GIF8"))


if __name__ == "__main__":
    unittest.main()

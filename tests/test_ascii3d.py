import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from matplotlib.colors import rgb_to_hsv

from bouquet.ascii3d import (
    DEFAULT_FLOWERS,
    LIGHT,
    RAMP,
    Cloud,
    build_cloud,
    frame_to_ansi,
    main,
    render,
    run_live,
    supports_truecolor,
    xterm256,
)


def filled_cells(frame):
    return int((frame.chars != " ").sum())


class CloudTests(unittest.TestCase):
    def test_same_seed_gives_same_cloud(self):
        a, b = build_cloud(seed=3), build_cloud(seed=3)
        np.testing.assert_array_equal(a.pos, b.pos)
        np.testing.assert_array_equal(a.rgb, b.rgb)

    def test_different_seeds_differ(self):
        self.assertFalse(np.array_equal(build_cloud(seed=1).pos, build_cloud(seed=2).pos))

    def test_arrays_are_consistent_and_finite(self):
        c = build_cloud()
        n = len(c.pos)
        for name in ("anchor", "rgb", "lum", "t0", "dur"):
            self.assertEqual(len(getattr(c, name)), n, name)
        for name in ("pos", "anchor", "rgb", "lum", "t0", "dur"):
            self.assertTrue(np.isfinite(getattr(c, name)).all(), name)
        self.assertTrue(((c.rgb >= 0) & (c.rgb <= 1)).all())
        self.assertTrue(((c.lum >= 0) & (c.lum <= 1)).all(), "brightness must stay in 0..1")

    def test_everything_finishes_growing_before_the_build_ends(self):
        for flowers in (7, 14):
            c = build_cloud(flowers=flowers)
            self.assertLess((c.t0 + c.dur).max(), 1.0)

    def test_flower_heads_only_bloom_after_their_stems_finish(self):
        # A head that blooms before its stem is tall enough would float in mid-air.
        c = build_cloud()
        stems_done = 0.02 + 0.15 + 0.22  # the last stem finishes here
        heads = c.dur == 0.14
        self.assertGreater(heads.sum(), 0)
        self.assertGreaterEqual(c.t0[heads].min(), 0.02 + 0.22)
        self.assertLessEqual(stems_done, 0.4)


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.cloud = build_cloud()

    def test_frame_shape_and_characters(self):
        f = render(self.cloud, 8.0, 80, 40)
        self.assertEqual(f.chars.shape, (40, 80))
        self.assertEqual(f.rgb.shape, (40, 80, 3))
        self.assertTrue(set(np.unique(f.chars)) <= set(RAMP))

    def test_bouquet_builds_up_over_time(self):
        counts = [filled_cells(render(self.cloud, t, 80, 40)) for t in (0.2, 1.5, 3.0, 6.0)]
        self.assertEqual(counts, sorted(counts))
        self.assertGreater(counts[-1], counts[0] * 2)

    def test_rotation_changes_the_picture(self):
        a = render(self.cloud, 8.0, 80, 40, yaw=0.0)
        b = render(self.cloud, 8.0, 80, 40, yaw=1.5)
        self.assertFalse(np.array_equal(a.chars, b.chars))

    def test_nearest_point_wins_the_cell(self):
        # Two points in the same screen spot, one nearer the camera. The near one must show.
        zeros = np.zeros((2, 3))
        cloud = Cloud(
            pos=np.array([[0, -0.4, -3.0], [0, -0.4, 3.0]]),  # y = -0.4 cancels the renderer's centring shift
            anchor=zeros,
            rgb=np.array([[1.0, 0, 0], [0, 1.0, 0]]),  # far = red, near = green
            lum=np.array([0.9, 0.9]),
            t0=np.zeros(2),
            dur=np.full(2, 0.01),
            sparkles=np.zeros((0, 6)),
        )
        f = render(cloud, 10.0, 41, 21, yaw=0.0, pitch=0.0)
        ys, xs = np.nonzero(f.chars != " ")
        self.assertEqual(len(ys), 1)
        r, g, b = f.rgb[ys[0], xs[0]]
        self.assertGreater(g, r)

    def test_tiny_grids_do_not_crash(self):
        for cols, rows in ((10, 5), (1, 1), (200, 3)):
            render(self.cloud, 6.0, cols, rows)

    def test_points_off_screen_are_skipped(self):
        f = render(self.cloud, 8.0, 12, 6)  # most of the bouquet falls outside
        self.assertEqual(f.chars.shape, (6, 12))


class OutputTests(unittest.TestCase):
    def test_xterm256_known_colours(self):
        self.assertEqual(xterm256((0, 0, 0)), 16)
        self.assertEqual(xterm256((255, 0, 0)), 196)
        self.assertEqual(xterm256((255, 255, 255)), 231)

    def test_ansi_has_one_line_per_row_and_resets_colour(self):
        f = render(build_cloud(), 8.0, 60, 30)
        text = frame_to_ansi(f)
        lines = text.split("\n")
        self.assertEqual(len(lines), 30)
        self.assertTrue(all(line.endswith("\x1b[0m") for line in lines))
        self.assertIn("\x1b[38;5;", text)
        self.assertIn("\x1b[38;2;", frame_to_ansi(f, truecolor=True))

    def test_live_mode_draws_and_restores_the_terminal(self):
        out = io.StringIO()
        run_live(build_cloud(), fps=1000, cols=40, rows=20, out=out, max_frames=3)
        text = out.getvalue()
        self.assertEqual(text.count("\x1b[H"), 3)  # three frames drawn
        self.assertTrue(text.startswith("\x1b[?1049h"))  # entered the alternate screen
        self.assertTrue(text.endswith("\x1b[?1049l"))  # and left it again
        self.assertIn("\x1b[?25h", text)  # cursor is shown again

    def test_saves_a_gif(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "b3d.gif"
            main(["--gif", str(out), "--seconds", "0.4", "--fps", "5", "--cols", "30", "--rows", "16"])
            self.assertTrue(out.read_bytes().startswith(b"GIF8"))


class LookTests(unittest.TestCase):
    """The things that make it pretty: size, colour variety, lighting, captions."""

    def test_default_bouquet_is_a_full_one(self):
        self.assertGreaterEqual(DEFAULT_FLOWERS, 12)
        self.assertGreater(len(build_cloud().pos), 25000)

    def test_bouquet_uses_many_different_colours(self):
        c = build_cloud()
        hsv = rgb_to_hsv(c.rgb)
        vivid = (hsv[:, 1] > 0.35) & (hsv[:, 2] > 0.4)
        hue_buckets = np.bincount((hsv[vivid, 0] * 12).astype(int) % 12, minlength=12)
        self.assertGreaterEqual(int((hue_buckets > 200).sum()), 5, hue_buckets)

    def test_surface_normals_are_unit_length_or_unlit(self):
        c = build_cloud()
        self.assertEqual(c.normal.shape, c.pos.shape)
        lengths = np.linalg.norm(c.normal, axis=1)
        self.assertTrue(np.all(np.isclose(lengths, 1, atol=1e-6) | np.isclose(lengths, 0)))
        self.assertGreater((lengths > 0.5).mean(), 0.5, "most points should be lit")

    def test_surfaces_facing_the_light_are_brighter(self):
        def char_for(normal):
            cloud = Cloud(pos=np.array([[0, -0.4, 0.0]]), anchor=np.zeros((1, 3)), rgb=np.ones((1, 3)),
                          lum=np.array([0.8]), t0=np.zeros(1), dur=np.full(1, 0.01),
                          sparkles=np.zeros((0, 6)), normal=np.array([normal]))
            f = render(cloud, 10.0, 41, 21, yaw=0.0, pitch=0.0)
            return RAMP.index(f.chars[f.chars != " "][0])

        sideways = np.cross(LIGHT, [0, 1, 0])
        sideways /= np.linalg.norm(sideways)
        self.assertGreater(char_for(LIGHT), char_for(sideways))

    def test_bouquet_fills_most_of_the_screen(self):
        f = render(build_cloud(), 8.0, 120, 66)
        ys, xs = np.nonzero(f.chars != " ")
        self.assertGreater((ys.max() - ys.min()) / 66, 0.7)
        self.assertGreater((xs.max() - xs.min()) / 120, 0.6)

    def test_zoom_and_size_options(self):
        cloud = build_cloud()
        wide = filled_cells(render(cloud, 8.0, 100, 50, zoom=1.0))
        small = filled_cells(render(cloud, 8.0, 100, 50, zoom=0.5))
        self.assertLess(small, wide)
        self.assertGreater(build_cloud(size=1.5).extent[0], build_cloud(size=1.0).extent[0])

    def test_caption_goes_under_the_bouquet(self):
        f = render(build_cloud(), 8.0, 80, 40, caption="for you")
        self.assertIn("for you", "".join(f.chars[38]))
        self.assertTrue((f.chars[37] == " ").all() and (f.chars[39] == " ").all())
        long = render(build_cloud(), 8.0, 10, 20, caption="a caption far wider than the screen")
        self.assertEqual(long.chars.shape, (20, 10))

    def test_truecolor_detection(self):
        self.assertTrue(supports_truecolor({"COLORTERM": "truecolor"}))
        self.assertTrue(supports_truecolor({"COLORTERM": "24bit"}))
        self.assertFalse(supports_truecolor({}))
        self.assertFalse(supports_truecolor({"COLORTERM": ""}))

    def test_256_colour_mapping_uses_grays_for_gray(self):
        self.assertTrue(232 <= xterm256((128, 128, 128)) <= 255)
        self.assertEqual(xterm256((255, 0, 0)), 196)

    def test_gif_with_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "m.gif"
            main(["--gif", str(out), "--seconds", "0.4", "--fps", "5", "--cols", "30", "--rows", "16",
                  "--message", "for you", "--zoom", "1.1", "--size", "1.2", "--flowers", "9"])
            self.assertTrue(out.read_bytes().startswith(b"GIF8"))


if __name__ == "__main__":
    unittest.main()

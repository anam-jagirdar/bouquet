# Bouquet

An animated flower bouquet drawn entirely with code. No images, no assets: every petal, stem and leaf is generated from math.

![Animated bouquet](docs/bouquet.gif)

Stems grow out of the paper wrapping, the buds open one after another, then the whole bouquet sways while sparkles drift upward.

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m bouquet                       # open a live window
python -m bouquet --save my.gif         # write a GIF instead
```

Every run is customizable:

| Option | What it does | Default |
| --- | --- | --- |
| `--save PATH` | Write a GIF to this path instead of opening a window | none |
| `--seed N` | A different layout and colour mix. Same seed, same bouquet. | 7 |
| `--flowers N` | How many flowers. More flowers means a fuller bouquet. | 7 |
| `--size X` | Flower size multiplier. 1.2 makes every flower 20% bigger. | 1.0 |
| `--frames N` | Frames in one full animation | 150 |
| `--fps N` | Frames per second | 30 |

The 2D version saves with `--save`. The 3D version below saves with `--gif`.

For a fuller, bigger bouquet, try `python -m bouquet --flowers 11 --size 1.1`. Flowers are kept inside the frame, and bunches of more than 8 get a third row of heights so they stay layered instead of colliding.

## 3D ASCII version

The same bouquet as a real 3D scene drawn in coloured text, right in your terminal. The camera spins around it while stems grow, petals spiral open, and sparkles drift up.

![3D ASCII bouquet](docs/bouquet3d.gif)

```bash
python -m bouquet.ascii3d                  # live, fills your terminal window
python -m bouquet.ascii3d --gif mine.gif   # save a shareable GIF instead
```

Make your terminal window big first (full screen works best). Press **Ctrl+C** to quit.

It scales itself to fill your terminal, so a bigger window shows more detail.

| Option | What it does |
| --- | --- |
| `--gif PATH` | Write a GIF instead of animating in the terminal |
| `--message "text"` | Show a caption under the bouquet, e.g. `--message "for you"` |
| `--flowers N` | Number of flowers (default 13) |
| `--size X` | Flower size multiplier. 1.3 makes the flowers 30% bigger. |
| `--zoom X` | Zoom the whole bouquet in or out |
| `--seed N` | Different layout and colours |
| `--cols N`, `--rows N` | Size in characters (default: your terminal size) |
| `--fps N` | Frames per second |
| `--seconds N` | GIF length (default 8) |
| `--color auto\|256\|truecolor` | `auto` uses 24-bit colour only in terminals that support it (iTerm2, VS Code). macOS Terminal uses 256 colours. |

Try `python -m bouquet.ascii3d --message "for you" --size 1.2`.

**What's in the bouquet:** roses, cosmos, daisies and sunflowers in dozens of shades, plus baby's breath and lavender as filler, wrapped in paper with a ribbon.

**How it works:**

1. The bouquet is a cloud of about 34,000 points in 3D: petals, stems, leaves, filler flowers, a paper cone and a ribbon. Flower centres and rose petal layers follow the golden-angle spiral (the same pattern as sunflower seeds).
2. Each frame, every point grows out of an anchor (stems from the bundle, petals from the flower centre) on its own schedule, and the cloud is rotated by the camera with perspective.
3. **Lighting.** Every point knows which way its surface faces. Surfaces turned toward a fixed light render brighter, so the shading shifts as the bouquet spins. Petals also run from a deeper colour at the base to a lighter one at the tip.
4. A **depth buffer** keeps only the nearest point in each terminal cell, so flowers correctly hide what is behind them.
5. The lit brightness picks a character from a ramp (`.:-=+*#%@`), and the point's colour becomes the character's colour.
6. The bouquet's bounding size is measured once, and the camera scales to fit whatever terminal you have.

Everything is vectorised with NumPy: a 120x66 frame takes about 5 ms to compute.

## How the 2D version works

The whole animation is a **pure function of time**: `draw_frame(ax, scene, t)` draws the bouquet at any moment `t` between 0 and 1, with no state carried between frames. That is why it is easy to test and why any single frame can be rendered on its own.

The timeline is a set of overlapping windows on `t`:

| Time | What happens |
| --- | --- |
| 0.00 - 0.35 | Stems grow and leaves unfurl, each one starting slightly after the last |
| 0.36 - 0.82 | Flowers bloom in a random order |
| 0.60 - 1.00 | The bouquet sways and sparkles fade in |

Techniques used:

- **Quadratic Bezier curves** for the stems. Drawing only the first fraction of a curve makes it look like it is growing.
- **Easing functions.** Stems use an ease-out curve. Blooms use an ease-out-back curve that overshoots slightly before settling, which gives the "pop" feel.
- **Parametric petal shapes.** One teardrop polygon, rotated and repeated in rings, becomes a rose, a daisy or a cosmos just by changing the count, width and colour per layer.
- **Seeded randomness.** `numpy.random.default_rng(seed)` makes every bouquet reproducible.

## Project layout

```
bouquet/geometry.py   Easing, Bezier curves, rotation, petal shape.
bouquet/flowers.py    The three flower drawers: rose, daisy, cosmos.
bouquet/scene.py      Layout, timeline, wrapping and per-frame drawing.
bouquet/cli.py        Command line for the 2D version: live window or GIF export.
bouquet/ascii3d.py    The 3D ASCII version: point cloud, camera, depth buffer, output.
tests/                Geometry, scene logic and GIF export.
```

## Tests

```bash
python -m unittest discover -v
```

## Make it yours

- Add a fourth flower type (a tulip or sunflower) in `flowers.py` and add it to `DRAWERS`
- Change the palette, background or wrapping colours at the top of `scene.py`
- Add a falling-petals effect after the bloom
- Add a ribbon text option to the 2D version, like the `--message` caption the 3D version already has

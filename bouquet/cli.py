"""Command-line entry point.

    python -m bouquet                      open a live window
    python -m bouquet --save bouquet.gif   write a GIF instead
"""
import argparse
from pathlib import Path

import matplotlib

from .scene import FIGSIZE, build_scene, draw_frame, make_axes


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="An animated flower bouquet drawn with code.")
    parser.add_argument("--save", type=Path, help="write a GIF to this path instead of opening a window")
    parser.add_argument("--frames", type=int, default=150, help="frames in one full animation (default 150)")
    parser.add_argument("--fps", type=int, default=30, help="frames per second (default 30)")
    parser.add_argument("--seed", type=int, default=7, help="changes the layout and colours (default 7)")
    parser.add_argument("--flowers", type=int, default=7, help="number of flowers (default 7)")
    parser.add_argument("--size", type=float, default=1.0, help="flower size multiplier (default 1.0)")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.save:
        matplotlib.use("Agg")  # no window needed when writing a file

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig = plt.figure(figsize=FIGSIZE, dpi=100)
    ax = make_axes(fig)
    scene = build_scene(args.seed, args.flowers, args.size)

    def update(frame: int):
        draw_frame(ax, scene, frame / args.frames)
        return []

    animation = FuncAnimation(fig, update, frames=args.frames, interval=1000 / args.fps, repeat=True)

    if args.save:
        animation.save(args.save, writer=PillowWriter(fps=args.fps))
        print(f"Saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

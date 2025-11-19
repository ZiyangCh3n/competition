import os
import sys
import math
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, Circle
import matplotlib.transforms as transforms

# You can tweak this if rods still look too thin or too fat
RENDER_RADIUS_SCALE = 1.0  # 1.0 = pure geometry, >1.0 = visually thicker rods


def importPickle(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, tuple):
        data = {'cellStates': data[0]}
    return data


def calc_cell_colors(state):
    r, g, b = state.color
    return (r, g, b), (r * 0.5, g * 0.5, b * 0.5)


def draw_capsule(ax, p, d, l, r, fill, stroke):
    """
    Draw a spherocylinder like the original PDF:
    - rectangle of length l and height 2r
    - circular caps at each end
    Then rotate according to d and translate to p.
    """
    # optionally scale radius for nicer visual thickness
    r_vis = r * RENDER_RADIUS_SCALE

    # base shapes in local coordinates (centered at origin, pointing along +x)
    # rectangle body
    rect = Rectangle(
        (-l / 2.0, -r_vis),
        l,
        2 * r_vis,
        linewidth=0.5,
        edgecolor=stroke,
        facecolor=fill
    )

    # circular caps (full circles; overlap with rect is fine visually)
    left_cap = Circle(
        (-l / 2.0, 0.0),
        r_vis,
        linewidth=0.5,
        edgecolor=stroke,
        facecolor=fill
    )
    right_cap = Circle(
        (l / 2.0, 0.0),
        r_vis,
        linewidth=0.5,
        edgecolor=stroke,
        facecolor=fill
    )

    # rotation from direction vector d, then translate to p
    angle = math.atan2(d[1], d[0])
    t = transforms.Affine2D().rotate(angle).translate(p[0], p[1]) + ax.transData

    for patch in (rect, left_cap, right_cap):
        patch.set_transform(t)
        ax.add_patch(patch)


def draw_signals(ax, data, index=0, z=0):
    levels = data.get('sigGrid', None)
    orig = data.get('sigGridOrig', None)
    dim = data.get('sigGridDim', None)
    grid_size = data.get('sigGridSize', None)

    if levels is None:
        return

    levels = np.array(levels).reshape(dim)
    slice2d = levels[index, :, :, z]
    mx = slice2d.max()
    if mx <= 0:
        return

    norm = slice2d.T / mx
    dx, dy, dz = list(map(float, grid_size))
    ox, oy, oz = orig
    nx, ny, nz = dim[1], dim[2], dim[3]

    extent = [
        ox - dx / 2,
        ox + dx * nx - dx / 2,
        oy - dy / 2,
        oy + dy * ny - dy / 2,
    ]

    ax.imshow(norm, origin='lower', extent=extent,
              cmap='Reds', alpha=0.6)


def compute_global_box(all_data, min_size=40.0):
    mnx, mxx = -20.0, 20.0
    mny, mxy = -20.0, 20.0

    for data in all_data:
        for _, s in data['cellStates'].items():
            x, y = s.pos[0], s.pos[1]
            l = s.length
            mnx = min(mnx, x - l)
            mxx = max(mxx, x + l)
            mny = min(mny, y - l)
            mxy = max(mxy, y + l)

    w = max(mxx - mnx, min_size)
    h = max(mxy - mny, min_size)

    cx = 0.5 * (mnx + mxx)
    cy = 0.5 * (mny + mxy)

    return cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2


def draw_frame(ax, data, bounds):
    xmin, xmax, ymin, ymax = bounds

    ax.clear()
    ax.set_facecolor('black')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.axis('off')

    draw_signals(ax, data)

    for _, state in data['cellStates'].items():
        fill, stroke = calc_cell_colors(state)
        draw_capsule(ax, state.pos, state.dir, state.length, state.radius, fill, stroke)


def main():
    if len(sys.argv) != 3:
        print("Usage: python cellmodeller_video.py <input_folder> <output_video_path.mp4>")
        sys.exit(1)

    in_dir = sys.argv[1]
    out_file = sys.argv[2]

    if not os.path.isdir(in_dir):
        print(f"Error: input directory {in_dir} does not exist.")
        sys.exit(1)

    infns = sorted(
        os.path.join(in_dir, f)
        for f in os.listdir(in_dir)
        if f.endswith('.pickle')
    )

    if not infns:
        print("Error: no .pickle files in input directory!")
        sys.exit(1)

    print(f"Output video → {out_file}")

    all_data = [importPickle(fn) for fn in infns]

    bounds = compute_global_box(all_data)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('black')

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=12, metadata=dict(artist='CellModeller'), bitrate=2000)

    with writer.saving(fig, out_file, dpi=150):
        for i, data in enumerate(all_data):
            print(f"Rendering frame {i+1}/{len(all_data)}")
            draw_frame(ax, data, bounds)
            writer.grab_frame()

    print("Done!")


if __name__ == "__main__":
    main()

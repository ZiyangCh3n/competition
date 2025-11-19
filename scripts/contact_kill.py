# Two-species growth in CellModeller with PA killing nearby SA
# Species 0 ("SA") grows faster; Species 1 ("PA") grows slower.
# Run with:  python CellModeller/Scripts/CellModellerGUI.py /path/to/two_species_growth.py

import random
from math import sqrt
import numpy as np
from CellModeller.Biophysics.BacterialModels.CLBacterium import CLBacterium
from CellModeller.Regulation.ModuleRegulator import ModuleRegulator
from CellModeller.GUI import Renderers

# -----------------------------
# Tunable parameters
# -----------------------------
N_SA_START = 3
N_PA_START = 1
INIT_SPREAD = 25.0  # microns around origin for initial seeding

SA_MU = 1.8   # "SA" base growth rate (fast)
PA_MU = 0.6   # "PA" base growth rate (slow)

DIV_LENGTH_MEAN_PA = 3.5   # mean target length for division
DIV_LENGTH_MEAN_SA = 1.0
DIV_LENGTH_JITTER = 0.6    # random jitter added to target length

MAX_CELLS = 10000

# Global crowding (simple logistic-like saturation)
# Keep capacity equal to your original MAX_CELLS so early dynamics are unchanged
CARRYING_CAPACITY = MAX_CELLS

# RGB colors for rendering in GUI
COL_SA   = [0.0, 1.0, 0.0]  # SA = green
COL_PA   = [0.0, 0.0, 1.0]  # PA = blue
COL_DEAD = [0.6, 0.6, 0.6]  # dead = gray

# Killing parameters
# Introduce effective radii for SA and PA whose sum equals your original KILL_RADIUS = 2.0
EFFECTIVE_RADIUS_SA = 1.0
EFFECTIVE_RADIUS_PA = 1.0
KILL_RADIUS = EFFECTIVE_RADIUS_SA + EFFECTIVE_RADIUS_PA   # numerically still 2.0
KILL_RADIUS_SQ = KILL_RADIUS * KILL_RADIUS

# Grid size for spatial hashing (>= kill radius)
GRID_SIZE = KILL_RADIUS

# Toggle: set to False to disable contact killing
CONTACT_KILLING = True
PRINT_EVERY = 100   # print every 100 steps (tune as you like)
STEP_COUNTER = 0
DEAD_LIFETIME = 20   # number of steps after which a dead cell is removed

# Precomputed neighbor offsets for 3x3 grid neighborhood
NEIGHBOR_OFFSETS = [(dxg, dyg) for dxg in (-1, 0, 1) for dyg in (-1, 0, 1)]


def grid_index(x, y):
    """Map (x, y) to integer grid coordinates."""
    return (int(np.floor(x / GRID_SIZE)), int(np.floor(y / GRID_SIZE)))

# -----------------------------
# CellModeller hooks
# -----------------------------
def setup(sim):
    # Biophysics engine (rod-shaped bacteria in 2D)
    biophys = CLBacterium(
        sim,
        jitter_z=False,
        max_cells=MAX_CELLS,
        max_planes=3,
        gamma=50.0,
    )

    # Regulation (we implement simple growth rules below)
    regul = ModuleRegulator(sim, sim.moduleName)

    # No signaling or genes objects for this minimal model
    sim.init(biophys, regul, None, None)

    # Seed initial cells for both species near the origin
    rng = random.Random(1)
    for _ in range(N_SA_START):
        x = (rng.random()*2 - 1) * INIT_SPREAD
        y = (rng.random()*2 - 1) * INIT_SPREAD
        # cellType=0 => SA
        sim.addCell(cellType=0, pos=(x, y, 0),
                    dir=((rng.random()*2-1), (rng.random()*2-1), 0))

    for _ in range(N_PA_START):
        x = (rng.random()*2 - 1) * (INIT_SPREAD/2)
        y = (rng.random()*2 - 1) * (INIT_SPREAD/2)
        # cellType=1 => PA
        sim.addCell(cellType=1, pos=(x, y, 0),
                    dir=((rng.random()*2-1), (rng.random()*2-1), 0))

    # Add a 3D renderer to visualize in the GUI
    if sim.is_gui:
        sim.addRenderer(Renderers.GLBacteriumRenderer(sim))

    # Pickle snapshots occasionally (change as desired)
    sim.pickleSteps = 100


def init(cell):
    """Called once when a new cell is created/added."""
    ctype = cell.cellType

    # Assign base growth rate by species
    if ctype == 0:  # SA
        cell.growthRate = SA_MU
        cell.color = COL_SA
        cell.targetVol = DIV_LENGTH_MEAN_SA + random.uniform(0.0, 0.15)
    elif ctype == 1:  # PA
        cell.growthRate = PA_MU
        cell.color = COL_PA
        cell.targetVol = DIV_LENGTH_MEAN_PA + random.uniform(0.0, 0.5)
    else:  # dead cell
        cell.growthRate = 0.0
        cell.color = [0.6, 0.6, 0.6]

    # Reset flags
    cell.divideFlag = False
    cell.deadCounter = 0


def update(cells):
    global STEP_COUNTER
    STEP_COUNTER += 1

    cells_to_remove = []

    # Global crowding factor (logistic-like slowdown)
    n_cells = len(cells)
    if CARRYING_CAPACITY > 0:
        crowd_factor = max(0.0, 1.0 - float(n_cells) / CARRYING_CAPACITY)
    else:
        crowd_factor = 1.0

    # ------------------------------------------------------
    # Branch 1: no contact killing, just growth/division
    # ------------------------------------------------------
    if not CONTACT_KILLING:
        for cid, c in cells.items():
            ctype = c.cellType

            if ctype == 2:  # dead
                c.growthRate = 0.0
                c.divideFlag = False
                c.color = COL_DEAD

                # dead cells count up, then get removed
                c.deadCounter += 1
                if c.deadCounter >= DEAD_LIFETIME:
                    cells_to_remove.append(cid)

            elif ctype == 0:  # SA
                c.growthRate = SA_MU * crowd_factor
                c.divideFlag = (c.volume > c.targetVol)
                c.deadCounter = 0

            elif ctype == 1:  # PA
                c.growthRate = PA_MU * crowd_factor
                c.divideFlag = (c.volume > c.targetVol)
                c.deadCounter = 0

        # remove dead cells after scanning
        for cid in cells_to_remove:
            cells.pop(cid, None)

        # occasional print
        if STEP_COUNTER % PRINT_EVERY == 0:
            n_sa = n_pa = n_dead = 0
            for c in cells.values():
                ctype = c.cellType
                if ctype == 0:
                    n_sa += 1
                elif ctype == 1:
                    n_pa += 1
                elif ctype == 2:
                    n_dead += 1
            total = len(cells)
            print(f"[step {STEP_COUNTER}] SA={n_sa}, PA={n_pa}, dead={n_dead}, total={total}")

        return  # done for this step

    # ------------------------------------------------------
    # Branch 2: contact killing ON (spatial grid for PA)
    # ------------------------------------------------------

    pa_grid = {}   # (gx, gy) -> list of (xp, yp, cid)
    sa_ids  = []

    # First pass: classify cells, update deadCounter & schedule deletions
    for cid, c in cells.items():
        ctype = c.cellType

        if ctype == 1:  # PA
            x, y, z = c.pos
            gx, gy = grid_index(x, y)
            pa_grid.setdefault((gx, gy), []).append((x, y, cid))

            c.growthRate = PA_MU * crowd_factor
            c.divideFlag = (c.volume > c.targetVol)
            c.deadCounter = 0

        elif ctype == 0:  # SA
            sa_ids.append(cid)
            # growth / division will be set after we know if it's killed
            c.deadCounter = 0

        elif ctype == 2:  # dead
            c.growthRate = 0.0
            c.divideFlag = False
            c.color = COL_DEAD

            c.deadCounter += 1
            if c.deadCounter >= DEAD_LIFETIME:
                cells_to_remove.append(cid)

    # Handle SA cells: check nearby PA in grid
    kill_radius_sq_local = KILL_RADIUS_SQ  # local binding
    for cid in sa_ids:
        c = cells[cid]
        x0, y0, z0 = c.pos
        gx0, gy0 = grid_index(x0, y0)

        killed = False
        x0_local = x0
        y0_local = y0

        # Use precomputed neighbor offsets
        for dxg, dyg in NEIGHBOR_OFFSETS:
            if killed:
                break

            cell_list = pa_grid.get((gx0 + dxg, gy0 + dyg))
            if not cell_list:
                continue

            for xp, yp, pa_id in cell_list:
                dx = x0_local - xp
                dy = y0_local - yp
                if dx*dx + dy*dy <= kill_radius_sq_local:
                    # kill SA -> becomes dead cellType 2
                    c.cellType = 2
                    c.growthRate = 0.0
                    c.divideFlag = False
                    c.color = COL_DEAD
                    c.deadCounter = 0  # start counting from now
                    killed = True
                    break

        if not killed:
            # still alive SA
            c.growthRate = SA_MU * crowd_factor
            c.divideFlag = (c.volume > c.targetVol)

    # Remove dead cells (both old dead and those that aged out this step)
    for cid in cells_to_remove:
        cells.pop(cid, None)

    # occasional print
    if STEP_COUNTER % PRINT_EVERY == 0:
        n_sa = n_pa = n_dead = 0
        for c in cells.values():
            ctype = c.cellType
            if ctype == 0:
                n_sa += 1
            elif ctype == 1:
                n_pa += 1
            elif ctype == 2:
                n_dead += 1
        total = len(cells)
        print(f"!!!!![step {STEP_COUNTER}] SA={n_sa}, PA={n_pa}, dead={n_dead}, total={total}")


def divide(parent, d1, d2):
    """Called when a cell divides; set properties of daughters."""
    # Keep the same species after division
    ptype = parent.cellType
    d1.cellType = ptype
    d2.cellType = ptype

    # Inherit color and growth rate by species
    if ptype == 0:
        for d in (d1, d2):
            d.color = COL_SA
            d.growthRate = SA_MU
            d.targetVol = DIV_LENGTH_MEAN_SA + random.uniform(0.0, 0.15)
    elif ptype == 1:
        for d in (d1, d2):
            d.color = COL_PA
            d.growthRate = PA_MU
            d.targetVol = DIV_LENGTH_MEAN_PA + random.uniform(0.0, 0.5)
    # else:
    #     for d in (d1, d2):
    #         d.color = [0.6, 0.6, 0.6]
    #         d.growthRate = 0.0
    #         d.targetVol = 3.0

    # Reset flags
    for d in (d1, d2):
        d.divideFlag = False
        d.deadCounter = 0
        # d.killFlag = False

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

SA_MU = 1.2   # "SA" base growth rate (fast)
PA_MU = 0.6   # "PA" base growth rate (slow)

DIV_LENGTH_MEAN_PA = 3.5   # mean target length for division
DIV_LENGTH_MEAN_SA = 1.0
DIV_LENGTH_JITTER = 0.6    # random jitter added to target length

MAX_CELLS = 20000

# RGB colors for rendering in GUI
COL_SA = [1.0,0,0]  # green
COL_PA = [0,0,1.0]  # blue
COL_DEAD = [0.6, 0.6, 0.6]

# Killing parameters (NEW)
KILL_RADIUS = 2.0   # microns; PA kills SA within this distance
KILL_RADIUS_SQ = KILL_RADIUS * KILL_RADIUS



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
        sim.addCell(cellType=0, pos=(x, y, 0), dir=((rng.random()*2-1), (rng.random()*2-1), 0))

    for _ in range(N_PA_START):
        x = (rng.random()*2 - 1) * (INIT_SPREAD/2)
        y = (rng.random()*2 - 1) * (INIT_SPREAD/2)
        # cellType=1 => PA
        sim.addCell(cellType=1, pos=(x, y, 0), dir=((rng.random()*2-1), (rng.random()*2-1), 0))

    # Add a 3D renderer to visualize in the GUI
    if sim.is_gui:
        sim.addRenderer(Renderers.GLBacteriumRenderer(sim))

    # Pickle snapshots occasionally (change as desired)
    sim.pickleSteps = 100


def init(cell):
    """Called once when a new cell is created/added."""
    # Assign base growth rate by species
    if cell.cellType == 0:  # SA
        cell.growthRate = SA_MU
        cell.color = COL_SA
        cell.targetVol = DIV_LENGTH_MEAN_SA + random.uniform(0.0, 0.15) 
    elif cell.cellType == 1:  # PA
        cell.growthRate = PA_MU
        cell.color = COL_PA
        cell.targetVol = DIV_LENGTH_MEAN_PA + random.uniform(0.0, 0.5)
    else: # dead cell
        cell.growthRate = 0.0
        cell.color = [0.6, 0.6, 0.6]

    # Target length before division (introduce some variability)
    # cell.targetVol = DIV_LENGTH_MEAN + (random.random()-0.5)*DIV_LENGTH_JITTER

    # Reset flags
    cell.divideFlag = False

# def get_pa_pos(cells):
#     """Called every simulation step; apply growth, division, and killing."""
#     # Pre-collect PA positions for killing logic
#     pa_positions = []
#     for (cid, c) in cells.items():
#         if c.cellType == 1:  # PA
#             x, y, z = c.pos
#             pa_positions.append((x, y))
#     return pa_positions

# def update(cells):
#     # global step_count
#     # step_count += 1
#     pa_positions = get_pa_pos(cells)
#     for cid, c in list(cells.items()):
#         # Division rule: for rods, use length not volume
#         if c.volume > c.targetVol:
#             c.divideFlag = True
#         else:
#             c.divideFlag = False

#         # Growth rates (baseline by species)
#         if c.cellType == 0:
#             x_s, y_s, z_s = c.pos
#             c.growthRate = SA_MU
#             for (x_p, y_p) in pa_positions:
#                 dx = x_s - x_p
#                 dy = y_s - y_p
#                 if dx*dx + dy*dy <= KILL_RADIUS*KILL_RADIUS:
#                     c.growthRate = 0.0
#                     c.color = np.divide(COL_SA,2)
#                     print(f"Killing SA {cid} near PA")
#                     break  # no need to check other PA
#         elif c.cellType == 1:
#             c.growthRate = PA_MU
#         else:
#             c.growthRate = 0.0

# def contact_kill(cells, x0, y0):
#     for (cid, c) in cells.items():
#         if c.cellType == 1:
#             x, y, z = c.pos
#             dist = np.linalg.norm(np.array([x0, y0])-np.array([x, y]))
#             if dist <= KILL_RADIUS:
#                 return True
#     return False

# def update(cells):
#     for (cid, c) in cells.items():
#         if c.cellType == 0:
#             x0, y0, z0 = c.pos
#             killflag = contact_kill(cells, x0, y0)
#             if killflag:
#                 c.cellType = 2
#                 c.growthRate = 0
#                 c.color = [0.6, 0.6, 0.6]
#             else:
#                 c.growthRate = SA_MU
#         elif c.cellType == 1:
#             c.growthRate = PA_MU
#         else:
#             c.growthRate = 0
#         if c.volume > c.targetVol:
#             c.divideFlag = True
#         else:
#             c.divideFlag = False

def update(cells):
    # Pre-collect PA and SA cell IDs and PA positions
    pa_list = []   # list of (cid, (x, y, z))
    sa_list = []   # list of cid

    for cid, c in cells.items():
        if c.cellType == 1:      # PA
            pa_list.append((cid, c.pos))
        elif c.cellType == 0:    # SA
            sa_list.append(cid)
        elif c.cellType == 2:    # dead cell
            # Ensure dead cells stay dead and don't divide
            c.growthRate = 0.0
            c.divideFlag = False

    # --- Handle SA cells: check if any PA within kill radius ---
    for cid in sa_list:
        c = cells[cid]
        x0, y0, z0 = c.pos

        killed = False
        for _, (xp, yp, zp) in pa_list:
            dx = x0 - xp
            dy = y0 - yp
            # No sqrt -> faster distance check
            if dx*dx + dy*dy <= KILL_RADIUS_SQ:
                # Kill this SA -> type 2 = dead
                c.cellType = 2
                c.growthRate = 0.0
                c.color = COL_DEAD
                c.divideFlag = False
                killed = True
                break

        if not killed:
            # Still alive SA
            c.growthRate = SA_MU
            c.divideFlag = (c.volume > c.targetVol)

    # --- Handle PA cells: just grow & divide ---
    for cid, _ in pa_list:
        c = cells[cid]
        c.growthRate = PA_MU
        c.divideFlag = (c.volume > c.targetVol)




def divide(parent, d1, d2):
    """Called when a cell divides; set properties of daughters."""
    # Keep the same species after division
    d1.cellType = parent.cellType
    d2.cellType = parent.cellType

    # Inherit color and growth rate by species
    if parent.cellType == 0:
        for d in (d1, d2):
            d.color = COL_SA
            d.growthRate = SA_MU
            d.targetVol = DIV_LENGTH_MEAN_SA + random.uniform(0.0, 0.15)
    elif parent.cellType == 1:
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
        # d.killFlag = False

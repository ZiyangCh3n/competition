# Two-species growth in CellModeller (no signals, no killing)
# Species 0 ("SA") grows faster; Species 1 ("PA") grows slower.
# Run with:  python CellModeller/Scripts/CellModellerGUI.py /path/to/two_species_growth.py

import random
from math import sqrt

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
DIV_LENGTH_JITTER = 0.6 # random jitter added to target length

MAX_CELLS = 20000

# RGB colors for rendering in GUI
COL_SA = [0.20, 0.80, 0.20]  # green
COL_PA = [0.20, 0.40, 0.95]  # blue

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
    # else:
    #     cell.growthRate = 0.0
    #     cell.color = [0.6, 0.6, 0.6]

    # Target length before division (introduce some variability)
    # cell.targetVol = DIV_LENGTH_MEAN + (random.random()-0.5)*DIV_LENGTH_JITTER

    # Reset flags
    cell.divideFlag = False


def update(cells):
    """Called every simulation step; apply growth & division rules."""
    for cid, c in cells.items():
        # Simple division rule: when current length exceeds target length
        # if c.length > c.targetVol:
        if c.volume > c.targetVol:
            c.divideFlag = True
        else:
            c.divideFlag = False
        # Growth rates remain constant per species in this minimal model
        if c.cellType == 0:
            c.growthRate = SA_MU
        elif c.cellType == 1:
            c.growthRate = PA_MU


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

    # Reset division targets with some jitter
    for d in (d1, d2):
        # d.targetVol = DIV_LENGTH_MEAN + (random.random()-0.5)*DIV_LENGTH_JITTER
        d.divideFlag = False

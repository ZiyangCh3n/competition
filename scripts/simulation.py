# Two-species growth in CellModeller with PA killing SA
# Species 0 ("SA") grows faster; Species 1 ("PA") grows slower.
# Contact-dependent killing + optional diffusive toxin killing using GridDiffusion.
#
# Run with:
#   python CellModeller/Scripts/CellModellerGUI.py /path/to/this_script.py

import random
from math import sqrt
import numpy as np
from CellModeller.Biophysics.BacterialModels.CLBacterium import CLBacterium
from CellModeller.Regulation.ModuleRegulator import ModuleRegulator
from CellModeller.GUI import Renderers
from CellModeller.Signalling.GridDiffusion import GridDiffusion
from CellModeller.Integration.CLCrankNicIntegrator import CLCrankNicIntegrator

# -----------------------------
# Tunable parameters (your originals)
# -----------------------------
N_SA_START = 3
N_PA_START = 1
INIT_SPREAD = 25.0  # microns around origin for initial seeding

SA_MU = 1.8   # "SA" base growth rate (fast)
PA_MU = 0.6   # "PA" base growth rate (slow)

DIV_LENGTH_MEAN_PA = 3.5   # mean target length for division
DIV_LENGTH_MEAN_SA = 1.0
DIV_LENGTH_JITTER = 0.6    # random jitter added to target length

MAX_CELLS = 20000

# Global crowding (simple logistic-like saturation)
CARRYING_CAPACITY = MAX_CELLS  # use same scale as your max cells

# RGB colors for rendering in GUI
COL_SA   = [1.0, 0,   0]   # SA
COL_PA   = [0,   0,   1.0] # PA
COL_DEAD = [0.6, 0.6, 0.6]

# Contact-killing parameters
# Original KILL_RADIUS was 2.0 -> use two radii that sum to 2
EFFECTIVE_RADIUS_SA = 1.0
EFFECTIVE_RADIUS_PA = 1.0
KILL_RADIUS = EFFECTIVE_RADIUS_SA + EFFECTIVE_RADIUS_PA   # = 2.0
KILL_RADIUS_SQ = KILL_RADIUS * KILL_RADIUS

# Grid size for spatial hashing (>= kill radius)
GRID_SIZE = KILL_RADIUS

# Killing method toggles
CONTACT_KILLING   = False   # PA kills SA by direct contact
DIFFUSIVE_KILLING = True  # PA secretes diffusive toxin that kills SA

PRINT_EVERY   = 100   # print every 100 steps
STEP_COUNTER  = 0
DEAD_LIFETIME = 20    # number of steps after which a dead cell is removed

# Diffusive toxin parameters (for GridDiffusion-based killing)
# One species + one signal: toxin_in (species[0]) and toxin_out (signals[0])
TOXIN_DIFF_RATE        = 10.0  # diffusion coefficient on grid (arbitrary)
TOXIN_MEMBRANE_DIFF    = 1.0   # in/out of cell
TOXIN_PROD_RATE_PA     = 1.0   # production rate in PA cells
TOXIN_KILL_THRESHOLD   = 0.5   # SA dies if intracellular toxin > this

# Precomputed neighbor offsets for 3x3 grid neighborhood (for PA spatial hash)
NEIGHBOR_OFFSETS = [(dxg, dyg) for dxg in (-1, 0, 1) for dyg in (-1, 0, 1)]

def toxin_to_color(cell):
    """
    Return an [R,G,B] color for a cell based on its intracellular toxin level.
    - Uses species[0] (toxin_in).
    - Low toxin: normal species color.
    - High toxin: fades to white.
    """
    # Dead cells keep their dead color
    if cell.cellType == 2:
        return COL_DEAD

    # Base color by species
    if cell.cellType == 0:      # SA
        base = COL_SA
    elif cell.cellType == 1:    # PA
        base = COL_PA
    else:
        base = [0.5, 0.5, 0.5]

    if DIFFUSIVE_KILLING:
        tox = float(cell.species[0])  # intracellular toxin
        # Normalize to kill threshold: 0 → no toxin, 1 → at kill threshold
        norm = min(tox / TOXIN_KILL_THRESHOLD, 1.0)
        # Blend towards white as toxin increases
        r = base[0] * (1.0 - norm) + 1.0 * norm
        g = base[1] * (1.0 - norm) + 1.0 * norm
        b = base[2] * (1.0 - norm) + 1.0 * norm
        return [r, g, b]
    else:
        # No diffusive killing: just use base species color
        return base



def grid_index(x, y):
    """Map (x, y) to integer grid coordinates for spatial hashing."""
    return (int(np.floor(x / GRID_SIZE)), int(np.floor(y / GRID_SIZE)))


# -----------------------------
# OpenCL reaction kernels for toxin (GridDiffusion + CLCrankNicIntegrator)
# -----------------------------
# We have 1 intracellular species (toxin_in) and 1 extracellular signal (toxin).
# Toxin is produced only in PA (cellType==1), and diffuses between inside/outside.

cl_prefix = r'''
    const float D_tox = %(D_tox).1ff;
    const float k_tox_PA = %(k_pa).1ff;

    float toxin_in = species[0];
    float toxin    = signals[0];
''' % {
    'D_tox': TOXIN_MEMBRANE_DIFF,
    'k_pa':  TOXIN_PROD_RATE_PA,
}

def specRateCL():
    """Intracellular reaction rates (for species[]) in OpenCL."""
    global cl_prefix
    return cl_prefix + r'''
        if (cellType == 1){
            // PA: produce toxin + exchange with outside
            rates[0] = k_tox_PA + D_tox*(toxin - toxin_in)*area/gridVolume;
        } else {
            // SA & others: only exchange toxin with outside
            rates[0] = D_tox*(toxin - toxin_in)*area/gridVolume;
        }
    '''

def sigRateCL():
    """Extracellular reaction rates (for signals[]) in OpenCL."""
    global cl_prefix
    return cl_prefix + r'''
        // Diffusion on the grid handled by GridDiffusion.
        // Here we only handle exchange with cells.
        rates[0] = -D_tox*(toxin - toxin_in)*area/gridVolume;
    '''


# -----------------------------
# CellModeller hooks
# -----------------------------
def setup(sim):
    global MAX_CELLS

    # Biophysics engine (rod-shaped bacteria in 2D)
    biophys = CLBacterium(
        sim,
        jitter_z=False,
        max_cells=MAX_CELLS,
        max_planes=3,
        gamma=10.0,
    )

    # Regulation (we implement simple growth rules + use CL kernels above)
    regul = ModuleRegulator(sim, sim.moduleName)

    # ---- Signalling: GridDiffusion for diffusive toxin ----
    # Grid big enough so your colony at ~0,0 (±25) sits well inside.
    grid_dim  = (80, 80, 8)         # number of grid points in x,y,z
    grid_size = (4.0, 4.0, 4.0)     # spacing between grid points
    grid_orig = (-160., -160., -16.)
    n_signals = 1                   # just toxin
    n_species = 1                   # intracellular toxin

    diff_rates = [TOXIN_DIFF_RATE]  # diffusion coeff. per signal on grid

    sig   = GridDiffusion(sim, n_signals, grid_dim, grid_size, grid_orig, diff_rates)
    integ = CLCrankNicIntegrator(sim, n_signals, n_species, MAX_CELLS, sig)

    sim.init(biophys, regul, sig, integ)

    # Seed initial cells for both species near the origin
    rng = random.Random(1)
    for _ in range(N_SA_START):
        x = (rng.random()*2 - 1) * INIT_SPREAD
        y = (rng.random()*2 - 1) * INIT_SPREAD
        sim.addCell(
            cellType=0,  # SA
            pos=(x, y, 0),
            dir=((rng.random()*2 - 1), (rng.random()*2 - 1), 0),
        )

    for _ in range(N_PA_START):
        x = (rng.random()*2 - 1) * (INIT_SPREAD/2)
        y = (rng.random()*2 - 1) * (INIT_SPREAD/2)
        sim.addCell(
            cellType=1,  # PA
            pos=(x, y, 0),
            dir=((rng.random()*2 - 1), (rng.random()*2 - 1), 0),
        )

    # Add a 3D renderer to visualize in the GUI
    if sim.is_gui:
        sim.addRenderer(Renderers.GLBacteriumRenderer(sim))
        sim.addRenderer(Renderers.GLGridRenderer(sig, integ))

    # Pickle snapshots occasionally (change as desired)
    sim.pickleSteps = 100


def init(cell):
    """Called once when a new cell is created/added."""
    ctype = cell.cellType

    if ctype == 0:  # SA
        cell.growthRate = SA_MU
        cell.color = COL_SA
        cell.targetVol = DIV_LENGTH_MEAN_SA + random.uniform(0.0, 0.15)

    elif ctype == 1:  # PA
        cell.growthRate = PA_MU
        cell.color = COL_PA
        cell.targetVol = DIV_LENGTH_MEAN_PA + random.uniform(0.0, 0.5)

    else:  # dead
        cell.growthRate = 0.0
        cell.color = COL_DEAD
        cell.targetVol = 3.0

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
    # Branch 1: no killing at all, just growth/division
    # ------------------------------------------------------
    if not CONTACT_KILLING and not DIFFUSIVE_KILLING:
        for cid, c in cells.items():
            ctype = c.cellType

            if ctype == 2:  # dead
                c.growthRate = 0.0
                c.divideFlag = False
                c.color = COL_DEAD

                c.deadCounter += 1
                if c.deadCounter >= DEAD_LIFETIME:
                    cells_to_remove.append(cid)

            elif ctype == 0:  # SA
                c.growthRate = SA_MU * crowd_factor
                c.divideFlag = (c.volume > c.targetVol)
                c.deadCounter = 0
                c.color = toxin_to_color(c)

            elif ctype == 1:  # PA
                c.growthRate = PA_MU * crowd_factor
                c.divideFlag = (c.volume > c.targetVol)
                c.deadCounter = 0
                c.color = toxin_to_color(c)

        for cid in cells_to_remove:
            cells.pop(cid, None)

        if STEP_COUNTER % PRINT_EVERY == 0:
            n_sa = n_pa = n_dead = 0
            for c in cells.values():
                if c.cellType == 0:
                    n_sa += 1
                elif c.cellType == 1:
                    n_pa += 1
                elif c.cellType == 2:
                    n_dead += 1
            total = len(cells)
            print(f"[step {STEP_COUNTER}] SA={n_sa}, PA={n_pa}, dead={n_dead}, total={total}")

        return

    # ------------------------------------------------------
    # Branch 2: at least one killing mechanism is ON
    #  - build PA spatial grid
    #  - first handle PA & dead
    #  - then for each SA: diffusive toxin check, then contact killing
    # ------------------------------------------------------

    pa_grid = {}   # (gx, gy) -> list of (xp, yp, cid)
    sa_ids  = []

    for cid, c in cells.items():
        ctype = c.cellType

        if ctype == 1:  # PA
            x, y, z = c.pos
            gx, gy = grid_index(x, y)
            pa_grid.setdefault((gx, gy), []).append((x, y, cid))

            c.growthRate = PA_MU * crowd_factor
            c.divideFlag = (c.volume > c.targetVol)
            c.deadCounter = 0
            c.color = toxin_to_color(c)

        elif ctype == 0:  # SA
            sa_ids.append(cid)
            # growth/division set after kill checks
            c.deadCounter = 0

        elif ctype == 2:  # dead
            c.growthRate = 0.0
            c.divideFlag = False
            c.color = COL_DEAD

            c.deadCounter += 1
            if c.deadCounter >= DEAD_LIFETIME:
                cells_to_remove.append(cid)

    kill_radius_sq_local = KILL_RADIUS_SQ

    # SA handling: diffusive toxin, then contact killing
    for cid in sa_ids:
        c = cells[cid]
        x0, y0, z0 = c.pos
        gx0, gy0 = grid_index(x0, y0)

        killed = False

        # 1) Diffusive toxin killing using intracellular species concentration
        if DIFFUSIVE_KILLING:
            tox_in = c.species[0]  # intracellular toxin species
            if tox_in >= TOXIN_KILL_THRESHOLD:
                c.cellType = 2
                c.growthRate = 0.0
                c.divideFlag = False
                c.color = COL_DEAD
                c.deadCounter = 0
                killed = True

        # 2) Contact killing (if not already dead from toxin)
        if CONTACT_KILLING and not killed:
            x0_local = x0
            y0_local = y0

            for dxg, dyg in NEIGHBOR_OFFSETS:
                cell_list = pa_grid.get((gx0 + dxg, gy0 + dyg))
                if not cell_list:
                    continue

                for xp, yp, pa_id in cell_list:
                    dx = x0_local - xp
                    dy = y0_local - yp
                    if dx*dx + dy*dy <= kill_radius_sq_local:
                        c.cellType = 2
                        c.growthRate = 0.0
                        c.divideFlag = False
                        c.color = COL_DEAD
                        c.deadCounter = 0
                        killed = True
                        break
                if killed:
                    break

        # 3) If still alive SA, grow/divide with crowding
        if not killed:
            c.growthRate = SA_MU * crowd_factor
            c.divideFlag = (c.volume > c.targetVol)
            c.color = toxin_to_color(c)

    # Remove dead cells that have aged out
    for cid in cells_to_remove:
        cells.pop(cid, None)

    if STEP_COUNTER % PRINT_EVERY == 0:
        n_sa = n_pa = n_dead = 0
        for c in cells.values():
            if c.cellType == 0:
                n_sa += 1
            elif c.cellType == 1:
                n_pa += 1
            elif c.cellType == 2:
                n_dead += 1
        total = len(cells)
        print(f"!!!!![step {STEP_COUNTER}] SA={n_sa}, PA={n_pa}, dead={n_dead}, total={total}")
    
    if STEP_COUNTER % PRINT_EVERY == 0 and DIFFUSIVE_KILLING:
        max_tox = max(c.species[0] for c in cells.values() if c.cellType == 0)
        print(f"[step {STEP_COUNTER}] max SA toxin_in = {max_tox:.2f}")

def divide(parent, d1, d2):
    """Called when a cell divides; set properties of daughters."""
    ptype = parent.cellType
    d1.cellType = ptype
    d2.cellType = ptype

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
    # if dead somehow divides, we could handle here, but in practice shouldn't

    for d in (d1, d2):
        d.divideFlag = False
        d.deadCounter = 0

# Two-species growth in CellModeller with PA killing SA
# Species 0 ("SA") grows faster; Species 1 ("PA") grows slower.
# Killing is via diffusive toxin using GridDiffusion (no contact killing).
# PA also secretes a second diffusive molecule that inhibits SA growth rate.
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
CARRYING_CAPACITY = MAX_CELLS  # use same scale as your max cells

# RGB colors for rendering in GUI
COL_SA   = [0, 1.0, 0]     # SA = green
COL_PA   = [0, 0, 1.0]     # PA = blue
COL_DEAD = [0.6, 0.6, 0.6]

PRINT_EVERY   = 100   # print every 100 steps
STEP_COUNTER  = 0
DEAD_LIFETIME = 20    # number of steps after which a dead cell is removed

# --------------------------------------------------
# Diffusive toxin parameters (signal 0, species 0)
# --------------------------------------------------
TOXIN_DIFF_RATE        = 50.0  # diffusion coefficient on grid (arbitrary)
TOXIN_MEMBRANE_DIFF    = 10.0  # in/out of cell
TOXIN_PROD_RATE_PA     = 1.0   # production rate in PA cells
TOXIN_KILL_THRESHOLD   = 0.5   # SA dies if extracellular toxin >= this

# Killing toggle (now only diffusive; contact killing removed)
DIFFUSIVE_KILLING = False

# --------------------------------------------------
# Diffusive inhibitor parameters (signal 1, species 1)
# --------------------------------------------------
# Second molecule produced by PA. It does NOT kill SA, but reduces SA growth.
INHIBITOR_ON           = True   # toggle for the second molecule
INHIB_DIFF_RATE        = 50.0   # diffusion coefficient on grid
INHIB_MEMBRANE_DIFF    = 10.0   # in/out of cell
INHIB_PROD_RATE_PA     = 1.0    # production rate in PA cells

# How strongly inhibitor suppresses SA growth:
# effective SA growth = SA_MU * crowd_factor * f(inhib_conc)
# f = max(0, 1 - alpha * inhibitor)
INHIB_EFFECT_STRENGTH  = 1    # per-unit concentration slope

# --------------------------------------------------
# Color switches
# --------------------------------------------------
# If COLOR_BY_INHIBITOR is True, SA color reflects inhibitor (green → yellow).
# If COLOR_BY_INHIBITOR is False but COLOR_BY_TOXIN is True, color reflects toxin (fade to white).
# If both False, use plain species colors.
COLOR_BY_TOXIN     = False
COLOR_BY_INHIBITOR = True   # when True, overrides toxin-based coloring
INHIB_COLOR_REF    = 1/INHIB_EFFECT_STRENGTH     # inhibitor conc at which SA is fully yellow


def inhibitor_growth_factor(inh_conc):
    """
    Map extracellular inhibitor concentration to a multiplicative factor
    on SA growth rate.

    Simple linear inhibition:
        f = max(0, 1 - alpha * inh_conc)

    where alpha = INHIB_EFFECT_STRENGTH.
    """
    if not INHIBITOR_ON:
        return 1.0
    factor = 1.0 - INHIB_EFFECT_STRENGTH * float(inh_conc)
    return max(0.0, factor)


def cell_color(cell):
    """
    Return an [R,G,B] color for a cell based on chosen coloring mode.
    - If dead: gray.
    - If COLOR_BY_INHIBITOR and SA: green → yellow with extracellular inhibitor.
    - Else if COLOR_BY_TOXIN: base color → white with extracellular toxin.
    - Else: species base color.
    """
    ctype = cell.cellType

    # Dead stays gray
    if ctype == 2:
        return COL_DEAD

    # Base species colors
    if ctype == 0:      # SA
        base = COL_SA
    elif ctype == 1:    # PA
        base = COL_PA
    else:
        base = [0.5, 0.5, 0.5]

    # 1) Inhibitor-based coloring (SA only), overrides toxin if enabled
    if COLOR_BY_INHIBITOR and ctype == 0:
        inh = float(cell.signals[1]) if INHIBITOR_ON else 0.0
        if INHIB_COLOR_REF > 0:
            norm = min(inh / INHIB_COLOR_REF, 1.0)
        else:
            norm = 0.0
        # Green → Yellow: [0,1,0] → [1,1,0]
        r = norm            # from 0 to 1
        g = 1.0             # always 1
        b = 0.0
        return [r, g, b]

    # 2) Toxin-based coloring (both species)
    if COLOR_BY_TOXIN and DIFFUSIVE_KILLING:
        tox = float(cell.signals[0])
        if TOXIN_KILL_THRESHOLD > 0:
            norm = min(tox / TOXIN_KILL_THRESHOLD, 1.0)
        else:
            norm = 0.0
        # Blend base → white as toxin increases
        r = base[0] * (1.0 - norm) + 1.0 * norm
        g = base[1] * (1.0 - norm) + 1.0 * norm
        b = base[2] * (1.0 - norm) + 1.0 * norm
        return [r, g, b]

    # 3) Default: plain species colors
    return base


# -----------------------------
# OpenCL reaction kernels for toxin & inhibitor
# -----------------------------
# We have:
#   species[0] = toxin_in
#   species[1] = inhibitor_in
#   signals[0] = toxin_out
#   signals[1] = inhibitor_out

cl_prefix = r'''
    const float D_tox  = %(D_tox).1ff;
    const float k_tox  = %(k_tox).1ff;
    const float D_inh  = %(D_inh).1ff;
    const float k_inh  = %(k_inh).1ff;

    float toxin_in     = species[0];
    float inhibitor_in = species[1];
    float toxin        = signals[0];
    float inhibitor    = signals[1];
''' % {
    'D_tox': TOXIN_MEMBRANE_DIFF,
    'k_tox': TOXIN_PROD_RATE_PA,
    'D_inh': INHIB_MEMBRANE_DIFF,
    'k_inh': INHIB_PROD_RATE_PA,
}

def specRateCL():
    """Intracellular reaction rates (for species[]) in OpenCL."""
    global cl_prefix
    # rates[0] = d(toxin_in)/dt
    # rates[1] = d(inhibitor_in)/dt
    return cl_prefix + r'''
        if (cellType == 1){
            // PA: produce toxin & inhibitor + exchange with outside
            rates[0] = k_tox + D_tox*(toxin - toxin_in)*area/gridVolume;
            rates[1] = k_inh + D_inh*(inhibitor - inhibitor_in)*area/gridVolume;
        } else {
            // SA & others: only exchange with outside
            rates[0] = D_tox*(toxin - toxin_in)*area/gridVolume;
            rates[1] = D_inh*(inhibitor - inhibitor_in)*area/gridVolume;
        }
    '''

def sigRateCL():
    """Extracellular reaction rates (for signals[]) in OpenCL."""
    global cl_prefix
    # rates[0] = d(toxin)/dt
    # rates[1] = d(inhibitor)/dt
    return cl_prefix + r'''
        // Diffusion on the grid handled by GridDiffusion.
        // Here we only handle exchange with cells.
        rates[0] = -D_tox*(toxin - toxin_in)*area/gridVolume;
        rates[1] = -D_inh*(inhibitor - inhibitor_in)*area/gridVolume;
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

    regul = ModuleRegulator(sim, sim.moduleName)

    # ---- Signalling: GridDiffusion for toxin + inhibitor ----
    grid_dim  = (40, 40, 8)         # number of grid points in x,y,z
    grid_size = (8.0, 8.0, 8.0)     # spacing between grid points (must be equal)
    grid_orig = (-160., -160., -16.)
    n_signals = 2                   # toxin + inhibitor
    n_species = 2                   # intracellular toxin + inhibitor

    diff_rates = [TOXIN_DIFF_RATE, INHIB_DIFF_RATE]

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

    # Add renderers
    if sim.is_gui:
        sim.addRenderer(Renderers.GLBacteriumRenderer(sim))
        # sim.addRenderer(Renderers.GLGridRenderer(sig, integ))

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
    # (inhibitor can still slow SA if INHIBITOR_ON is True)
    # ------------------------------------------------------
    if not DIFFUSIVE_KILLING:
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
                inh_out = c.signals[1] if INHIBITOR_ON else 0.0
                inhib_factor = inhibitor_growth_factor(inh_out)
                c.growthRate = SA_MU * crowd_factor * inhib_factor
                c.divideFlag = (c.volume > c.targetVol)
                c.deadCounter = 0
                c.color = cell_color(c)

            elif ctype == 1:  # PA
                c.growthRate = PA_MU * crowd_factor
                c.divideFlag = (c.volume > c.targetVol)
                c.deadCounter = 0
                c.color = COL_PA
                # Optional: also color PA by toxin field
                # c.color = cell_color(c)

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
    # Branch 2: diffusive killing ON (no contact killing)
    # ------------------------------------------------------
    for cid, c in list(cells.items()):
        ctype = c.cellType

        if ctype == 2:  # dead
            c.growthRate = 0.0
            c.divideFlag = False
            c.color = COL_DEAD

            c.deadCounter += 1
            if c.deadCounter >= DEAD_LIFETIME:
                cells_to_remove.append(cid)

        elif ctype == 0:  # SA
            killed = False

            # 1) Diffusive toxin killing using extracellular toxin
            tox_out = c.signals[0]
            if tox_out >= TOXIN_KILL_THRESHOLD:
                c.cellType = 2
                c.growthRate = 0.0
                c.divideFlag = False
                c.color = COL_DEAD
                c.deadCounter = 0
                killed = True

            # 2) If still alive, apply inhibitor-dependent growth slowdown
            if not killed:
                inh_out = c.signals[1] if INHIBITOR_ON else 0.0
                inhib_factor = inhibitor_growth_factor(inh_out)
                c.growthRate = SA_MU * crowd_factor * inhib_factor
                c.divideFlag = (c.volume > c.targetVol)
                c.color = cell_color(c)

        elif ctype == 1:  # PA
            c.growthRate = PA_MU * crowd_factor
            c.divideFlag = (c.volume > c.targetVol)
            c.deadCounter = 0
            c.color = COL_PA
            # c.color = cell_color(c)

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
        max_tox_sa = max(c.species[0] for c in cells.values() if c.cellType == 0)
        max_tox_pa = max(c.species[0] for c in cells.values() if c.cellType == 1)
        max_inh_sa = max(c.species[1] for c in cells.values() if c.cellType == 0)
        max_inh_pa = max(c.species[1] for c in cells.values() if c.cellType == 1)
        print(f"[step {STEP_COUNTER}] max SA toxin_in = {max_tox_sa:.2f}, max PA toxin_in = {max_tox_pa:.2f}, "
              f"max SA inhib_in = {max_inh_sa:.2f}, max PA inhib_in = {max_inh_pa:.2f}")


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

    for d in (d1, d2):
        d.divideFlag = False
        d.deadCounter = 0

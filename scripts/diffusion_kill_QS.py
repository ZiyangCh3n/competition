# Two-species growth in CellModeller with PA killing SA
# Species 0 ("SA") grows faster; Species 1 ("PA") grows slower.
# Killing is via diffusive toxin using GridDiffusion (no contact killing).
# PA also secretes a second diffusive molecule that inhibits SA growth rate.
# Quorum-like behavior:
#   - PA are initially "silent" (no toxin/no inhibitor).
#   - At inhibitor QS threshold: PA start producing inhibitor only.
#   - At toxin QS threshold: PA start producing toxin + inhibitor.
#
# Run with:
#   python CellModeller/Scripts/CellModellerGUI.py /path/to/this_script.py

import random
import numpy as np
from CellModeller.Biophysics.BacterialModels.CLBacterium import CLBacterium
from CellModeller.Regulation.ModuleRegulator import ModuleRegulator
from CellModeller.GUI import Renderers
from CellModeller.Signalling.GridDiffusion import GridDiffusion
from CellModeller.Integration.CLCrankNicIntegrator import CLCrankNicIntegrator

# -----------------------------
# Cell type constants
# -----------------------------
SA_TYPE            = 0
PA_TYPE_ACTIVE     = 1   # PA after toxin QS: produce toxin + inhibitor
DEAD_TYPE          = 2
PA_TYPE_SILENT     = 3   # PA before any QS: no production
PA_TYPE_INHIB_ONLY = 4   # PA after inhibitor QS but before toxin QS: inhibitor only

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

MAX_CELLS = 2500

# Global crowding (simple logistic-like saturation)
CARRYING_CAPACITY = MAX_CELLS*5

# RGB colors
COL_SA        = [0, 1.0, 0]     # SA = green
COL_PA_SILENT = [0, 0, 1.0]     # PA silent = blue
COL_PA_INHIB_ONLY = [1.0, 0.5, 0.0]   # PA inhib-only orange (RGB)
COL_PA_ACTIVE = [1.0, 0, 0]     # PA toxin-producing = red
COL_DEAD      = [0.6, 0.6, 0.6]

PRINT_EVERY   = 100  # print every 100 steps
STEP_COUNTER  = 0
DEAD_LIFETIME = 20 # number of steps after which a dead cell is removed

# --------------------------------------------------
# Diffusive toxin parameters (signal 0, species 0)
# --------------------------------------------------
TOXIN_DIFF_RATE        = 50.0 # diffusion coefficient on grid (arbitrary)
TOXIN_MEMBRANE_DIFF    = 10.0 # in/out of cell
TOXIN_PROD_RATE_PA     = 5.0 # production rate in PA cells
TOXIN_KILL_THRESHOLD   = 1.0   # SA dies if extracellular toxin >= this

DIFFUSIVE_KILLING = True

# --------------------------------------------------
# Diffusive inhibitor parameters (signal 1, species 1)
# --------------------------------------------------

# --- Inhibitor decay (new) ---
# Extracellular decay makes the field local (highly recommended)
INHIBITOR_DECAY_OUT = 0.01    # per-step decay of signals[1] (start 0.005–0.02)

# Optional tiny intracellular decay (usually 0)
INHIBITOR_DECAY_IN  = 0.0     # per-step decay of species[1] (e.g., 0–0.002)


# Second molecule produced by PA. It does NOT kill SA, but reduces SA growth.
INHIBITOR_ON           = True
INHIB_DIFF_RATE        = 100.0
INHIB_MEMBRANE_DIFF    = 40.0
INHIB_PROD_RATE_PA     = 10.0

# SA growth slowdown:
# effective SA growth = SA_MU * crowd_factor * f(inhib_conc)
# f = max(0, 1 - alpha * inhibitor)
INHIB_EFFECT_STRENGTH  = 0.5 # per-unit concentration slope

# --------------------------------------------------
# Metabolic cost of production (new)
# --------------------------------------------------
# Growth penalty for PA when they produce inhibitor and toxin.
# Silent:          PA_MU * crowd_factor
# Inhibitor-only:  PA_MU * crowd_factor * (1 - INHIB_GROWTH_COST)
# Toxin+inhibitor: PA_MU * crowd_factor * (1 - INHIB_GROWTH_COST - TOXIN_GROWTH_COST)
INHIB_GROWTH_COST = 0.2
TOXIN_GROWTH_COST = 0.3

# --------------------------------------------------
# Quorum-sensing-like switches (separate for toxin vs inhibitor)
# --------------------------------------------------
# Toxin QS: when PA start producing toxin (and also inhibitor).
QS_ON_TOXIN            = True
QS_POP_THRESHOLD_TOXIN = 150
QS_ACTIVE_TOXIN        = False  # becomes True when threshold crossed

# Inhibitor QS: when PA start producing inhibitor.
QS_ON_INHIB            = True
QS_POP_THRESHOLD_INHIB = 30
QS_ACTIVE_INHIB        = False  # becomes True when threshold crossed

# --------------------------------------------------
# Color switches
# --------------------------------------------------
# If COLOR_BY_INHIBITOR is True, SA color reflects inhibitor (green → yellow).
# If COLOR_BY_INHIBITOR is False but COLOR_BY_TOXIN is True, color reflects toxin (fade to white).
# If both False, use plain species colors.
COLOR_BY_TOXIN     = False
COLOR_BY_INHIBITOR = True   # when True, overrides toxin-based coloring for SA
INHIB_COLOR_REF    = 1/INHIB_EFFECT_STRENGTH     # inhibitor conc at which SA is fully yellow


def inhibitor_growth_factor(inh_conc):
    """
    Map extracellular inhibitor concentration to a multiplicative factor
    on SA growth rate.

    Simple linear inhibition:
        f = max(0, 1 - alpha * inh_conc)

    Only active if INHIBITOR_ON and inhibitor QS is active.
    """
    if not INHIBITOR_ON or not QS_ACTIVE_INHIB:
        return 1.0
    factor = 1.0 - INHIB_EFFECT_STRENGTH * float(inh_conc)
    return max(0.0, factor)


def pa_growth_factor(ctype):
    """
    Metabolic cost of production for PA:
    - Silent:          no cost
    - Inhibitor-only:  cost = INHIB_GROWTH_COST
    - Toxin+inhibitor: cost = INHIB_GROWTH_COST + TOXIN_GROWTH_COST
    """
    if ctype == PA_TYPE_SILENT:
        return 1.0
    elif ctype == PA_TYPE_INHIB_ONLY:
        return max(0.0, 1.0 - INHIB_GROWTH_COST)
    elif ctype == PA_TYPE_ACTIVE:
        return max(0.0, 1.0 - INHIB_GROWTH_COST - TOXIN_GROWTH_COST)
    else:
        return 1.0


def cell_color(cell):
    """
    Return an [R,G,B] color for a cell based on chosen coloring mode.
    - Dead: gray.
    - PA silent: blue.
    - PA inhibitor-only: orange.
    - PA toxin-producing: red.
    - SA can be recolored by inhibitor (green→yellow) or all cells by toxin (→white).
    """
    ctype = cell.cellType

    if ctype == DEAD_TYPE:
        return COL_DEAD

    # Base species colors
    if ctype == SA_TYPE:
        base = COL_SA

    elif ctype == PA_TYPE_ACTIVE:
        base = COL_PA_ACTIVE    # red

    elif ctype == PA_TYPE_INHIB_ONLY:
        base = COL_PA_INHIB_ONLY   # orange

    elif ctype == PA_TYPE_SILENT:
        base = COL_PA_SILENT       # blue

    else:
        base = [0.5, 0.5, 0.5]

    # # Inhibitor-based coloring for SA (after inhibitor QS)
    # if COLOR_BY_INHIBITOR and ctype == SA_TYPE and QS_ACTIVE_INHIB:
    #     inh = float(cell.signals[1]) if INHIBITOR_ON else 0.0
    #     if INHIB_COLOR_REF > 0:
    #         norm = min(inh / INHIB_COLOR_REF, 1.0)
    #     else:
    #         norm = 0.0
    #     # Green → Yellow: [0,1,0] → [1,1,0]
    #     r = norm
    #     g = 1.0
    #     b = 0.0
    #     return [r, g, b]

    # Inhibitor-based coloring for SA (after inhibitor QS)
    if COLOR_BY_INHIBITOR and ctype == SA_TYPE and QS_ACTIVE_INHIB:
        inh = float(cell.signals[1]) if INHIBITOR_ON else 0.0
        f = inhibitor_growth_factor(inh)  # f in [0,1], same function used for growth
        # Map growth factor to color: full growth (f=1) = green; fully inhibited (f=0) = yellow
        r = 1.0 - f
        g = 1.0
        b = 0.0
        return [r, g, b]


    # Toxin-based coloring (after toxin QS)
    if COLOR_BY_TOXIN and DIFFUSIVE_KILLING and QS_ACTIVE_TOXIN:
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

    return base


# -----------------------------
# OpenCL reaction kernels for toxin & inhibitor
# -----------------------------
# species[0] = toxin_in
# species[1] = inhibitor_in
# signals[0] = toxin_out
# signals[1] = inhibitor_out
#
# IMPORTANT: cellType mapping in CL:
#   SA_TYPE            = 0
#   PA_TYPE_ACTIVE     = 1
#   DEAD_TYPE          = 2
#   PA_TYPE_SILENT     = 3
#   PA_TYPE_INHIB_ONLY = 4

cl_prefix = r'''
    const float D_tox   = %(D_tox).6ff;
    const float k_tox   = %(k_tox).6ff;
    const float D_inh   = %(D_inh).6ff;
    const float k_inh   = %(k_inh).6ff;
    const float dec_inh_out = %(dec_inh_out).6ff;
    const float dec_inh_in  = %(dec_inh_in).6ff;

    float toxin_in     = species[0];
    float inhibitor_in = species[1];
    float toxin        = signals[0];
    float inhibitor    = signals[1];
''' % {
    'D_tox': TOXIN_MEMBRANE_DIFF,
    'k_tox': TOXIN_PROD_RATE_PA,
    'D_inh': INHIB_MEMBRANE_DIFF,
    'k_inh': INHIB_PROD_RATE_PA,
    'dec_inh_out': INHIBITOR_DECAY_OUT,
    'dec_inh_in':  INHIBITOR_DECAY_IN,
}


def specRateCL():
    global cl_prefix
    return cl_prefix + r'''
        if (cellType == 1){
            // PA toxin-active: produce toxin + inhibitor + exchange
            rates[0] = k_tox + D_tox*(toxin - toxin_in)*area/gridVolume;
            rates[1] = k_inh + D_inh*(inhibitor - inhibitor_in)*area/gridVolume
                       - dec_inh_in * inhibitor_in;
        } else if (cellType == 4){
            // PA inhibitor-only: produce inhibitor only, toxin just exchanges
            rates[0] = D_tox*(toxin - toxin_in)*area/gridVolume;
            rates[1] = k_inh + D_inh*(inhibitor - inhibitor_in)*area/gridVolume
                       - dec_inh_in * inhibitor_in;
        } else {
            // SA, DEAD, and SILENT PA: only exchange (+ optional tiny decay)
            rates[0] = D_tox*(toxin - toxin_in)*area/gridVolume;
            rates[1] = D_inh*(inhibitor - inhibitor_in)*area/gridVolume
                       - dec_inh_in * inhibitor_in;
        }
    '''

def sigRateCL():
    global cl_prefix
    return cl_prefix + r'''
        // Exchange with cells + extracellular decay
        rates[0] = -D_tox*(toxin - toxin_in)*area/gridVolume;
        rates[1] = -D_inh*(inhibitor - inhibitor_in)*area/gridVolume
                   - dec_inh_out * inhibitor;
    '''


# -----------------------------
# CellModeller hooks
# -----------------------------
def setup(sim):
    global MAX_CELLS

    biophys = CLBacterium(
        sim,
        jitter_z=False,
        max_cells=MAX_CELLS,
        max_planes=3,
        gamma=10.0,
    )

    regul = ModuleRegulator(sim, sim.moduleName)

    # ---- Signalling: GridDiffusion for toxin + inhibitor ----
    grid_dim  = (80, 80, 8) # number of grid points in x,y,z
    grid_size = (4.0, 4.0, 4.0) # spacing between grid points (must be equal)
    grid_orig = (-160., -160., -16.)
    n_signals = 2 # toxin + inhibitor
    n_species = 2 # intracellular toxin + inhibitor

    diff_rates = [TOXIN_DIFF_RATE, INHIB_DIFF_RATE]

    sig   = GridDiffusion(sim, n_signals, grid_dim, grid_size, grid_orig, diff_rates)
    integ = CLCrankNicIntegrator(sim, n_signals, n_species, MAX_CELLS, sig)

    sim.init(biophys, regul, sig, integ)

    # Seed initial cells for both species near the origin
    rng = random.Random(1)
    # SA
    for _ in range(N_SA_START):
        x = (rng.random()*2 - 1) * INIT_SPREAD
        y = (rng.random()*2 - 1) * INIT_SPREAD
        sim.addCell(
            cellType=SA_TYPE,
            pos=(x, y, 0),
            dir=((rng.random()*2 - 1), (rng.random()*2 - 1), 0),
        )
    # PA start silent
    for _ in range(N_PA_START):
        x = (rng.random()*2 - 1) * (INIT_SPREAD/2)
        y = (rng.random()*2 - 1) * (INIT_SPREAD/2)
        sim.addCell(
            cellType=PA_TYPE_SILENT,
            pos=(x, y, 0),
            dir=((rng.random()*2 - 1), (rng.random()*2 - 1), 0),
        )

    if sim.is_gui:
        sim.addRenderer(Renderers.GLBacteriumRenderer(sim))
        sim.addRenderer(Renderers.GLGridRenderer(sig, integ))

    sim.pickleSteps = 10


def init(cell):
    """Called once when a new cell is created/added."""
    ctype = cell.cellType

    if ctype == SA_TYPE:
        cell.growthRate = SA_MU
        cell.color = COL_SA
        cell.targetVol = DIV_LENGTH_MEAN_SA + random.uniform(0.0, 0.15)

    elif ctype in (PA_TYPE_ACTIVE, PA_TYPE_SILENT, PA_TYPE_INHIB_ONLY):
        cell.growthRate = PA_MU
        cell.color = cell_color(cell)
        cell.targetVol = DIV_LENGTH_MEAN_PA + random.uniform(0.0, 0.5)

    else:  # dead
        cell.growthRate = 0.0
        cell.color = COL_DEAD
        cell.targetVol = 3.0

    cell.divideFlag = False
    cell.deadCounter = 0


def update(cells):
    global STEP_COUNTER, QS_ACTIVE_TOXIN, QS_ACTIVE_INHIB
    STEP_COUNTER += 1

    cells_to_remove = []

    # Global crowding factor (logistic-like slowdown)
    n_cells = len(cells)
    n_pa = sum(
        1 for c in cells.values()
        if c.cellType in (PA_TYPE_ACTIVE, PA_TYPE_SILENT, PA_TYPE_INHIB_ONLY)
    )

    if CARRYING_CAPACITY > 0:
        crowd_factor = max(0.0, 1.0 - float(n_cells) / CARRYING_CAPACITY)
    else:
        crowd_factor = 1.0

    # ----- QS activation of PRODUCTION via PA state switches -----
    if QS_ON_INHIB and (not QS_ACTIVE_INHIB) and (n_pa >= QS_POP_THRESHOLD_INHIB):
        QS_ACTIVE_INHIB = True
        # Silent PA become inhibitor-only
        for c in cells.values():
            if c.cellType == PA_TYPE_SILENT:
                c.cellType = PA_TYPE_INHIB_ONLY

    if QS_ON_TOXIN and (not QS_ACTIVE_TOXIN) and (n_pa >= QS_POP_THRESHOLD_TOXIN):
        QS_ACTIVE_TOXIN = True
        # Any remaining silent or inhib-only PA become fully toxin-active
        for c in cells.values():
            if c.cellType in (PA_TYPE_SILENT, PA_TYPE_INHIB_ONLY):
                c.cellType = PA_TYPE_ACTIVE

    # ------------------------------------------------------
    # Branch 1: no killing, just growth + QS-regulated production
    # ------------------------------------------------------
    if not DIFFUSIVE_KILLING:
        for cid, c in cells.items():
            ctype = c.cellType

            if ctype == DEAD_TYPE:
                c.growthRate = 0.0
                c.divideFlag = False
                c.color = COL_DEAD
                c.deadCounter += 1
                if c.deadCounter >= DEAD_LIFETIME:
                    cells_to_remove.append(cid)

            elif ctype == SA_TYPE:
                inh_out = c.signals[1] if INHIBITOR_ON else 0.0
                inhib_factor = inhibitor_growth_factor(inh_out)
                c.growthRate = SA_MU * crowd_factor * inhib_factor
                c.divideFlag = (c.volume > c.targetVol)
                c.deadCounter = 0
                c.color = cell_color(c)

            elif ctype in (PA_TYPE_ACTIVE, PA_TYPE_SILENT, PA_TYPE_INHIB_ONLY):
                pa_factor = pa_growth_factor(ctype)
                c.growthRate = PA_MU * crowd_factor * pa_factor
                c.divideFlag = (c.volume > c.targetVol)
                c.deadCounter = 0
                c.color = cell_color(c)

        for cid in cells_to_remove:
            cells.pop(cid, None)

        if STEP_COUNTER % PRINT_EVERY == 0:
            n_sa = n_pa = n_dead = 0
            for c in cells.values():
                if c.cellType == SA_TYPE:
                    n_sa += 1
                elif c.cellType in (PA_TYPE_ACTIVE, PA_TYPE_SILENT, PA_TYPE_INHIB_ONLY):
                    n_pa += 1
                elif c.cellType == DEAD_TYPE:
                    n_dead += 1
            total = len(cells)
            print(f"[step {STEP_COUNTER}] SA={n_sa}, PA={n_pa}, dead={n_dead}, total={total}, "
                  f"QS_T={QS_ACTIVE_TOXIN}, QS_I={QS_ACTIVE_INHIB}")
        return

    # ------------------------------------------------------
    # Branch 2: diffusive killing ON
    # ------------------------------------------------------
    for cid, c in list(cells.items()):
        ctype = c.cellType

        if ctype == DEAD_TYPE:
            c.growthRate = 0.0
            c.divideFlag = False
            c.color = COL_DEAD
            c.deadCounter += 1
            if c.deadCounter >= DEAD_LIFETIME:
                cells_to_remove.append(cid)

        elif ctype == SA_TYPE:
            killed = False

            # 1) Diffusive toxin killing using extracellular toxin
            tox_out = c.signals[0]
            if tox_out >= TOXIN_KILL_THRESHOLD:
                c.cellType = DEAD_TYPE
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

        elif ctype in (PA_TYPE_ACTIVE, PA_TYPE_SILENT, PA_TYPE_INHIB_ONLY):
            pa_factor = pa_growth_factor(ctype)
            c.growthRate = PA_MU * crowd_factor * pa_factor
            c.divideFlag = (c.volume > c.targetVol)
            c.deadCounter = 0
            c.color = cell_color(c)

    for cid in cells_to_remove:
        cells.pop(cid, None)

    if STEP_COUNTER % PRINT_EVERY == 0:
        n_sa = n_pa = n_dead = 0
        for c in cells.values():
            if c.cellType == SA_TYPE:
                n_sa += 1
            elif c.cellType in (PA_TYPE_ACTIVE, PA_TYPE_SILENT, PA_TYPE_INHIB_ONLY):
                n_pa += 1
            elif c.cellType == DEAD_TYPE:
                n_dead += 1
        total = len(cells)
        print(f"!!!!![step {STEP_COUNTER}] SA={n_sa}, PA={n_pa}, dead={n_dead}, total={total}, "
              f"QS_T={QS_ACTIVE_TOXIN}, QS_I={QS_ACTIVE_INHIB}")

    if STEP_COUNTER % PRINT_EVERY == 0 and DIFFUSIVE_KILLING:
        max_tox_sa = max(c.species[0] for c in cells.values() if c.cellType == SA_TYPE)
        max_tox_pa = max(c.species[0] for c in cells.values()
                         if c.cellType in (PA_TYPE_ACTIVE, PA_TYPE_SILENT, PA_TYPE_INHIB_ONLY))
        max_inh_sa = max(c.species[1] for c in cells.values() if c.cellType == SA_TYPE)
        max_inh_pa = max(c.species[1] for c in cells.values()
                         if c.cellType in (PA_TYPE_ACTIVE, PA_TYPE_SILENT, PA_TYPE_INHIB_ONLY))
        print(f"[step {STEP_COUNTER}] max SA toxin_in = {max_tox_sa:.2f}, max PA toxin_in = {max_tox_pa:.2f}, "
              f"max SA inhib_in = {max_inh_sa:.2f}, max PA inhib_in = {max_inh_pa:.2f}")
        diffs = []
        for c in cells.values():
            if c.cellType in (PA_TYPE_ACTIVE, PA_TYPE_INHIB_ONLY, SA_TYPE):
                diffs.append(abs(float(c.species[1]) - float(c.signals[1])))
        if diffs:
            print(f"[step {STEP_COUNTER}] mean |in-inh - out-inh| = {np.mean(diffs):.3g} (should be ~0 when exchange is fast)")



def divide(parent, d1, d2):
    """Called when a cell divides; daughters inherit parent's PA state."""
    ptype = parent.cellType

    d1.cellType = ptype
    d2.cellType = ptype

    if ptype == SA_TYPE:
        for d in (d1, d2):
            d.color = cell_color(d)
            d.growthRate = SA_MU
            d.targetVol = DIV_LENGTH_MEAN_SA + random.uniform(0.0, 0.15)
    elif ptype in (PA_TYPE_ACTIVE, PA_TYPE_SILENT, PA_TYPE_INHIB_ONLY):
        for d in (d1, d2):
            d.color = cell_color(d)
            d.growthRate = PA_MU
            d.targetVol = DIV_LENGTH_MEAN_PA + random.uniform(0.0, 0.5)

    for d in (d1, d2):
        d.divideFlag = False
        d.deadCounter = 0

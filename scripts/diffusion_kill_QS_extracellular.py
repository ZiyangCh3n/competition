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
N_SA_START   = 3
N_PA_START   = 1
INIT_SPREAD  = 25.0  # microns around origin for initial seeding

SA_MU = 1.8   # "SA" base growth rate (fast)
PA_MU = 0.6   # "PA" base growth rate (slow)

DIV_LENGTH_MEAN_PA = 3.5   # mean target length for division
DIV_LENGTH_MEAN_SA = 1.0
DIV_LENGTH_JITTER  = 0.6   # random jitter added to target length

MAX_CELLS = 2500

# Global crowding (simple logistic-like saturation)
CARRYING_CAPACITY = MAX_CELLS*5

# RGB colors
COL_SA           = [0, 1.0, 0]     # SA = green
COL_PA_SILENT    = [0, 0, 1.0]     # PA silent = blue
COL_PA_INHIB_ONLY= [1.0, 0.5, 0.0] # PA inhib-only orange
COL_PA_ACTIVE    = [1.0, 0, 0]     # PA toxin-producing = red
COL_DEAD         = [0.6, 0.6, 0.6]

PRINT_EVERY   = 100  # print every 100 steps
STEP_COUNTER  = 0
DEAD_LIFETIME = 20   # steps after which a dead cell is removed

# --------------------------------------------------
# Extracellular toxin (signal 0)
# --------------------------------------------------
TOXIN_DIFF_RATE      = 50.0     # grid diffusion
TOXIN_PROD_RATE_PA   = 5.0      # PA_ACTIVE secretes outside
TOXIN_DECAY_OUT      = 0.00     # optional extracellular decay (0–0.02)
TOXIN_KILL_THRESHOLD = 1.0      # SA dies if extracellular toxin >= this

DIFFUSIVE_KILLING = True

# --------------------------------------------------
# Extracellular inhibitor (signal 1)
# --------------------------------------------------
INHIBITOR_ON         = True
INHIB_DIFF_RATE      = 100.0   # grid diffusion
INHIB_PROD_RATE_PA   = 10.0    # PA_ACTIVE & PA_INHIB_ONLY secrete outside
INHIBITOR_DECAY_OUT  = 0.01    # extracellular decay (0.005–0.02 recommended)

# SA growth slowdown:
# effective SA growth = SA_MU * crowd_factor * f(inhib_conc)
# f = max(0, 1 - alpha * inhibitor)
INHIB_EFFECT_STRENGTH = 0.5     # per-unit concentration slope

# --------------------------------------------------
# Metabolic cost of production (for PA growth)
# --------------------------------------------------
INHIB_GROWTH_COST = 0.2
TOXIN_GROWTH_COST = 0.3

# --------------------------------------------------
# Quorum-sensing-like switches (separate for toxin vs inhibitor)
# --------------------------------------------------
# NOTE: QS still gates PRODUCTION, but NOT the SA response (effect).
QS_ON_TOXIN            = True
QS_POP_THRESHOLD_TOXIN = 150
QS_ACTIVE_TOXIN        = False  # becomes True when threshold crossed

QS_ON_INHIB            = True
QS_POP_THRESHOLD_INHIB = 30
QS_ACTIVE_INHIB        = False  # becomes True when threshold crossed

# --------------------------------------------------
# Color switches
# --------------------------------------------------
# If COLOR_BY_INHIBITOR is True, SA color reflects growth effect (green → yellow).
COLOR_BY_TOXIN     = False
COLOR_BY_INHIBITOR = True

# -----------------------------
# Effect functions
# -----------------------------
def inhibitor_growth_factor(inh_conc):
    """
    Map extracellular inhibitor concentration to a multiplicative factor
    on SA growth rate: f = max(0, 1 - alpha * inh_conc)

    IMPORTANT CHANGE:
    - We NO LONGER gate the effect by QS_ACTIVE_INHIB.
      SA always respond to the actual inhibitor concentration.
    """
    if not INHIBITOR_ON:
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
    Return [R,G,B] based on chosen coloring mode.
    - Dead: gray; PA silent: blue; PA inhib-only: orange; PA active: red.
    - SA can be recolored by inhibitor growth effect (green→yellow).
    """
    ctype = cell.cellType
    if ctype == DEAD_TYPE:
        return COL_DEAD

    if ctype == SA_TYPE:
        base = COL_SA
    elif ctype == PA_TYPE_ACTIVE:
        base = COL_PA_ACTIVE
    elif ctype == PA_TYPE_INHIB_ONLY:
        base = COL_PA_INHIB_ONLY
    elif ctype == PA_TYPE_SILENT:
        base = COL_PA_SILENT
    else:
        base = [0.5, 0.5, 0.5]

    # SA coloring by *growth effect* from inhibitor (matches phenotype)
    if COLOR_BY_INHIBITOR and ctype == SA_TYPE:
        inh = float(cell.signals[1]) if INHIBITOR_ON else 0.0
        f = inhibitor_growth_factor(inh)  # 1→green, 0→yellow
        r = 1.0 - f
        g = 1.0
        b = 0.0
        return [r, g, b]

    # Optional: toxin-based whitening (off by default)
    if COLOR_BY_TOXIN and DIFFUSIVE_KILLING and QS_ACTIVE_TOXIN:
        tox = float(cell.signals[0])
        norm = min(tox / TOXIN_KILL_THRESHOLD, 1.0) if TOXIN_KILL_THRESHOLD > 0 else 0.0
        r = base[0] * (1.0 - norm) + 1.0 * norm
        g = base[1] * (1.0 - norm) + 1.0 * norm
        b = base[2] * (1.0 - norm) + 1.0 * norm
        return [r, g, b]

    return base

# -----------------------------
# OpenCL reaction kernels (EXTRACELLULAR ONLY)
# -----------------------------
# signals[0] = toxin_out
# signals[1] = inhibitor_out
#
# cellType mapping:
#   SA_TYPE            = 0
#   PA_TYPE_ACTIVE     = 1
#   DEAD_TYPE          = 2
#   PA_TYPE_SILENT     = 3
#   PA_TYPE_INHIB_ONLY = 4
#
# Diffusion on the grid is handled by GridDiffusion; here we only add
# extracellular decay and PA secretion (no intracellular species).

cl_prefix = r'''
    const float k_tox   = %(k_tox).6ff;
    const float k_inh   = %(k_inh).6ff;
    const float dec_tox = %(dec_tox).6ff;
    const float dec_inh = %(dec_inh).6ff;

    float toxin     = signals[0];
    float inhibitor = signals[1];
''' % {
    'k_tox':   TOXIN_PROD_RATE_PA,
    'k_inh':   INHIB_PROD_RATE_PA,
    'dec_tox': TOXIN_DECAY_OUT,
    'dec_inh': INHIBITOR_DECAY_OUT,
}

def specRateCL():
    """No intracellular species: return a do-nothing kernel body."""
    global cl_prefix
    return cl_prefix + r'''
        // No intracellular pools; nothing to do here. (n_species == 0)
    '''

def sigRateCL():
    """Extracellular reactions: decay + PA secretion (no membrane exchange)."""
    global cl_prefix
    return cl_prefix + r'''
        // Base decay
        float r_tox = - dec_tox * toxin;
        float r_inh = - dec_inh * inhibitor;

        // Secretion:
        // - Toxin: only PA_TYPE_ACTIVE (1) secretes
        // - Inhibitor: PA_TYPE_ACTIVE (1) and PA_TYPE_INHIB_ONLY (4) secrete
        if (cellType == 1){
            r_tox += k_tox;
            r_inh += k_inh;
        } else if (cellType == 4){
            r_inh += k_inh;
        }

        rates[0] = r_tox;
        rates[1] = r_inh;
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
    grid_dim  = (80, 80, 8)      # voxels in x,y,z
    grid_size = (4.0, 4.0, 4.0)  # micron spacing (equal in x,y,z)
    grid_orig = (-160., -160., -16.)
    n_signals = 2                # toxin + inhibitor
    n_species = 0                # *** no intracellular species ***

    # Grid diffusion coefficients for signals (order matches signals[])
    diff_rates = [TOXIN_DIFF_RATE, INHIB_DIFF_RATE]

    sig   = GridDiffusion(sim, n_signals, grid_dim, grid_size, grid_orig, diff_rates)
    integ = CLCrankNicIntegrator(sim, n_signals, n_species, MAX_CELLS, sig)

    # Optional: smaller dt if you crank rates; default is usually ~0.02
    # sim.dt = 0.01

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
    crowd_factor = max(0.0, 1.0 - float(n_cells) / CARRYING_CAPACITY) if CARRYING_CAPACITY > 0 else 1.0

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
                inh_out = c.signals[1] if INHIBITOR_ON else 0.0
                f = inhibitor_growth_factor(inh_out)
                c.growthRate = SA_MU * crowd_factor * f
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
                if c.cellType == SA_TYPE: n_sa += 1
                elif c.cellType in (PA_TYPE_ACTIVE, PA_TYPE_SILENT, PA_TYPE_INHIB_ONLY): n_pa += 1
                elif c.cellType == DEAD_TYPE: n_dead += 1
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
                f = inhibitor_growth_factor(inh_out)
                c.growthRate = SA_MU * crowd_factor * f
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
            if c.cellType == SA_TYPE: n_sa += 1
            elif c.cellType in (PA_TYPE_ACTIVE, PA_TYPE_SILENT, PA_TYPE_INHIB_ONLY): n_pa += 1
            elif c.cellType == DEAD_TYPE: n_dead += 1
        total = len(cells)
        print(f"!!!!![step {STEP_COUNTER}] SA={n_sa}, PA={n_pa}, dead={n_dead}, total={total}, "
              f"QS_T={QS_ACTIVE_TOXIN}, QS_I={QS_ACTIVE_INHIB}")

def divide(parent, d1, d2):
    """Called when a cell divides; daughters inherit parent's PA state."""
    ptype = parent.cellType

    d1.cellType = ptype
    d2.cellType = ptype

    if ptype == SA_TYPE:
        for d in (d1, d2):
            d.color = cell_color(d)  # recolor immediately
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

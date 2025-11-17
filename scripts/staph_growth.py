import random
from CellModeller.Regulation.ModuleRegulator import ModuleRegulator
from CellModeller.Biophysics.BacterialModels.CLBacterium import CLBacterium
from CellModeller.GUI import Renderers
import numpy
import math

max_cells = 2**15

def setup(sim):
    biophys = CLBacterium(sim, jitter_z=False, max_cells=max_cells)
    regul = ModuleRegulator(sim, sim.moduleName)
    sim.init(biophys, regul, None, None)

    # Start with one cell
    sim.addCell(cellType=0, pos=(0, 0, 0), dir=(1, 0, 0))

    if sim.is_gui:
        sim.addRenderer(Renderers.GLBacteriumRenderer(sim))

    sim.pickleSteps = 100


def init(cell):
    cell.targetVol = 1 + random.uniform(0.0, 0.5)
    cell.growthRate = 2.0


def update(cells):
    for (cid, cell) in cells.items():
        cell.color = [cell.cellType*0.6+0.1, 1.0-cell.cellType*0.6, 0.3]
        if cell.volume > cell.targetVol:
            cell.divideFlag = True
            x,y,z = cell.dir[0], cell.dir[1], cell.dir[2]
            cell.dir = [-y, x, z]


def divide(parent, d1, d2):
    # new targets
    d1.targetVol = 1 + random.uniform(0.0, 0.5)
    d2.targetVol = 1 + random.uniform(0.0, 0.5)

    # # pdir = numpy.asarray(parent.dir, dtype=float)
    # perp = numpy.array([-parent.dir[1], parent.dir[0], 0.0])

    # d1.dir = perp.copy()
    # d2.dir = perp.copy()

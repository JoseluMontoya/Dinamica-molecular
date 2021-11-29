import numpy as np

from .lattice import lattice
from .verlet import verlet_solve
from .force import giveSpeed

class dynamicLattice(lattice):
    '''
    Objeto de tipo lattice, con atributo de la mása de las partículas, y
    métodos para generar un array de velocidades iniciales y evolucionar el
    sistema utilizando verlet
    '''
    def __init__(self, latt, a, m):
        super().__init__(latt, a)
        self.m = m
    
    def gen_v0(self, T):
        self.v0 = giveSpeed(len(self.r0), self.m, T)

    def evolveSystem(self, f, dt, N, r0=0, v0=0):
        if (type(r0) != np.ndarray) and hasattr(self, "r0"):
            r0 = self.r0
        if (type(v0) == np.ndarray) and hasattr(self, "v0"):
            v0 = self.v0
        r, v, t = verlet_solve(f, r0, v0, dt, N)
        self.r = r
        self.v = v
        self.t = t
        return r, v, t
        

from functools import partial
import numpy as np
import ast

from Librerias.lattice import lattice
from Librerias.energy import V_LennardJones, energyLattice, energyLatticePeriodic

e = 0.167 # eV
sigma = 2.3151 # Angstrom
n = 12
m = 6
a = 3.603 # Angstrom

V = partial(V_LennardJones, e, sigma, n, m)
FCC = lattice("FCC", a)
rc = 3*sigma

dim = [5, 5, 5]
coord = FCC.gen_lattice(dim)

E = energyLattice(V, rc, coord)
print("El valor de la energía por átomo es: {0} eV ".format(
    np.sum(E)/(len(coord))))

FCC.plot(coord, E=E)
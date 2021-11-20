import numpy as np
import ast
from functools import partial

from Librerias.helpers import r, getT
from Librerias.lattice import lattice
from Librerias.force import F_LennardJones, forceLattice, giveSpeed

e = 0.167 # eV
sigma = 2.3151 # Angstrom
n = 12
m = 6
a = 3.603 # Angstrom

Fr = partial(F_LennardJones, e, sigma, n, m)
FCC = lattice("FCC", a)
rc = 3*sigma

dim = [5, 5, 5]
r0 = FCC.gen_lattice(dim)

F = partial(forceLattice, Fr, rc) 
F0 = F(r0)
FCC.plot(r0, F=F0, onlyF=True)

# Extra 1
Temp = 300
m = 63.55*931.494061e6
T = partial(getT, m)
v0, Temp = giveSpeed(len(r0), m, Temp)
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from Librerias.dynamicLattice import dynamicLattice
from Librerias.helpers import kinetic
from Librerias.force import F_LennardJones, forceLattice
from Librerias.energy import V_LennardJones, energyLattice


e = 0.167 # eV
sigma = 2.3151 # Angstrom
n = 12
m = 6
rc = 3*sigma

dt = 1e-17
mass = 63.546*931.494061e6/9e36 #eV/(Angstrom/s)^2

dim = [2, 2, 2]
a = 3.603 # Angstrom
Temperatura = 100


# Preparación de funciones

Fr = partial(F_LennardJones, e, sigma, n, m)
F = partial(forceLattice, Fr, rc) 
def aceleracion(r, t): return F(r)/mass

# Generación de la red y aplicación de Verlet

FCC = dynamicLattice("FCC", a, mass)
FCC.gen_lattice(dim)
FCC.plot(FCC.r0)
FCC.gen_v0(Temperatura)
FCC.evolveSystem(aceleracion, dt, 500000)

# Plot

V = partial(V_LennardJones, e, sigma, n, m)
T = partial(kinetic, mass)
U = partial(energyLattice,V, rc)

EKinetic = np.sum(list(map(T, FCC.v)), axis=-1)
EPotential = np.sum(list(map(U, FCC.r)), axis=-1)

plt.plot(FCC.t, EKinetic, label="Kinetic")
plt.plot(FCC.t, EPotential, label="Potential")
plt.plot(FCC.t, EKinetic + EPotential, label="Total")
plt.legend()
plt.show()


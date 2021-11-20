from functools import partial
import matplotlib.pyplot as plt

from Librerias.velet import verlet_solve
from Librerias.helpers import f2, U2, kinetic
from Librerias.helpers import r as R


G = 6.6738e-11 # m3 kg-1 s-2
M = 1.9891e30 # kg
m = 5.972e24 # kg
N = 24*365*5 # dias
dt = 60*60 # s
r0 = (-1.4719e11, 0) # m
v0 = (0, -3.0287e4) # m s-1

f = partial(f2, G, M)
U = partial(U2, G, M, m)
T = partial(kinetic, m)
r, v, t = verlet_solve(f, r0, v0, dt, N)
t = t/dt

# Parte 1
plt.plot(t, R(r), label="$r$")
plt.xlabel("Tiempo ($h$)")
plt.ylabel("Radio ($m$)")
plt.legend()
plt.show()

# Parte 2
plt.plot(r.T[0][0], r.T[1][0], "ro", label="Origen")
plt.plot(r.T[0], r.T[1], linewidth=0.8, label="Órbita")
plt.title("Órbita")
plt.xlabel("$x$ ($m$)")
plt.ylabel("$y$ ($m$)")
plt.legend()
plt.show()

# Parte 3
plt.plot(t, T(v), label="$T$")
plt.plot(t, U(r), label="$U$")
plt.plot(t, T(v) + U(r), label="$U+T$")
plt.xlabel("Tiempo ($h$)")
plt.ylabel("Energía ($J$)")
plt.legend()
plt.show()
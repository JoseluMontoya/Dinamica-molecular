import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from Librerias.helpers import giveSpeed
from Librerias.lattice import lattice
from Librerias.verletCy import verlet_solve


FCC = lattice("FCC", 3.603)
dim = (7, 3, 3)
e = 0.167
sigma = 2.3151
rc = 3*sigma
m = 1.05e-25/16
Temp = 600
dt = 1e-15
N = 5000
r0 = FCC.gen_lattice(dim)
v0 = giveSpeed(len(r0), m, Temp)
aComp = np.zeros((len(r0), 3), dtype=np.float)
aRot = np.zeros((len(r0), 3), dtype=np.float)
w = np.zeros((len(r0), 3), dtype=np.float)

rxmax = np.max(r0.T[0])
rymax = np.max(r0.T[1])
rzmax = np.max(r0.T[2])


# aComp.T[0][r0.T[0] < rxmax*2/7] = 9e24
# aComp.T[0][rxmax*5/7 < r0.T[0]] = -9e24

aRot.T[0][r0.T[0] == 0] = -5e25
# aRot.T[0][r0.T[0] == rxmax] = 5e25
w.T[0][r0.T[0] == 0] = 1
# w.T[0][r0.T[0] == rxmax] = 1


t0 = time.time()
r, v, U = verlet_solve(e, sigma, m, r0, v0, dt, N, aComp, aRot, w)
t = time.time() - t0
T = 1/2*m*np.sum(np.linalg.norm(v, axis=-1)**2, axis=-1)
tiempo = np.arange(0, N*dt, dt)
print("#==")
print("Tiempo con red de {0} dimensiones y con {1} pasos: {2} s (Normal)".format(dim, N, round(t, 4)))
print("#==")

plt.plot(tiempo, U, label='Potencial')
plt.plot(tiempo, T, label='Cinética')
plt.plot(tiempo, U+T, label='Total')
plt.legend()
plt.show()

fig, ax, imagen, titulo = FCC.plot(r0, show=False)
skip = 1
def animate(i):
    imagen._offsets3d = r[skip*i].T
    titulo.set_text("Evolución red cristalina   ${} \cdot 10^{{-15}} s$".format(skip*i))
    return imagen, titulo
animacion = animation.FuncAnimation(fig,animate,frames = int(N/skip)
                                    ,repeat = True, interval = 1)
plt.show()
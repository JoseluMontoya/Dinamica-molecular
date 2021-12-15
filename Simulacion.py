import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from Librerias.helpers import giveSpeed
from Librerias.lattice import lattice
from Librerias.verletCy import verlet_solve
from Librerias.helpers import set_axes_equal


a = 3.603
FCC = lattice("FCC", a)
dim = (3, 7, 3)
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

rmax = np.max(r0.T, axis=1)


aComp.T[1][r0.T[1] < rmax[1]*2/7] = -9e24
aComp.T[1][rmax[1]*5/7 < r0.T[1]] = 9e24

# aRot.T[1][r0.T[1] == 0] = -5e25
# # aRot.T[1][r0.T[1] == rmax[1]] = 5e25
# w.T[1][r0.T[1] == 0] = 1
# # w.T[1][r0.T[1] == rmax[1]] = 1


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

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.view_init(0, 0)
titulo = ax.set_title('Evolución red cristalina   $0 \cdot 10^{{-15}} s$')
imagen, = ax.plot(r0.T[0], r0.T[1], r0.T[2], linestyle="", marker="o")
set_axes_equal(ax)

skip = 5
framerate = 120
def animate(i):
    imagen.set_data(r[skip*i].T[0], r[skip*i].T[1])
    imagen.set_3d_properties(r[skip*i].T[2])
    titulo.set_text("Evolución red cristalina   ${} \cdot 10^{{-15}} s$".format(skip*i))
    return titulo, imagen
animacion = animation.FuncAnimation(fig,animate,frames = int(N/skip)
                                    ,repeat = True, interval = 1000/framerate)
animacion.save("animacion.mp4", writer=animation.FFMpegWriter(fps=framerate))
plt.show()
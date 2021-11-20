from matplotlib.pyplot import axis
import numpy as np
from numpy.linalg import norm

import warnings
warnings.filterwarnings("ignore")


def r(vec):
    """
    Funcion que calcula el módulo de los vectores en el último eje del array vec 
    """
    return norm(vec, axis=-1)


def f2(G, M, r, t):
    """
    Función f correspontiente al ejercício 2 para el problema de Verlet
    """
    return -G*M*r/norm(r)**3


def U2(G, M, mEarth, r):
    """
    Función de energía potencial del ejercício 2
    """
    return G*M*mEarth/norm(r, axis=-1)**2

def f3(x):
    """
    Función f correspontiente al ejercício 3 para la integración con Montecarlo
    """
    return np.sin(1/(x*(2 - x)))**2


def f3extra(x):
    """
    Función f correspontiente al ejercício 3 para la integración con Montecarlo
    """
    return np.exp(np.sin(x*1000))*x


def kinetic(m, v):
    """
    Funcion que calcula la energía cinética de cada partícula en un conjunto con
    velocidad v y masa m
    """
    return 1/2 * m * r(v)**2


def getT(m, v):
    """
    Función que calcula la temperatura de un grupo de partículas de masa m y
    velociades v
    """
    kb = 8.6181024e-5 #eV/K
    return np.sum(r(v)**2, axis=len(np.shape(v)) - 2)*m/(3*kb*np.shape(v)[-2])


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
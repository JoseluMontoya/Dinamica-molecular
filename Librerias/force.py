import numpy as np
import numpy.random as rnd
from .helpers import r, getT


def F_LennardJones(e, sigma, n, m, coord):
    originalShape = np.shape(coord)
    coord = np.reshape(coord, (np.prod(np.shape(coord)[:2]), 3))
    F = (4*e*
         (m*(sigma/r(coord))**m - n*(sigma/r(coord))**n)/r(coord)**2
         )[..., None]*coord
    return np.reshape(F, originalShape)


def forceLattice(F, rc, red):
    '''
    Función que calcula la fuerza que sufre cada átomo de la red , con una
    fuerza entre átomos Fr y con un cutoff rc

        Parámetros
        ----------
        F: func
            Función de fuerza dependiente de la distancia r
        rc: int o float
            Radio a partir del cual cortar la interacción con V
        red: ndarray
            Array con las posiciones de los átomos

        Devuelve
        --------
        ndarray
            Array con los valores de la fuerza que experimenta cada átomo
    '''
    N = len(red)
    
    # Preparamos red y coord para generar un array con todas las posibles
    # distancias interatómicas de la red
    coordA = np.reshape(np.repeat(red, N,axis=0), (N, N, 3))
    coordB = np.reshape(np.tile(red, (N, 1)), (N, N, 3))

    # Generamos el array de todas las posibles distancias
    R = coordB - coordA
    # Convertimos en 0 las fuerzas que ocurren a un r mayor o igual a rc
    R = np.where(np.reshape(r(R) < rc, (N, N, 1)), R, 0)
    FTotal = np.nansum(F(R), axis=1)
    return FTotal


def giveSpeed(N, m, T="Random"):
    '''
    Da una velocidad aleatória (angstroms por segundo) a N partículas de
    masa m (eV/c^2) tal que tengan una temperatura T (K)
    '''
    v = rnd.uniform(-0.5, 0.5, (N, 3))
    Trand = getT(m, v)
    if type(T) == float or type(T) == int:
        sc = np.sqrt(T/Trand)
        v = sc*v
        Trand = getT(m, v)
    return v*3e18
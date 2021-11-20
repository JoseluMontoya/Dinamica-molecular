import numpy as np
from .helpers import r


def V_LennardJones(e, sigma, n, m, r):
    return 4*e*((sigma/r)**n - (sigma/r)**m)


def energyLattice(V, rc, coord):
    '''
    Función que calcula la energía potencial de cada átomo en las posiciones
    coord, con un potencial de interacción V y con un cutoff rc

        Parámetros
        ----------
        V: func
            Potencial de interacción con simetría esférica
        rc: int o float
            Radio a partir del cual cortar la interacción con V
        coord: ndarray
            Array con las posiciones de todos los vectores

        Devuelve
        --------
        ndarray
            Array con los valores de energía de cada partícula
    '''
    N = len(coord)
    
    # Preparamos red y coord para generar un array con todas las posibles
    # distancias interatómicas de la red
    coordA = np.reshape(np.repeat(coord, N, axis=0), (N, N, 3))
    coordB = np.reshape(np.tile(coord, (N, 1)), (N, N, 3))

    # Generamos el array de todas las posibles distancias
    R = r(coordB - coordA)
    # Convertimos en 0 las energias debidas a una distancia mayor o iguales
    # que rc
    R = np.where(R < rc, R, 0)
    E = np.nansum(V(R), axis=1)/2
    return E
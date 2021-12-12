# -*- coding: utf-8 -*-

""" Este script incluye los modulos necesarios para generar redes cristalinas (en funcion de una celda base) y 
        calcuar las interacciones entre los átomos que las componen, ademas de crear un archivo de texto donde 
        se almacenan los datos calculados: numero de atomos y posicion de cada uno.

        @authors: 
              Guillem Pellicer Chover: nusgrem.guillem@protonmail.com ===> github: @Guillem1999
              Vicente Lillo Poveda : [insert email here] ===> github: [insert github here]
              Jose Luis Montoya [insert surname here]: [insert email here] ===> github: @JoseluMontoya
"""

# Importamos las librerias necesarias:

import numpy as np
from numpy.linalg import norm

''' Para evitar utilizar "warnings", que puede dar problemas a la hora de compilar
    el módulo, se ha decidido evitar las divisiones por 0 cambiando dichos valores
    por np.inf, a pesar de que esta implementación es mas lenta que la anterior, con
    la esperanza de que esto ahorre problemas posteriormente.
    
    PD: Al sustituir np.nansum por np.sum, el rendimiento se incrementa'''
# import warnings
# warnings.filterwarnings("ignore") # buscar no

# ============ · Funciones auxiliares para simplificar el cálculo · ==============

def r(vec):                 return norm(vec, axis=-1) 

def F(G, M, r, t):          return -G*M*r/norm(r)**3 

def UGrav(G, M, mEarth, r): return -G*M*mEarth/norm(r, axis=-1)

def kinetic(m, v):          return 1/2 * m * r(v)**2

def getT(m, v):
    kb = 8.6181024e-5 #eV/K
    return np.sum(r(v)**2, axis=len(np.shape(v)) - 2)*m/(3*kb*np.shape(v)[-2])

# ============ · Implementación de Verlet · ===============

def verlet_step(i,t,r,v,vint,h,f):
    r[i+1] = r[i] + h*vint[i]
    k = h*f(r[i+1], t[i+1])
    v[i+1] = vint[i] + 1/2*k
    vint[i+1] = vint[i] + k

def verlet_solve(f, r0, v0, h, N):
    '''
    Función que resuelve la ecuación diferencial
                dv/dt = f(r, t)
    con v = dr/dt, usando el método de Verlet

        Parámetros
        ----------
        f: func
            Función f de la ecuación diferencial
        r0, v0: ndarray
            Valores iniciales para r y v
        h: int o float
            Tamaño de paso temporal
        N: int
            Número de pasos temporales

        Devuelve
        --------
        tuple
            Tupla con los arrays solución r, v y t
    '''
    # Inicializamos los arrays
    r    = np.empty(((N+1,)+np.shape((r0))))
    v    = np.empty(np.shape(r))
    vint = np.empty(np.shape(r))
    t    = np.arange(0,(N+1)*h,h)
    # Definimos las condiciones iniciales
    r[0] = np.array(r0)
    v[0] = np.array(v0)
    vint[0] = v[0] + 1/2*h*f(r[0], t[0])
    # Iteramos para obtener la evolución del sistema
    for i in range(N): verlet_step(i,t,r,v,vint,h,f) 
    
    return r, v, t


# ============ · Potencial y fuerza segun Lennard Jones · ===============

def F_LennardJones(e, s, n, m, coord):
    r = norm(coord, axis=-1)
    r[r == 0] = np.inf # Esto es para evitar las divisiones por 0 y así no utilizar "warnings"
    return (4*e*(m*(s/r)**m - n*(s/r)**n)/r**2)[..., None]*coord

def V_LennardJones(e, s, n, m, r): 
    r[r == 0] = np.inf # Esto es para evitar las divisiones por 0 y así no utilizar "warnings"
    return 4*e*((s/r)**n - (s/r)**m)

# ============ · Calculo de la fuerza entre partículas, así como su energia · =============== 

''' Es posible combinar las dos funciones siguientes, cosa que optimizaría el cálculo
    en el caso de necesitar F y E, paralelizandolo, pero supondria un costo computacional
    extra si solo se requiere uno de dichos cálculos'''

def forceLattice(F, rc,red):
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
    FTotal = np.sum(F(R), axis=1)
    return FTotal

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
    E = np.sum(V(R), axis=1)/2
    return E

# ============================= · energy.py · ================================================

def giveSpeed(N, m, T=False): # Si T = False, tenemos velocidades aleatorias, si no, calculamos las velocidades en funcion de la T dada.
    '''
    Da una velocidad aleatória (angstroms por segundo) a N partículas de
    masa m (eV/c^2) tal que tengan una temperatura T (K)
    '''
    v = np.random.uniform(-0.5, 0.5, (N, 3))
    Trand = getT(m, v)
    if type(T) == float or type(T) == int:
        sc = np.sqrt(T/Trand)
        v = sc*v
        Trand = getT(m, v)
    return v*3e18

# ============================= · Fin del módulo · ================================================

if __name__ == '__main__': pass # Igual hay que eliminar esta línea antes de compilar
        
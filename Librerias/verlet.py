import numpy as np


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

    r = np.empty(((N+1,)+np.shape((r0))))
    v = np.empty(np.shape(r))
    vint = np.empty(np.shape(r))
    t = np.empty(N+1)

    r[0] = np.array(r0)
    v[0] = np.array(v0)
    t[0] = 0
    vint[0] = v[0] + 1/2*h*f(r[0], t[0])

    for i in range(N):
        t[i+1] = h*(i + 1)
        r[i+1] = r[i] + h*vint[i]
        k = h*f(r[i+1], t[i+1])
        v[i+1] = vint[i] + 1/2*k
        vint[i+1] = vint[i] + k
    return r, v, t
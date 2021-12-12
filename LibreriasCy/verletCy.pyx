#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
from cython.view cimport array
from libc.math cimport sqrt, cos, sin, atan2
import numpy as np
cimport numpy as np

np.import_array()
DTYPE = np.float
ctypedef np.double_t DTYPE_t


cpdef verlet_solve(double e,
                   double sigma,
                   double m,
                   double[:,::1] r0,
                   double[:,::1] v0,
                   double h,
                   int N,
                   double[:,::1] aComp,
                   double[:,::1] aRot,
                   double[:,::1] w):
    '''
    Función que evoluciona el sistema de M = len(r0) partículas con velocidad
    inicial v0, aceleraciones lineales y radiales aComp y aRot, y velocidades
    angulares w, en N pasos con un intervalo de tiempo h
        Parámetros
        ----------
        e, sigma, m: double
            Valores de e y sigma del potencial de Lennard-Jones, y masa de las
            partículas
        r0, v0: ndarray
            Valores iniciales para r y v
        h: double
            Tamaño de paso temporal
        N: int
            Número de pasos temporales
        aComp, aRot, w: ndarray
            Arrays que indican que aceleración constante, aceleración circular
            y velocidad angular se le asignan a cada partícula. La partícula a
            la que se le asignan corresponde con la posición en el array en que
            se encuentran los parámetros, y estos pueden estar orientados en la
            dirección X, Y o Z, lo cual viene indicado por su posición en los
            sub arrays. Por ejemplo, si queremos darle a la primera parícula
            una aceleración constante de 1 en el eje X, y otra aceleración
            circular de módulo 2 que gire alrededor del eje X con una velocidad
            angular 3, lo que tendriamos que pasar es:

                aComp = [[1, 0, 0],
                         [0, 0, 0],
                             .
                             .
                             .
                         [0, 0, 0]]

                aRot = [[2, 0, 0],
                        [0, 0, 0],
                            .
                            .
                            .
                        [0, 0, 0]]

                w = [[3, 0, 0],
                     [0, 0, 0],
                         .
                         .
                         .
                     [0, 0, 0]]
 
        Devuelve
        --------
        tuple
            Tupla con los arrays solución de la posición de cada partícula en
            cada instante (r), velocidad de cada partícula en cada instante (v)
            y energía potencial de todo el sistema en cada instante (U)
    '''
    cdef:
        int M = r0.shape[0]
        double R
        double F
        double rc = 3*sigma
        double[:] wX = np.empty(M, dtype=DTYPE)
        double[:] wY = np.empty(M, dtype=DTYPE)
        double[:] wZ = np.empty(M, dtype=DTYPE)
        double[:] aRotX = np.empty(M, dtype=DTYPE)
        double[:] aRotY = np.empty(M, dtype=DTYPE)
        double[:] aRotZ = np.empty(M, dtype=DTYPE)
        double[:] aCompX = np.empty(M, dtype=DTYPE)
        double[:] aCompY = np.empty(M, dtype=DTYPE)
        double[:] aCompZ = np.empty(M, dtype=DTYPE)
        double[::1] U = np.zeros(N, dtype=DTYPE)
        double[:, ::1] f = np.zeros((M, 3), dtype=DTYPE)
        double[:,:,::1] r = np.empty((N, M, 3), dtype=DTYPE)
        double[:,:,::1] v = np.empty((N, M, 3), dtype=DTYPE)
        double[:,:,::1] vint = np.empty((N, M, 3), dtype=DTYPE)
        Py_ssize_t t, i, j, k


    r[0] = r0
    v[0] = v0

    aCompX =  aComp.T[0]
    aCompY =  aComp.T[1]
    aCompZ =  aComp.T[2]
    aRotX =  aRot.T[0]
    aRotY =  aRot.T[1]
    aRotZ =  aRot.T[2]
    wX =  w.T[0]
    wY =  w.T[1]
    wZ =  w.T[2]


    for i in range(M):

        for j in range(i):
            R = sqrt((r0[j][0] - r0[i][0])**2 +
                     (r0[j][1] - r0[i][1])**2 +
                     (r0[j][2] - r0[i][2])**2)
            if R > rc:
                F = 0
            else:
                F = 4*e*(6*(sigma/R)**6 - 12*(sigma/R)**12)/R**2
            for k in range(3):
                f[i][k] += F*(r0[j][k] - r0[i][k])/m
            U[0] += 4*e*((sigma/R)**12 - (sigma/R)**6 )/2

        for j in range(i+1, M):
            R = sqrt((r0[j][0] - r0[i][0])**2 +
                     (r0[j][1] - r0[i][1])**2 +
                     (r0[j][2] - r0[i][2])**2)
            if R > rc:
                F = 0
            else:
                F = 4*e*(6*(sigma/R)**6 - 12*(sigma/R)**12)/R**2
            for k in range(3):
                f[i][k] += F*(r0[j][k] - r0[i][k])/m
            U[0] += 4*e*((sigma/R)**12 - (sigma/R)**6 )/2

    
    for i in range(M):
        vint[0][i][0] = v[0][i][0] + 1/2*h*(
            f[i][0] +
            aCompX[i]
            + aRotY[i]*cos(atan2(r0[i][0],r0[i][2]))
            - aRotZ[i]*sin(atan2(r0[i][1],r0[i][0]))
            )
        vint[0][i][1] = v[0][i][1] + 1/2*h*(
            f[i][1] +
            aCompY[i]
            - aRotX[i]*sin(atan2(r0[i][2],r0[i][1]))
            + aRotZ[i]*cos(atan2(r0[i][1],r0[i][0]))
            )
        vint[0][i][2] = v[0][i][2] + 1/2*h*(
            f[i][2] +
            aCompZ[i]
            + aRotX[i]*cos(atan2(r0[i][2],r0[i][1]))
            - aRotY[i]*sin(atan2(r0[i][0],r0[i][2]))
            )


    for t in range(N-1):
        for i in range(M):
            for k in range(3):
                r[t+1][i][k] = r[t][i][k] + h*vint[t][i][k]
        
        for i in range(M):
            f[i][0] = 0.
            f[i][1] = 0.
            f[i][2] = 0.

            for j in range(i):
                R = sqrt((r[t+1][j][0] - r[t+1][i][0])**2 +
                         (r[t+1][j][1] - r[t+1][i][1])**2 +
                         (r[t+1][j][2] - r[t+1][i][2])**2)
                if R > rc:
                    F = 0
                else:
                    F = 4*e*(6*(sigma/R)**6 - 12*(sigma/R)**12)/R**2
                for k in range(3):
                    f[i][k] += F*(r[t+1][j][k] - r[t+1][i][k])/m
                U[t+1] += 4*e*((sigma/R)**12 - (sigma/R)**6)/2

            for j in range(i+1, M):
                R = sqrt((r[t+1][j][0] - r[t+1][i][0])**2 +
                         (r[t+1][j][1] - r[t+1][i][1])**2 +
                         (r[t+1][j][2] - r[t+1][i][2])**2)
                if R > rc:
                    F = 0
                else:
                    F = 4*e*(6*(sigma/R)**6 - 12*(sigma/R)**12)/R**2
                for k in range(3):
                    f[i][k] += F*(r[t+1][j][k] - r[t+1][i][k])/m
                U[t+1] += 4*e*((sigma/R)**12 - (sigma/R)**6)/2

            v[t+1][i][0] = vint[t][i][0] + 1/2*h*(
                f[i][0] +
                aCompX[i]
                + aRotY[i]*cos(wY[i]*h*(t+1) + atan2(r0[i][0],r0[i][2]))
                - aRotZ[i]*sin(wZ[i]*h*(t+1) + atan2(r0[i][1],r0[i][0]))
                )
            vint[t+1][i][0] = vint[t][i][0] + h*(
                f[i][0] +
                aCompX[i]
                + aRotY[i]*cos(wY[i]*h*(t+1) + atan2(r0[i][0],r0[i][2]))
                - aRotZ[i]*sin(wZ[i]*h*(t+1) + atan2(r0[i][1],r0[i][0]))
                )

            v[t+1][i][1] = vint[t][i][1] + 1/2*h*(
                f[i][1] +
                aCompY[i]
                - aRotX[i]*sin(wX[i]*h*(t+1) + atan2(r0[i][2],r0[i][1]))
                + aRotZ[i]*cos(wZ[i]*h*(t+1) + atan2(r0[i][1],r0[i][0]))
                )
            vint[t+1][i][1] = vint[t][i][1] + h*(
                f[i][1] +
                aCompY[i]
                - aRotX[i]*sin(wX[i]*h*(t+1) + atan2(r0[i][2],r0[i][1]))
                + aRotZ[i]*cos(wZ[i]*h*(t+1) + atan2(r0[i][1],r0[i][0]))
                )

            v[t+1][i][2] = vint[t][i][2] + 1/2*h*(
                f[i][2] +
                aCompZ[i]
                + aRotX[i]*cos(wX[i]*h*(t+1) + atan2(r0[i][2],r0[i][1]))
                - aRotY[i]*sin(wY[i]*h*(t+1) + atan2(r0[i][0],r0[i][2]))
                )
            vint[t+1][i][2] = vint[t][i][2] + h*(
                f[i][2] +
                aCompZ[i]
                + aRotX[i]*cos(wX[i]*h*(t+1) + atan2(r0[i][2],r0[i][1]))
                - aRotY[i]*sin(wY[i]*h*(t+1) + atan2(r0[i][0],r0[i][2]))
                )


    return np.array(r), np.array(v), np.array(U)
import json
import matplotlib.pyplot as plt
import numpy as np
from .helpers import r, set_axes_equal


lattices = json.load(open("Datos/data.json"))["Lattices"]

class lattice:
    """
    Objeto que genera una red de Bravais

        Atributos
        ---------
        latt: str
            Tipo de red que genera
        a: float
            Parámetro de red en Angstroms. Por defecto es a=1
        base: ndarray
            Base de la red latt
        ucell: ndarray
            Coordenadas de los puntos que conforman la celda unidad
            que vamos a usar para generar la red

        Métodos
        -------
        gen_lattice(dim)
            Genera las coordenadas de los puntos de la red de las dimensiones
            indicadas
        plot(dim, show=True, save=False)
            Genera un plot de la red de las dimensiones indicadas
        gen_dat(dim)
            Genera archivo .dat de la red de las dimensiones indicadas
    """

    def __init__(self, latt, a=1):
        self.latt = latt
        self.a = a
        self.base = np.array(lattices[self.latt]["base"])
        self.ucell = np.array(lattices[self.latt]["ucell"])

    def __gen_point(self, point):
        """
        Genera las coordenadas de un punto de la red usando la base de esta

            Parámetros
            ----------
            point: array_like
                Coordenadas del punto en la base de la red a generar

            Devuelve
            --------
            ndarray
                Coordenadas del punto en la base euclidea
        """
        return sum([x * (self.a * e)
                    for x, e in zip(point, self.base)])

    def __gen_ucell(self):
        """
        Genera coordenadas de los puntos de la celda convencional
        de la red para su uso con condiciones periódicas

            Devuelve
            --------
            ndarray
                Coordenadas de los puntos de la celda convencional con
                condiciones periodicas
        """
        return np.array(
            [self.__gen_point(coord) for coord in self.ucell]
            )

    def gen_lattice(self, dim):
        """
        Genera las coordenadas de los puntos de la red de las dimensiones
        indicadas

            Parámetros
            ----------
            dim: array-like
                Dimensiones en las coordenadas cartesianas en numero de
                celdas convecionales

            Devuelve
            --------
            ndarray
                Coordenadas de los puntos de la red
            self.r: ndarray
                Atributo con las coordenadas de la red
        """
        # Generamos los puntos de la celda convencional
        ucell = self.__gen_ucell()

        # Generamos N = np.prod(dim) copias de los puntos de
        # la celda convencional
        base = np.tile(ucell, (np.prod(dim), 1))

        # Obtenermos las coordenadas de las celdas convencionales
        # en la celda de simulación
        coord = np.vstack(np.meshgrid(*[np.arange(n) for n in dim])
                            ).reshape(3,-1).T * self.a

        # Añadimos una copia de cada coordenada de la celda de simulación
        # por cada punto de la celda convecional
        coord = np.repeat(coord, len(ucell), axis=0)

        # Movemos las N copias de la celda convencional a cada punto de
        # la celda de simulación
        r = base + coord
        self.r0 = r
        return r

    def plot(self, coord=0, E=0, F=0, onlyF=False, show=True, save=False):
        """
        Genera un plot de la red de las dimensiones indicadas

            Parámetros
            ----------
                coord: ndarray, optional
                    Coordenadas de la red a plotear distintas de las guardadas
                E: ndarray, optional
                    Array con las energía de cada partícula. Es usa para generar
                    colorear las partículas en base a estas.
                F: ndarray
                    Array con las fuerzas que actuan sobre cada partícula
                onlyF: bool, optional
                    Si mostras solo las fuerzas o no
                show: bool, optional
                    Si mostrar o no el plot
                save: bool, optional
                    Si guardar o no una imagen del plot
        """
        title = "Red " + self.latt
        xlabel="$x (\AA)$"
        ylabel="$y (\AA)$"
        zlabel="$z (\AA)$"
        size = 50

        fig = plt.figure()

        ax = fig.add_subplot(111, projection="3d")
        if (type(coord) != np.ndarray) and hasattr(self, "r0"):
            coord = self.r0

        if not(onlyF):
            if type(E) == np.ndarray:
                atoms = ax.scatter(coord.T[0], coord.T[1], coord.T[2], s=size, c=E, cmap="coolwarm")
                cbar_atoms = fig.colorbar(atoms)
                cbar_atoms.set_label("$E$ ($eV$)")
            else:
                ax.scatter(coord.T[0], coord.T[1], coord.T[2], s=size)
        
        if type(F) == np.ndarray:
            X, Y, Z = coord.T
            U, V, W = F.T
            R = r(F)
            R = np.append(R, np.repeat(R, 2, axis=0), axis=0)
            forces = ax.quiver(X, Y, Z, U, V, W, colors=plt.cm.viridis(R))
            cbar_forces = fig.colorbar(forces)
            cbar_forces.set_label("$F$ ($eV\ s^{-1}$)")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)

        set_axes_equal(ax)

        if save:
            plt.savefig(self.latt + "_" + str(self.a) + "_"+ str(len(coord)))

        if show:
            plt.show()
        return fig, ax

    def gen_dat(self, dim):
        """
        Genera archivo .dat de la red de las dimensiones indicadas

            Parámetros
            ----------
                dim: array-like
                    Dimensiones en las coordenadas cartesianas en numero de
                    celdas convecionales a guardar

        """
        coords = self.gen_lattice(dim)
        body = "{0}\n".format(len(coords))
        for point in coords:
            body += (
                "         ".join([str(num) for num in point])
                + "\n"
            )
        filename = self.latt + "_" + str(self.a) + "_"+ str(dim)
        filename += ".dat"
        f = open(filename, "w")
        f.write(body)
        f.close()


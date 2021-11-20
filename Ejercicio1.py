import ast

from Librerias.lattice import lattice, lattices


lattSelection = "FCC"
dim = [5, 5, 5]

latt = lattice(lattSelection)
coord = latt.gen_lattice(dim)
latt.plot(coord)

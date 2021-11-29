import ast

from Librerias.lattice import lattice, lattices


lattSelection = "FCC"
dim = [5, 5, 5]

latt = lattice(lattSelection)
latt.gen_lattice(dim)
latt.plot()

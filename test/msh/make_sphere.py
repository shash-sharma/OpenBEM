import gmsh
import sys
import os
import numpy as np

gmsh.initialize()
gmsh.model.add("sphere")

coarse = False

r = 0.5
lc = 2 * np.pi * r / 60    

outname = f"{os.path.dirname(os.path.abspath(__file__))}/sphere_fine"

sphere = gmsh.model.occ.addSphere(0, 0, 0, 0.5, tag=-1)

gmsh.model.occ.synchronize()

pgr = gmsh.model.addPhysicalGroup(2, [sphere], tag=-1, name="sphere")

gmsh.model.occ.synchronize()

dimTags = gmsh.model.getEntities(dim=0)
gmsh.model.mesh.setSize(dimTags, lc)

gmsh.option.setNumber("Mesh.MshFileVersion", 2)
gmsh.model.mesh.setAlgorithm(dim=2, tag=sphere, val=1)

gmsh.model.mesh.generate(2)

gmsh.write(outname + ".msh")
print(f"Wrote mesh to: {outname}.msh")

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()

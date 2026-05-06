import numpy as np
import gmsh

# gmsh.initialize()
# gmsh.model.add("ImportedGeometry")
#
# volumes = gmsh.model.occ.importShapes("CAD/assembly.step")
# gmsh.model.occ.synchronize()
# gmsh.model.occ.removeAllDuplicates()
# gmsh.model.occ.synchronize()
# target_size = 2
# gmsh.option.setNumber("Mesh.MeshSizeMax",target_size)
# gmsh.model.mesh.generate(3)
# gmsh.option.setNumber("Mesh.SaveAll",1)
# gmsh.write("model.msh")
# gmsh.model.occ.synchronize()
# gmsh.finalize()

gmsh.initialize()
gmsh.open("model.msh")
gmsh.fltk.run()



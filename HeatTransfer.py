import numpy as np
import gmsh
from helper import *
import sys

class HeatTransfer:
    def __init__(self):
        self.mass_matrix = None # initialized in import geometry
        self.stiffness_matrix = None # initialized in import geometry
        self.load_array = None # used for flux and convective boundary conditions
        self.temperature_array = None
        self.temperature_dot_array = None
        self.boundary_array = None
        self.tet_volumes = None
        self.nodes = None
        self.tri_nodes = None # nodes for each surface triangle
        self.node_coordinates = None
        self.tet_nodes_dict= {}
        self.scale = 1.0/1000.0 # adjust ImportGeometry.py so it converts units to meters through scaling and this can be user option but not necessary
        #self.tet_tags = None
        self.num_nodes = 0
        self.num_tets = 0
        self.num_tri = 0 # number of boundary triangles
        self.material_properties = {}
        self.starting_temperature = 300 # should be able to set different volumes to different starting temperatures
        #or even define a gradient in space and interpolate temperatures onto nodes
        self.boundary_conditions = {}
        self.volume_dict = {}
        self.surface_dict = {}
        self.temperature_surface = {} # stores entity tags of surfaces set to constant temperature
        self.tet_gradients={} # stores tet tag and associated gradients
        self.tet_volumes={} # stores tet tag and volume
        self.tri_areas = {} # areas of boundary triangles - triangle and faces are different things
    def calc_all_tet_properties(self):
        #tet_tags, tet_nodes = gmsh.model.mesh.getElementsByType(4) # get all tetrahedra
        for tet_tag,nodes  in self.tet_nodes_dict.items():
            pt_coords = np.zeros((4,3))
            for i,node in enumerate(nodes):
                coords = self.scale*gmsh.model.mesh.getNode(node)[0]
                pt_coords[i,:] = np.array(coords)
             #print("nodes",)
            #pt_coords = np.reshape(pt_coords_flat, (4, 3))  # each row has x y z coords
            self.tet_gradients[tet_tag] = calc_single_tet_gradient(pt_coords)
            self.tet_volumes[tet_tag] = calculate_volume(pt_coords)

        #Get Inverse, Rows are the gradients for each pt, gradient for first point is -sum(other gradients)

    def import_material_properties(self):

        self.material_properties["steel"] = [7850, 510, 20]  # rho, Cp, conductivity SI units
        self.material_properties["copper"] = [8960, 376, 400]
    def import_boundary_conditions(self):
        # add function to read values in from file
        self.boundary_conditions["temperature"] = ["Temperature",500] # deg K
        self.boundary_conditions["cold"] = ["Temperature",300] # deg K
        self.boundary_conditions["insulation"] = ["Insulation", 0]  # deg K
    def import_geometry(self):

        tet_tags, tet_node_tags = gmsh.model.mesh.getElementsByType(4)  # gets tets and node tags
        # print("tet tags")
        # print(element_tags)
        face_nodes = gmsh.model.mesh.getElementFaceNodes(4, 3)
        #gmsh.model.mesh.createEdges()
        #gmsh.model.mesh.createFaces()
        #face_tags, face_orientations = gmsh.model.mesh.getFaces(3, face_nodes)
        #print("Face Tags",face_tags)
        node_tags, node_coords, node_params = gmsh.model.mesh.getNodes()

        num_pts = node_tags.size
        reshaped = np.reshape(node_coords*self.scale, (num_pts,3))
        self.nodes = (np.array(list(range(num_pts)))).T
        self.node_coordinates = reshaped

        print("max x", max((reshaped[:,0])))
        print("max y", max((reshaped[:, 1])))
        print("max z", max((reshaped[:, 2])))
        # print("Node Tags")
        # print(node_tags)

        print("Num pts: ", num_pts)

        num_tets = tet_tags.size
        print("Num tets: ", num_tets)
        for i in range(0,num_tets):
            node_tags = tet_node_tags[(4 * i):(4 * i + 4)]
            self.tet_nodes_dict[tet_tags[i]] = node_tags.astype(int)

        self.num_nodes = num_pts
        self.num_tets = num_tets
        self.mass_matrix = np.zeros((self.num_nodes,self.num_nodes))
        self.stiffness_matrix = np.zeros((self.num_nodes, self.num_nodes))
        self.boundary_array = np.zeros((self.num_nodes,1))
        self.temperature_array = np.zeros((self.num_nodes, 1)) + self.starting_temperature
        self.temperature_dot_array = np.zeros((self.num_nodes, 1))
        self.load_array = np.zeros((self.num_nodes, 1))

    def import_groups(self):
        physical_groups = gmsh.model.getPhysicalGroups()
        # convert to list of names and tags
        num_volumes = sum(1 for group in physical_groups if group[0] == 3)
        num_surfaces = sum(1 for group in physical_groups if group[0] == 2)
        volume_group_dict = {}
        surface_group_dict = {}
        for group in physical_groups:
            if group[0] == 2:
                name = gmsh.model.getPhysicalName(2, group[1])
                surface_group_dict[name] = group[1]
            else:
                name = gmsh.model.getPhysicalName(3, group[1])
                volume_group_dict[name] = group[1]


        print("Number of Volumes: ", num_volumes)
        print("Number of Surfaces: ", num_surfaces)
        print(volume_group_dict)
        print(surface_group_dict)

        # for i in range(0,num_surfaces):
        self.volume_dict = volume_group_dict
        self.surface_dict = surface_group_dict

    def assemble_mass_matrix(self):  # pass volume dictionary and material matrix
        for material, physical_tag in self.volume_dict.items():
            tags = gmsh.model.getEntitiesForPhysicalGroup(3, physical_tag) #get mesh blocks belonging to group
            rho, cp, k = self.material_properties[material]
            for tag in tags:
                self.calculate_individual_mass_matrix(tag,rho,cp)

    def assemble_stiffness_matrix(self):
        for material, physical_tag in self.volume_dict.items():
            tags = gmsh.model.getEntitiesForPhysicalGroup(3, physical_tag) #get mesh blocks belonging to group
            rho, cp, k = self.material_properties[material]
            for tag in tags:
                self.calculate_individual_stiffness_matrix(tag,k)

    def calculate_individual_mass_matrix(self,volume_tag,rho,cp):  # takes in volume entity tag and add contributions to global matrix
        #using lumped mass matrix - only add values to diagonals
        #gmsh.model.mesh.createFaces([(3,volume_tag)])
        tet_tags, tet_node_tags = gmsh.model.mesh.getElementsByType(4, volume_tag)  # get tetrahedron tags and the 4 nodes for the tets
        quadrature_matrix = np.zeros((4, 4))
        np.fill_diagonal(quadrature_matrix, 1)
        for tet_tag in tet_tags :
            volume = self.tet_volumes[tet_tag]
            nodes = self.tet_nodes_dict[tet_tag] #node tags used to index mass_matrix, tag-1 is position since nodes start at 1
            local_mass_matrix = 1/4*quadrature_matrix*volume
            for i in range(0,4):
                self.mass_matrix[nodes[i]-1,nodes[i]-1] += local_mass_matrix[i,i]*rho*cp
    def calculate_individual_stiffness_matrix(self,volume_tag,k):
        tet_tags, tet_node_tags = gmsh.model.mesh.getElementsByType(4,
                                                                   volume_tag)  # get tetrahedron tags and the 4 nodes for the tets

        for tet_tag in tet_tags:
            volume = self.tet_volumes[tet_tag]
            nodes = self.tet_nodes_dict[tet_tag]  # node tags [0 1 2 3], (tag-1) is index since nodes start at 1
            gradient = self.tet_gradients[tet_tag] #4x3 array
            local_stiffness = np.matmul(gradient,gradient.T)*volume

            indices = np.ix_(nodes-1,nodes-1)
            self.stiffness_matrix[indices] += local_stiffness*k

    def calculate_boundary_matrix(self):

        for surface_name,physical_tag in self.surface_dict.items():
            data = self.boundary_conditions[surface_name]
            boundary_type = data[0]
            match boundary_type:
                case "Temperature":

                    tags = gmsh.model.getEntitiesForPhysicalGroup(2, physical_tag)

                    temperature = data[1]
                    print("Temp",temperature)
                    for tag in tags:
                        self.temperature_surface[tag] = temperature
                        self.constant_temperature_adjustment(tag)

                case "Flux":
                    print("")
                case "Convection":
                    print("")
                # tags = gmsh.model.getEntitiesForPhysicalGroup(2, physical_tag)
                # for tag in tags:
                    #tags,nodes = gmsh.model.mesh.getElementsByType(2, tag)
                    # print(surface_name + " tag: "+str(tag)+"  tags:" )
                    # print(tags)
                    # print("Node sampling")
                    # print(nodes[0:6])

    def constant_temperature_adjustment(self,tag):
        #get faces belonging to tag
        tri_tags, nodes = gmsh.model.mesh.getElementsByType(2, tag)
        nodes_unique = np.unique(nodes)
        for node in nodes_unique:
            self.mass_matrix[node-1,node-1] = 1
            self.stiffness_matrix[node - 1, :] = 0 # only rows need to be set to 0 , not columns since columns will affect other nodes also; each row is for a node , columns are effect of one node on many and should be left alone
            #self.stiffness_matrix[:, node - 1] = 0


    def apply_constant_temperature(self):
        for tag,temperature in self.temperature_surface.items():
            tri_tags, nodes = gmsh.model.mesh.getElementsByType(2, tag)
            #print(nodes)
            nodes_unique = np.unique(nodes)
            for node in nodes_unique:
                self.temperature_array[node - 1,0] = temperature



    def calculate_boundary_tri_areas(self):
        #grab all triangles
        types,tri_tags,tri_nodes = gmsh.model.mesh.getElements(2,-1)
        num_tri = len(tri_tags[0])

        tri_nodes = np.reshape(tri_nodes[0],(num_tri,3))
        self.num_tri = num_tri
        self.tri_nodes = tri_nodes.astype(int)
        coords = np.zeros((3,3))
        for i,tag in enumerate(tri_tags[0]):
            nodes = tri_nodes[i,:]
            for i,node in enumerate(nodes):
                coords[i, :] = self.scale*gmsh.model.mesh.getNode(node)[0]

            area = calc_triangle_area(coords)
            self.tri_areas[tag] = area

    def write_data(self):
        with open("data.txt","w") as file:
            file.write(str(self.num_nodes) + "\n")
            file.write(str(self.num_tri)+ "\n")
            #write all coordinates to file
        with open("data.txt","a") as file:
            np.savetxt(file,self.node_coordinates,delimiter = " ")
            #file.write("\n")
            np.savetxt(file,self.tri_nodes,delimiter = " ",fmt="%i")
            # for surface_name,physical_tag in self.surface_dict.items():
            #     tags = gmsh.model.getEntitiesForPhysicalGroup(2, physical_tag)
            # #for each tag,
    #simulations -  can do different time stepping methods to experiment
    def simulate_backwards_euler(self,dt,time):
        self.apply_constant_temperature()
        current_temperature = self.temperature_array.copy()

        np.savetxt("Temperature_initial", self.temperature_array)
        next_temperature = np.zeros((self.num_nodes,1)) # column array to store new temperature values
        steps = int(time/dt)
        print(steps)
        #can add code to check condition number and figure out if preconditioning is needed
        #steps = 1
        M_inverse = np.linalg.inv(self.mass_matrix)
        first = np.linalg.inv((np.eye(self.num_nodes) + dt * M_inverse @ self.stiffness_matrix))
        second =  dt * M_inverse @ self.load_array

        tdot = -M_inverse@self.stiffness_matrix@ current_temperature
        np.savetxt("Initial Tdot", tdot)
        # for step in range(0,steps):
        #     next_temperature = first @ (current_temperature+second)
        #     current_temperature = next_temperature.copy()
        #     print(next_temperature[28,0])

        np.savetxt("Temperature", current_temperature)








def main():
    gmsh.initialize()
    gmsh.open("model.msh")
    x,y,z,X,Y,Z = gmsh.model.getBoundingBox(-1,-1)
    print(x)
    print(X)
    testsim = HeatTransfer()
    #extract volume and surface grouping names and tags
    testsim.import_groups()
    # import material properties
    testsim.import_material_properties()
    testsim.import_boundary_conditions()
    #import mesh data and calculate useful values
    testsim.import_geometry() #
    testsim.calc_all_tet_properties()
    #Compute Mass and Stiffness matrices
    testsim.assemble_mass_matrix()
    testsim.assemble_stiffness_matrix()
    testsim.calculate_boundary_tri_areas()

    testsim.calculate_boundary_matrix()

    #run simulation
    testsim.simulate_backwards_euler(0.001,0.005)
    testsim.write_data()


    #solve matrix w/ preconditioner and maybe rescaling?
    np.savetxt("Mass Matrix", testsim.mass_matrix, delimiter=',', fmt='%.2f')
    np.savetxt("Stiffness Matrix", testsim.stiffness_matrix, delimiter=',', fmt='%.2f')

    gmsh.finalize()


# #gmsh.fltk.run()


main()
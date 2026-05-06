import numpy as np
import gmsh
import sys

class HeatTransfer:
    def __init__(self):
        self.mass_matrix = None
        self.tet_volumes = None
        self.tet_nodes_dict= {}
        #self.tet_tags = None
        self.num_nodes = 0
        self.num_tets = 0
        self.material_properties = {}
        self.volume_dict = {}
        self.surface_dict = {}
        self.tet_gradients={} # stores tet tag and associated gradients
        self.tet_volumes={} # stores tet tag and volume
    def calc_all_tet_properties(self):
        #tet_tags, tet_nodes = gmsh.model.mesh.getElementsByType(4) # get all tetrahedra
        for tet_tag,nodes  in self.tet_nodes_dict.items():
            pt_coords = np.zeros((4,3))
            for i,node in enumerate(nodes):
                coords = gmsh.model.mesh.getNode(node)[0]
                pt_coords[i,:] = np.array(coords)
             #print("nodes",)
            #pt_coords = np.reshape(pt_coords_flat, (4, 3))  # each row has x y z coords
            self.tet_gradients[tet_tag] = calc_single_tet_gradient(pt_coords)
            self.tet_volumes[tet_tag] = calculate_volume(pt_coords)

        #Get Inverse, Rows are the gradients for each pt, gradient for first point is -sum(other gradients)

    def import_material_properties(self):

        self.material_properties["steel"] = [7850, 510, 20]  # rho, Cp, conductivity SI units
        self.material_properties["copper"] = [8960, 376, 400]

    def import_geometry(self):

        tet_tags, tet_node_tags = gmsh.model.mesh.getElementsByType(4)  # gets tets and node tags
        # print("tet tags")
        # print(element_tags)
        face_nodes = gmsh.model.mesh.getElementFaceNodes(4, 3)
        gmsh.model.mesh.createEdges()
        gmsh.model.mesh.createFaces()
        face_tags, face_orientations = gmsh.model.mesh.getFaces(3, face_nodes)
        node_tags, node_coords, node_params = gmsh.model.mesh.getNodes()
        num_pts = node_tags.size
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


    def calculate_individual_mass_matrix(self,volume_tag,rho,cp):  # takes in volume entity tag and add contributions to global matrix
        #using lumped mass matrix - only add values to diagonals
        dim, tet_tags, tet_node_tags = gmsh.model.mesh.getElements(3,
                                                                   volume_tag)  # get tetrahedron tags and the 4 nodes for the tets
        tet_nodes = tet_node_tags[0]  # remove outer list
        num_tets = len(tet_tags[0])  # number of tetrahedrons
        tet_tags = tet_tags[0]
        #print(tet_tags)
        #nodes for each tet in matrix for calculating volume, etc
        quadrature_matrix = np.zeros((4, 4))
        np.fill_diagonal(quadrature_matrix, 1)
        for tet_tag in tet_tags :
            volume = self.tet_volumes[tet_tag]
            nodes = self.tet_nodes_dict[tet_tag] #node tags used to index mass_matrix, tag-1 is position since nodes start at 1
            local_mass_matrix = 1/4*quadrature_matrix*volume
            for i in range(0,4):
                self.mass_matrix[nodes[i]-1,nodes[i]-1] += local_mass_matrix[i,i]*rho*cp


        #     stiffness_local = np.zeros((4,4))
        #     #stiffness matrix
        #     gradient_local[:,:] = calculate_gradients(pts)
        #     tet_gradients[i,:,:] = gradient_local[:,:]
        #     for a in range(0,4):
        #         for b in range(0,4):
        #             stiffness_local[a,b] = volume * np.dot(gradient_local[a,:],gradient_local[b,:])
        #     #add local mass and stiffness matrices to respective global matrices
        #     for a in range(0,4):
        #         for b in range(0,4):
        #             M[nodes[a],nodes[b]] = M[nodes[a],nodes[b]] + volume*quadrature_matrix[a,b]
        #             A[nodes[a], nodes[b]] = A[nodes[a], nodes[b]] + stiffness_local[a,b]
        #return individual_matrix
    def assemble_stiffness_matrix(self,volume_tag):
        individual_matrix = np.zeros((self.num_nodes, self.num_nodes))
        dim, tet_tags, tet_node_tags = gmsh.model.mesh.getElements(3,
                                                                   volume_tag)  # get tetrahedron tags and the 4 nodes for the tets
        tet_node_tags = tet_node_tags[0]  # remove outer list
        num_tets = len(tet_tags[0])  # number of tetrahedrons

        tet_nodes = np.zeros((self.num_nodes, 4))
        local_mass_matrix = np.zeros((self.num_nodes, self.num_nodes))

        # nodes for each tet in matrix for calculating volume, etc
        for i in range(0, num_tets):
            tet_nodes[i, :] = tet_node_tags[(4 * i):(4 * i + 4)] - 1
        tet_nodes = tet_nodes.astype(int)





def main():
    gmsh.initialize()
    gmsh.open("model.msh")
    testsim = HeatTransfer()
    #extract volume and surface grouping names and tags
    testsim.import_groups()
    # import material properties
    testsim.import_material_properties()
    #import mesh data and calculate useful values
    testsim.import_geometry() #
    testsim.calc_all_tet_properties()
    #Create mass and stiffness matrix
    testsim.assemble_mass_matrix()

    #Fill mass and stiffness matrix


    #solve matrix w/ preconditioner and maybe rescaling?
    np.savetxt("Mass Matrix", testsim.mass_matrix, delimiter=',', fmt='%.2f')
    gmsh.finalize()

def calculate_volume(pts):
    vol_array = np.ones((4,4))
    vol_array[:,0:3] = pts
    return 1/6*abs(np.linalg.det(vol_array))
def calculate_gradients(pts: np.ndarray):
    J = np.zeros((3,3))
    #fill jacobian
    for i in range(0,3):
        for j in range(0,3):
            J[i,j] = pts[j+1,i]-pts[0,i]
    transform = np.zeros((3,1))
    #print(transform.shape)
    left = np.array([[pts[1,0]-pts[0,0]], [pts[1,1]-pts[0,1]],[ pts[1,2]-pts[0,2] ]])

    transform[:] = np.linalg.inv(J) @ left
    gradients = np.zeros((4,3))
    #rint(gradients[0,])
    gradients[0,:] = (np.linalg.inv(J.T) @ np.array([[-1],[-1],[-1]])).T #Transpose vector to make it horizontal
    for i in range(1,4):
        init = np.zeros((3,1))
        init[i-1,0] = 1
        gradients[i,:] = (np.linalg.inv(J.T) @ init).T
    return gradients

def calc_single_tet_gradient(pt_coords):
    J_T = np.zeros((3, 3))

    for i in range(0, 3):
        J_T[i, :] = pt_coords[i+1, :] - pt_coords[0, :]
    J_T_invert = np.linalg.inv(J_T)
    first_gradient = -1*np.sum(J_T_invert,axis=0)
    gradient = np.vstack((first_gradient,J_T_invert))
    return gradient















# #gmsh.fltk.run()


main()
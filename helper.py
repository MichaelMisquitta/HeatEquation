import numpy as np
def calculate_volume(pts):
    vol_array = np.ones((4,4))
    vol_array[:,0:3] = pts
    return 1/6*abs(np.linalg.det(vol_array))
# def calculate_gradients(pts: np.ndarray):
#     J = np.zeros((3,3))
#     #fill jacobian
#     for i in range(0,3):
#         for j in range(0,3):
#             J[i,j] = pts[j+1,i]-pts[0,i]
#     transform = np.zeros((3,1))
#     #print(transform.shape)
#     left = np.array([[pts[1,0]-pts[0,0]], [pts[1,1]-pts[0,1]],[ pts[1,2]-pts[0,2] ]])
#
#     transform[:] = np.linalg.inv(J) @ left
#     gradients = np.zeros((4,3))
#     #rint(gradients[0,])
#     gradients[0,:] = (np.linalg.inv(J.T) @ np.array([[-1],[-1],[-1]])).T #Transpose vector to make it horizontal
#     for i in range(1,4):
#         init = np.zeros((3,1))
#         init[i-1,0] = 1
#         gradients[i,:] = (np.linalg.inv(J.T) @ init).T
#     return gradients

def calc_single_tet_gradient(pt_coords):
    J_T = np.zeros((3, 3))

    for i in range(0, 3):
        J_T[i, :] = pt_coords[i+1, :] - pt_coords[0, :]
    J_T_invert = np.linalg.inv(J_T)
    first_gradient = -1*np.sum(J_T_invert,axis=0)
    gradient = np.vstack((first_gradient,J_T_invert))
    return gradient

def calc_triangle_area(pt_coords):
    # [xyz;xyz;xyz]
    vec = np.zeros((2,3))
    vec[0,:] = pt_coords[1,:] - pt_coords[0,:]
    vec[1, :] = pt_coords[2, :] - pt_coords[0, :]
    area = 1/2*np.linalg.norm(np.cross(vec[0,:],vec[1,:]))
    return area
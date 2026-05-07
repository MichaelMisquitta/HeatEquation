import numpy as np
nodes = np.array([0, 1, 3])
nodest = np.array([0, 1, 2])
testmat = np.array([[1, 9, 5, 3],[1, 2, 5, 6],[1, 4, 7, 6],[7, 1, 5, 6]])
print(testmat)
indices = np.ix_(nodes,nodest)
testmat[indices] = np.zeros((3,3))
print(testmat)
#print(indices)
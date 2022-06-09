import os
import numpy as np

if __name__ == '__main__':
    annot = np.loadtxt("D:/thesis-data/ConservationDrones/TestReal/annotations/0000000371_0000000000.csv", delimiter=",", dtype=int)
    
    #annot = annot[:, :]
    #annot[:, 5] = 1
    #annot[:, 0] += 1
   # new_dim = np.zeros((annot.shape[0], 2))
   # annot = np.append(annot,new_dim, axis=1)
    #annot[:, -1] = 0
    np.savetxt("D:/thesis-data/ConservationDrones/TestReal/annotations/0000000371_0000000000.txt", annot, fmt='%i', delimiter=',')
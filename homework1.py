import numpy as np
import scipy.io
from numpy import linalg


#Load features file
# For windows:
features = scipy.io.loadmat(r'C:\Users\diogo\Desktop\IST\PBDat\EurosportAll\girosmallveryslow2.mp4_features.mat')
print(type(features))
# For Linux:
#features = scipy.io.loadmat(r'C:\Users\diogo\Desktop\IST\PBDat\EurosportAll\girosmallveryslow2.mp4_features.mat')




#Print key features of the .mat file 
#This .mat file is an dictionary, so we have to take the key we want
keys = features.keys()
print(keys)


#Put features in an numpy matrix. 
#Each column of this matrix represents the embbeding of an frame [Acho]
#That is each column of matrix B is an ebbeding of an frame
#So we have 10482 data points [frames in this case]
B = features['features']
print("Features shape:", B.shape)


#Get our base subspace
A = B[:, 5894:5906] #We want rank 12
rank_A= np.linalg.matrix_rank(A)
print("Matrix A shape:", A.shape)
print("Matrix A rank:", rank_A)


#So we now have our feature subspace -> Matrix A
#We need to get the projections of the rest of the images onto the subspace
#Compute the Orthogonal projection operator (pi)
pi = np.matmul(np.matmul(A, np.linalg.inv(np.matmul(A.T,A))),A.T)
print("Orthogonal projection operator shape:", pi.shape)


#1- Find the 100 images whose projection on the subspace has larger norm
P = np.matmul(pi,B)
print("Matrix P shape:", P.shape)

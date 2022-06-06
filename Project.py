import scipy.io as sio
import pandas as pd
from numpy.linalg import inv
import numpy as np
from numpy.linalg import matrix_rank
from numpy.linalg import svd
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mat4py import loadmat
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error


def get_matrix(file_path):

    mat = sio.loadmat(file_path)
    count = 0
    for i in mat.keys():
        if count == 3:
            df = pd.DataFrame(mat[i])
        count+=1
    Matrix = df.to_numpy(float)
    return Matrix

#Base = Matrix[:,5894:5906]
#rank_B = matrix_rank(Base) # Get rank from the MAtrix
# Projection on to the subspace
''''
# HMW 1
C = np.matmul(inv(np.matmul(Base.T,Base)),np.matmul(Base.T,Matrix ))
P = np.matmul(Base, C) #Calculate b prependicular
R = np.subtract(Matrix,P)
NormP = np.linalg.norm(P, axis =0)
index_Pnorm_sorted = np.argsort(NormP, axis=0)
print(index_Pnorm_sorted)
NormR = np.linalg.norm(R, axis =0)
index_Rnorm_sorted = np.argsort(NormR, axis=0)
print(rank_B)

u, s, vh = np.linalg.svd(Base)
u.shape, s.shape, vh.shape
s_lowrank = s[:3]
s_lowrank = np.diag(s_lowrank)
vh_lowrank = vh[:3]
u_lowrank = u[:,:3]
u_lowrank.shape, s_lowrank.shape, vh_lowrank.shape
Base_lowrank = np.matmul(np.matmul(u_lowrank,s_lowrank), vh_lowrank)
#Base_lowrank.shape
u_lowrank.shape

'''
#REVER
def get_outliers(data, base, rank):
    # Get base with svd
    #Calculate orthogonal projection on to the Subspace. SVD ARRANJA A MELHOR BASE, COM MENOS ERRO. Calculate PI with the base or with svd of the base ??
    u, s, v = np.linalg.svd(base, full_matrices=False)

    #PI = np.matmul((np.matmul(base, inv(np.matmul(base.T ,base)))), base.T)
    
    PI = np.matmul(u[:,:rank] , u[:,:rank].T)
    print(PI.shape)

    PIN = np.identity(len(PI))-PI

    #print(PI)
    #print(PI2)
    # Calculate the angle between the vectors of the collumn and the Sub orthogonal projection on to the Subspace

    angle = (np.linalg.norm(np.matmul(PI,data), axis =0))
    angle_null = (np.linalg.norm(np.matmul(PIN,data), axis =0))
    angle = angle/(np.linalg.norm(data, axis =0))
    angle_null = angle_null/(np.linalg.norm(data, axis =0))
    plt.hist(angle)
    plt.show()
    outliers=[]
    count = 0
    for i in range (len(angle_null)):
        if angle_null[i] > 0.8:
            #ind_outliers = i
            out = i , angle_null[i]
            outliers.append(out)
            count = count +1
    print(count)
    
    #print(outliers)
    #sort the angle so we can see which one is farder away from the Subspace, whit the histogram we can observe that angles => 0.9 are outliers
    #What to do with the outliers
    return outliers

#With skeletons
#SVD ARRANJA A MELHOR BASE, COM MENOS ERRO
# PEGAR NO VETOR COM menos 

def get_missing_data(data): #REVER, Fiz com e media de cada coluna, faze
    # Professor avisa que é melhor apagar as probabibidades

    data_delete = np.delete(data, 0, 0) #Take out the frame row
    M = data_delete.astype(bool).astype(int) # Getting the Mask
    
    #Substituir os valores de 0
    Count_Visible = (np.count_nonzero(data_delete, axis=1)) 
    data_row_sum = np.sum(data_delete, axis=1)
    #print(Data_mean)
    data_mean = np.divide(data_row_sum,Count_Visible)
    Dfill=np.zeros(data_delete.shape)
    #print(np.true_divide(Skeletons_delete.sum(1),(Skeletons_delete!=0).sum(1)))

    for i in range(data_delete.shape[0]):# Secalhar utilizar a mascara
        for j in range(data_delete.shape[1]):
            if data_delete[i,j] == 0 :
                Dfill[i,j] = data_mean[i]
            else:
                Dfill[i,j] = data_delete[i,j]


    err=64.35
    e = 70
    Data_til = np.multiply(M, Dfill )+np.multiply((1-M),Dfill)
    k = matrix_rank(Data_til)
    r = 4
    while e >= err:
    
        u, s, v = np.linalg.svd(Data_til,full_matrices=False)
        s_lowrank = s[:r]
        s_lowrank = np.diag(s_lowrank)
        vh_lowrank = v[:r]
        u_lowrank = u[:,:r]
        Data_til = np.matmul(np.matmul(u_lowrank,s_lowrank), vh_lowrank)
        #print(Data_til)
        #Data_til = u[:,:r] @ s[0:r,:r]@ v[:r,:]
        Data_i = np.multiply(M, Dfill )+np.multiply((1-M),Data_til)
        e = np.linalg.norm(np.multiply(M,(Data_i-Data_til)))
        Data_til = Data_i
        print(e)
        
    print(Data_til)
    return Data_til, e

#print(Data_til)
#print(Skeletons)

file_path_features = '/home/goncalo/Desktop/ist/PBDat_Project/girosmallveryslow2.mp4_features.mat'
file_path_skeletons = '/home/goncalo/Desktop/ist/PBDat_Project/esqueletosveryslow.mat'
file_path_skeletons_complete ='/home/goncalo/Desktop/ist/PBDat_Project/esqueletosveryslow_complete.mat'

features  =  get_matrix(file_path_features)
skeletons = get_matrix(file_path_skeletons)
skeletons_complete = get_matrix(file_path_skeletons_complete)
#print(skeletons_complete)
rank = matrix_rank(skeletons_complete)

rank1 = matrix_rank(skeletons)


outliers_features = get_outliers(features, features[:,5894:5906], 10 )
missing_data_fill, erro = get_missing_data(skeletons)
print(skeletons_complete)
print(skeletons)
ske_delete = np.delete(skeletons_complete, 0, 0)
mse = mean_squared_error(ske_delete,missing_data_fill) #error between the missing 
print(mse)
''''
import cv2
vidcap = cv2.VideoCapture('big_buck_bunny_720p_5mb.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1'''
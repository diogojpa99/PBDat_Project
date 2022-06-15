import scipy.io as sio
import pandas as pd
from numpy.linalg import inv
import numpy as np
from numpy.linalg import matrix_rank
from numpy.linalg import svd
from scipy import linalg
import matplotlib.pyplot as plt
from mat4py import loadmat
from sklearn.metrics import mean_squared_error



def get_matrix(file_path):

    mat = sio.loadmat(file_path)
    count = 0
    for i in mat.keys():
        if count == 3:
            df = pd.DataFrame(mat[i])
        count+=1
    Matrix = df.to_numpy(dtype = np.float64)
    return Matrix

def get_rank_and_base (data):

    # Get the base from svd. Why svd ?? Gives us the best possible base, with less error

    u, s, v = np.linalg.svd(data, full_matrices=False)

    # Obseve which is the rank of the matrix. Single values give us the importance of each linear indepent collumn

    fig = plt.figure()
    plt.plot(s)     
    fig.suptitle('Single Values', fontsize=16)
    plt.show()

    #Get the base3

    

    print('By the graph which is the rank of these data set ? ')
    rank = int(input())
    base = u[:,:rank]
    return rank, base


def get_outliers(data, base, rank):

    # Discover outliers taking into account the angle of the projection.
    # Get base with svd
    #Calculate orthogonal projection on to the Subspace. SVD ARRANJA A MELHOR BASE, COM MENOS ERRO. Calculate PI with the base or with svd of the base ??

    #PI = np.matmul((np.matmul(base, inv(np.matmul(base.T ,base)))), base.T)
    #PI = np.matmul(u[:,:rank] , u[:,:rank].T)

    PI = np.matmul(base, base.T)
    PIN = np.identity(len(PI))-PI

    # Calculate the angle between the vectors of the collumn and the Sub orthogonal projection on to the Subspace

    angle = (np.linalg.norm(np.matmul(PI,data), axis =0))
    angle_null = (np.linalg.norm(np.matmul(PIN,data), axis =0))
    angle = angle/(np.linalg.norm(data, axis =0))
    angle_null = angle_null/(np.linalg.norm(data, axis =0))
    3
    # Plot the normalization of the angles. Observe which are too far away from the null subspace

    fig = plt.figure()
    fig.suptitle(f'Projection angle on to the null subspace: Rank {rank}', fontsize=14)
    plt.hist(angle_null, bins=300)
    fig = plt.figure()
    fig.suptitle(f'Projection angle on to the subspace: Rank {rank}', fontsize=14)
    plt.hist(angle, bins=300)
    plt.show()

    # Consider outliers every collumn above 0.8 

    count = 0
    outliers=[]
    inliers=[]
    for i in range (len(angle_null)):
        if angle_null[i] > 0.57:
            #ind_outliers = i
            out = i
            outliers.append(out)
            count = count +1
        if angle[i]>0.80 :
            inl = i
            inliers.append(inl)
    
    #Accuracy--> If any outlier is wrong
    wrong_outliers = 0
    for i in range(len(outliers)):
        for j in range(len(inliers)):
            if outliers[i]==inliers[j]:
                wrong_outliers = wrong_outliers +1
    
    Accuracy = wrong_outliers/len(outliers)
    
    #print(Accuracy*100,'%')
    print(outliers)
    print('Number of outliers', count)
    data_delete = np.delete(data, (outliers), 1)
    #print(data_delete.shape)
    #sort the angle so we can see which one is farder away from the Subspace, whit the histogram we can observe that angles => 0.9 are outliers
    #What to do with the outliers ???
    return data_delete

def data_visualization(data):
    # CLEANING DATA

    #Take out the the row of frames

    data_delete = np.delete(data, 0, 0) 

    #Porbilities wiht less tha 0.5 --> Cordinates to zero

    for j in range (data_delete.shape[1]):
        for i in range (2 ,data_delete.shape[0] , 3):
            if data_delete[i,j] < 0.5: 
                data_delete[i-1,j], data_delete[i-2, j] = 0 , 0
                
    #Take ou the probabilities from the data matrix of skeletons

    count = 2
    index_probilities=np.zeros(18, int)
    prob = np.zeros((18, data_delete.shape[1]))

    for i in range (0, 18): #(2, 30, 3)
        prob[i,:] = data_delete[i+count,:]
        index_probilities[i] = i+count
        count = count + 2
    data_delete = np.delete(data_delete, (index_probilities), 0)

    #Delete collumns where more than 24 cordinates are zero

    collumn_zeros = np.zeros((data_delete.shape[1]),int)
    index_zeros =[]
    for i in range(collumn_zeros.shape[0]):
        collumn_zeros[i] = np.count_nonzero(data_delete[:,i]==0)
        if collumn_zeros[i] >= 24 :
            index_zeros.append(i)
    data_delete = np.delete(data_delete, (index_zeros), 1)
    data_delete1 = data_delete

    #Which row has got less zeros 

    row_zeros = np.zeros((data_delete.shape[0]),int)
    for i in range (row_zeros.shape[0]):
        row_zeros[i] = np.count_nonzero(data_delete[i,:]==0)
    fig = plt.figure()
    fig.suptitle('Zeros by row', fontsize=16)
    plt.plot(row_zeros) # Concluimos que as cordennadas na pose 1, que tambem é o centroide, é a posição com menos zeros. Daí utilizarmos a posição 1 para puxar os esqueletos para o seu frame 
    plt.show()

    # Put skeletons Centered in Pose 1 

    index_zero_1 = []
    for j in range (data_delete.shape[1]):
        for i in range (data_delete.shape[0]):
            if data_delete[2,j] == 0 or data_delete[3,j] == 0  : 
                index_zero_1.append(j)
                break
            if i != 2 and i != 3 :
                if i % 2 == 0 :
                    data_delete[i, j] = np.subtract(data_delete[i, j],data_delete[2,j])
                else:
                    data_delete[i, j] = np.subtract(data_delete[i, j],data_delete[3,j])

    data_delete = np.delete(data_delete,index_zero_1, 1)
    data_delete = np.delete(data_delete, 2 ,0 )
    data_delete = np.delete(data_delete, 2 ,0 )
    data_delete1 = np.delete(data_delete1, index_zero_1, 1)
    data_delete1 = np.delete(data_delete1, 2, 0 )
    data_delete1 = np.delete(data_delete1, 2 ,0 )
    return data_delete, data_delete1
      

def get_missing_data(data, data_clean): #REVER, Fiz com e media de cada coluna, faze
    # Professor avisa que é melhor apagar as probabibidades
    #data_delete = np.delete(data, 0, 0) #Take out the frame row

    M = data_clean.astype(bool).astype(int) # Getting the Mask
    
    err=59.091

    e = 100
    Data_til = np.multiply(M, data_clean )+np.multiply((1-M),data_clean)
    Data_til1 = Data_til
    r = 10
    while e >= err:
    
        u, s, v = np.linalg.svd(Data_til,full_matrices=False)
        s_lowrank = s[:r]
        s_lowrank = np.diag(s_lowrank)
        vh_lowrank = v[:r]
        u_lowrank = u[:,:r]
        Data_til = np.matmul(np.matmul(u_lowrank,s_lowrank), vh_lowrank)
        Data_i = np.multiply(M, data_clean )+np.multiply((1-M),Data_til)
        e = np.linalg.norm(np.multiply(M,(Data_til1-Data_til)))
        Data_til = Data_i
        print(e)
    
        
    return Data_til, e


file_path_features = '/home/goncalo/Desktop/ist/PBDat_Project/girosmallveryslow2.mp4_features.mat'
file_path_skeletons = '/home/goncalo/Desktop/ist/PBDat_Project/esqueletosveryslow.mat'
file_path_skeletons_complete ='/home/goncalo/Desktop/ist/PBDat_Project/esqueletosveryslow_complete.mat'

#---------------------Get Data Matrix-------------------------------------

features  =  get_matrix(file_path_features)
skeletons = get_matrix(file_path_skeletons)
skeletons_complete = get_matrix(file_path_skeletons_complete)

#--------------------Get Rank and Base---------------------------------

features_rank, features_base = get_rank_and_base(features)
#skeletons_rank, skeletons_base = get_rank_and_base(skeletons)

#---------------------Getting the outliers------------------------------

features_clean = get_outliers(features,features_base , features_rank )
#features_outliers = get_outliers(skeletons, skeletons_base, skeletons_rank )

skeletons_complete_clean, skeletons2 = data_visualization(skeletons_complete)
skeletons_clean, skeletons1 = data_visualization(skeletons)

#----------------------Missing Data replaced--------------------------

missing_data_fill, erro = get_missing_data(skeletons1, skeletons_clean)
#---------------------Error with the completed matrix ---------------

mse = mean_squared_error(skeletons_complete_clean , missing_data_fill) #error between the missing 
print('Error between Skeletons missing data and the complete matrix: ', mse)

#---------------------------KNN Clustering-----------------------

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

pca = PCA(2)
df = pca.fit_transform(missing_data_fill.T)
#print(df)
#df1 = pca.fit_transform(features_clean.T)

#-----------How much clusteres exist-------

sse = []
for k in range(1, 30):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)
    sse.append(kmeans.inertia_)


plt.style.use("fivethirtyeight")
plt.plot(range(1, 30), sse)
plt.xticks(range(1, ))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

kmeans = KMeans( n_clusters=4)
labels = kmeans.fit_predict(df)

u_labels = np.unique(labels)

for i in u_labels:
    plt.scatter(df[labels == i , 0] , df[labels == i , 1] , label = i)
plt.legend()
plt.title('Kmeans Clustering')
plt.show()


#-------------------------Spectral Clustering-----------------------
'''
from sklearn.cluster import SpectralClustering

sc = SpectralClustering(n_clusters=4).fit_predict(features_clean.T)

u_labels1 = np.unique(sc)

for i in u_labels1:
    plt.scatter(features_clean.T[sc == i , 0] ,features_clean.T[sc == i , 1] , sc = i)
plt.legend()
plt.title('Spectral Clustering')
plt.show()'''

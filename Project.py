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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA



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
    plt.plot(s[0:20])     
    fig.suptitle('Single Values', fontsize=16)
    plt.show()

    #Get the base3

    

    print('By the graph which is the rank of these data set ? ')
    rank = int(input())
    base = u[:,:rank]
    return rank, base

def get_outliers(data, base, rank, frames):
    
    #
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
    plt.hist(angle_null, bins =300)
    fig = plt.figure()
    fig.suptitle(f'Projection angle on to the subspace: Rank {rank}', fontsize=14)
    plt.hist(angle,bins = 300)
    plt.show()

    # Consider outliers every collumn above 0.8 

    count = 0
    outliers=[]
    inliers=[]
    for i in range (len(angle_null)):
        if angle_null[i] > 0.6:
            #ind_outliers = i
            out = i
            outliers.append(out)
            count = count +1
        if angle[i]>0.5 :
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
    #print(outliers)
    print('Number of outliers', count)
    frames = np.delete(frames, (outliers))
    data_delete = np.delete(data, (outliers), 1)

    return data_delete, outliers, frames

def data_visualization(data , f_outliers):
    # CLEANING DATA

    #Take out the the row of frames

    frames_skeletons = data[0,:]
    data_delete = np.delete(data, 0, 0) 
    

    #Take out frames in features which are outliers

    frame_outliers =[]
    count = 0
    for i in range (len(frames_skeletons)):
        for j in range (len(f_outliers)):
            if f_outliers[j] == frames_skeletons[i]:
                frame_outliers.append(i)
                count=count+1

    data_delete = np.delete(data_delete, (frame_outliers), 1)
    frames_skeletons = np.delete(frames_skeletons, (frame_outliers))

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
    data_delete = np.delete(data_delete, index_probilities, 0)

    #Delete collumns where more than 24 cordinates are zero

    collumn_zeros = np.zeros((data_delete.shape[1]),int)
    index_zeros =[]
    for i in range(collumn_zeros.shape[0]):
        collumn_zeros[i] = np.count_nonzero(data_delete[:,i]==0)
        if collumn_zeros[i] >= 24 :
            index_zeros.append(i)
    data_delete = np.delete(data_delete, index_zeros, 1)
    prob = np.delete(prob, index_zeros, 1)
    frames_skeletons = np.delete(frames_skeletons, index_zeros)

    #Which row has got less zeros 

    row_zeros = np.zeros((data_delete.shape[0]),int)
    for i in range (row_zeros.shape[0]):
        row_zeros[i] = np.count_nonzero(data_delete[i,:]==0)
    fig = plt.figure()
    fig.suptitle('Zeros by row', fontsize=16)
    plt.plot(row_zeros) # Concluimos que as cordennadas na pose 1, que tambem é o centroide, é a posição com menos zeros. Daí utilizarmos a posição 1 para puxar os esqueletos para o seu frame 
    plt.show()

    return data_delete, frames_skeletons, prob

def skeletons_centered(skeletons_clean, skeletons_prob, skeletons_frame):

    index_zero_1 = []
    for j in range (skeletons_clean.shape[1]):
        for i in range (skeletons_clean.shape[0]):
            if skeletons_clean[2,j] == 0 or skeletons_clean[3,j] == 0  : 
                index_zero_1.append(j)
                break
            if i != 2 and i != 3 :

                if skeletons_clean[i, j] == 0:
                    break

                if i % 2 == 0 :
                    skeletons_clean[i, j] = np.subtract(skeletons_clean[i, j],skeletons_clean[2,j])
                    if skeletons_clean[i, j]==0:
                        skeletons_clean[i,j] = 0.0001
                else:
                    skeletons_clean[i, j] = np.subtract(skeletons_clean[i, j],skeletons_clean[3,j])
                    if skeletons_clean[i, j]==0:
                        skeletons_clean[i,j] = 0.0001

    skeletons_clean = np.delete(skeletons_clean,index_zero_1, 1)
    skeletons_frame = np.delete(skeletons_frame, index_zero_1)
    position1_x = skeletons_clean[2,:]
    position1_y = skeletons_clean[3,:]
    skeletons_clean = np.delete(skeletons_clean, 2 ,0 )
    skeletons_clean = np.delete(skeletons_clean, 2 ,0 )
    skeletons_prob = np.delete(skeletons_prob, 1, 0)
    skeletons_prob = np.delete(skeletons_prob, index_zero_1, 1  )

    print(np.count_nonzero(skeletons_clean==0))
    return skeletons_clean, skeletons_frame, skeletons_prob, position1_x, position1_y

def get_missing_data(data_clean, err): 

    # Create Mask

    M = data_clean.astype(bool).astype(int) # Getting the Mask

    # Get the mean of rows without zero 

    mean = np.true_divide(data_clean.sum(1),(data_clean!=0).sum(1))

    # Putting in zero positions the mean 

    for i in range (data_clean.shape[0]):
        for j in range(data_clean.shape[1]):
            if data_clean[i,j]==0 :
                data_clean[i,j]= mean[i]
    
    # PING PONG

    
    e = 100
    Data_til = np.multiply(M, data_clean )+np.multiply((1-M),data_clean)
    r = 8
    norm_data = np.linalg.norm(data_clean)
    while e >= err:
    
        u, s, v = np.linalg.svd(Data_til,full_matrices=False)
        s_lowrank = s[:r]
        s_lowrank = np.diag(s_lowrank)
        vh_lowrank = v[:r]
        u_lowrank = u[:,:r]
        Data_til = np.matmul(np.matmul(u_lowrank,s_lowrank), vh_lowrank)
        Data_i = np.multiply(M, data_clean )+np.multiply((1-M),Data_til)
        e = np.linalg.norm(np.multiply(M,(Data_i-Data_til)))/norm_data 
        Data_til = Data_i
        print(e)
    
        
    return Data_til

#def skeletons_descentralize(ske, pos_x, pos_y):

    

def joining_skeletons_features(ske_data, feat_data, ske_frames, feat_frames, prob):
    
    data_shape_0 = ske_data.shape[0] + feat_data.shape[0]
    data = np.empty([data_shape_0, feat_data.shape[1]])
    zeros = np.zeros([ske_data.shape[0]])

    count=0
    count1=0
    for i in range(len(feat_frames)):
        
        count = np.count_nonzero(feat_frames[i] == ske_frames)
        
        if count > 0 :
            p=0
            index = 0
            for j in range (count):
                #print('feature frame',feat_frames[i],'skeleton frame',ske_frames[count1+j])
                if p < np.sum(prob[:,count1+j], axis=0):
                    p = np.sum(prob[:,count1+j], axis=0)
                    index = count1+j
                #print('igual', p, 'index',count1+j, 'sum probabilities', np.sum(prob[:,count1+j], axis=0))   
            
            data[:,i]=np.append(feat_data[:,i],ske_data[:,index], axis=0)
        else:
            data[:,i]=np.append(feat_data[:,i],zeros, axis=0)

        count1 = count1+count
    #print(feat_data)
    #print(ske_data)
    #print(data)
    return data

def Kmeans_clustering(data_all, frames):

    #pca = PCA(100)
    #data = pca.fit_transform(data_all.T)
    #data = data_all.T
    #plt.plot(df,linestyle='-', marker='o')  
    #plt.show()
    #print(df)
    #df1 = pca.fit_transform(features_clean.T)
    u, s, v = np.linalg.svd(data_all, full_matrices=False)
    s_lowrank = s[:100]
    s_lowrank = np.diag(s_lowrank)
    vh_lowrank = v[0:100,:]
    u_lowrank = u[:,:100]
    lowdim = np.matmul(s_lowrank , vh_lowrank)
    print(lowdim.shape)
#-----------How much clusteres exist-------
    
    '''
    sse = []
    for k in range(1, 15):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(lowdim.T)
        sse.append(kmeans.inertia_)


    plt.style.use("fivethirtyeight")
    plt.plot(range(1, 15), sse)
    plt.xticks(range(1, 15))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()  '''


    kmeans = KMeans( n_clusters=7, random_state=0)
    labels = kmeans.fit_predict(lowdim.T)
    centroids = kmeans.cluster_centers_
    u_labels = np.unique(labels)
    cluster = list(zip(frames, labels))
    print(cluster[0:500])
    print('\n')
    # fig,axis=plt.subplots(1, 2, figsize = (15,5))
    for i in u_labels:
        plt.scatter(lowdim.T[labels == i , 0] , lowdim.T[labels == i , 1] ,s=10, label = i)
    plt.scatter(centroids[:,0] , centroids[:,1] , s = 20,marker='^', color= 'k')
    plt.legend()
    plt.title('Kmeans Clustering')
    plt.show()

    return



#---------------------------MAIN_---------------------------------------------

file_path_features = '/home/goncalo/Desktop/ist/PBDat_Project/girosmallveryslow2.mp4_features.mat'
file_path_skeletons = '/home/goncalo/Desktop/ist/PBDat_Project/esqueletosveryslow.mat'
file_path_skeletons_complete ='/home/goncalo/Desktop/ist/PBDat_Project/esqueletosveryslow_complete.mat'

#---------------------Get Data Matrix-------------------------------------

features  =  get_matrix(file_path_features)
skeletons = get_matrix(file_path_skeletons)
skeletons_complete = get_matrix(file_path_skeletons_complete)
features_frames = np.arange(features.shape[1])

#--------------------Get Rank and Base---------------------------------

features_rank, features_base = get_rank_and_base(features)

#---------------------Getting the outliers------------------------------

features_clean, features_outliers,features_frames  = get_outliers(features,features_base , features_rank,features_frames  )
skeletons_clean, skeletons_frames, skeletons_probabilities = data_visualization(skeletons, features_outliers)
#skeletons_complete_clean, skeletons2 = data_visualization(skeletons_complete, features_outliers)
features_rank, features_base = get_rank_and_base(features_clean)
#-------------------Center Skeletons on Pose 1 -----------------------

skeletons_cent,skeletons_frames,skeletons_probabilities, pose1_x, pose1_y = skeletons_centered(skeletons_clean, skeletons_probabilities , skeletons_frames)
#skeletons_rank, skeletons_base = get_rank_and_base(skeletons_clean)
#Skeletons_clean2 , Skeletons_outliers = get_outliers(skeletons_clean, skeletons_base, skeletons_rank )

#----------------------Missing Data replaced--------------------------

Skeletons_fill = get_missing_data(skeletons_cent, 0.037)
#Skeletons_fill1 = get_missing_data(skeletons_clean, 0.05)

#----------------------Take skeleotns of Center----------------------------------

#skeletons_com = skeletons_descentralize(Skeletons_fill, pose1_x,pose1_y)

#---------------------------Joining Skeletons and Features--------------------

all_data = joining_skeletons_features(Skeletons_fill, features_clean, skeletons_frames, features_frames, skeletons_probabilities)
#all_data1= joining_skeletons_features(Skeletons_fill1, features_clean, skeletons_frames, features_frames, skeletons_probabilities)
#all_data1= joining_all_skeletons_features(Skeletons_fill1, features_clean, skeletons_frames, features_frames)
#---------------------------KNN Clustering-----------------------

Kmeans_clustering(all_data, features_frames)
#Kmeans_clustering(all_data1, features_frames)
#Kmeans_clustering(skeletons_cent, skeletons_frames)

#-------------------------Spectral Clustering-----------------------
'''
from sklearn.cluster import SpectralClustering

sc = SpectralClustering(n_clusters=6)
labels1 = sc.fit_predict(df)
u_labels1 = np.unique(labels1)

print(features_frames[4300:4400])
print(labels1[4300:4400])
for i in u_labels1:
    axis[1].scatter(df[labels1 == i , 0] ,df[labels1 == i , 1] , label = i)
axis[1].legend()
axis[1].set_title('Spectral Clustering')
plt.show()


u, s, vh = np.linalg.svd(features_clean)

s_lowrank = s[:2]
#s_lowrank = np.diag(s_lowrank)
vh_lowrank = vh[:2]
u_lowrank = u[:,:100]

feat_coef = s_lowrank @ vh_lowrank

plt.plot(feat_coef,linestyle='-', marker='o')
plt.show()
'''

    
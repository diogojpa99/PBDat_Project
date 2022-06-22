'''
There is some code in comentary because it is about visualization, so it was depent on how the PC had the files saved. But do not worry 
because we are going to show them on the video apresentation.

Names: 
Gonçalo Mesquita 94196
Diogo Araújo 93906
Paulo Cruz 93150

'''

import scipy.io as sio
import pandas as pd
import numpy as np
from numpy.linalg import svd
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2 




#-------------------------FUNCTIONS-----------------

def get_matrix(file_path):

    mat = sio.loadmat(file_path)
    count = 0
    for i in mat.keys():
        if count == 3:
            df = pd.DataFrame(mat[i])
        count+=1
    Matrix = df.to_numpy(dtype = np.float64)
    return Matrix

def draw_poses(data ,img, missing_data=False):

    PAIRS = [(0,1),(0,14),(0,15),(1,2),(1,5),(1,8),(1,11),(2,3),(3,4),(5,6),(6,7),(8,9),(9,10),(11,12),(12,13),(14,16),(15,17)]
    x_pose = data[1:54:3, 0]
    y_pose = data[2:54:3, 0]
    colour = (255,255,0)
    circle = 3
    hand = 5
    line_with = 2
    img_width = img.shape[1]
    img_height = img.shape[0]
    validation = (x_pose + y_pose).astype(bool)

    for kp1,kp2 in PAIRS:
        if missing_data or validation[kp1] & validation[kp2]:
            x1 = int(x_pose[kp1] * img_width )
            y1 = int(y_pose[kp1] * img_height )
            x2 = int(x_pose[kp2] * img_width )
            y2 = int(y_pose[kp2] * img_height )

            cv2.line(img, (x1,y1), (x2,y2), colour, line_with)
            cv2.circle(img, (x1,y1), circle, colour, -1)
            cv2.circle(img, (x2,y2), circle, colour, -1)
            #plt.plot((x1,y1), (x2,y2))
    #cv2.imshow('img', img)
    plt.imshow(img)
    plt.show()
    return

def plot_outliers_features_images(outlier, rank, count):
    
    # reading images
    fig = plt.figure(figsize=(20, 10))
    
    for i in range(200):
        Image = cv2.imread(f'/home/goncalo/Desktop/ist/PBDat_Project/frames/frame{outlier[i]}.jpg')
        fig.add_subplot(10,20, i+1)
        plt.imshow(Image)
    plt.set_title(f'Outlier of features with rank: {rank}, have {count} outliers')

    return

def get_outliers_features(data, rank, frames, err):

    # Get the base from svd. Why svd ?? Gives us the best possible base, with less error

    u, s, v = np.linalg.svd(data, full_matrices=False)

    # Obseve which is the rank of the matrix. Single values give us the importance of each linear indepent collumn

    fig,axis=plt.subplots(1, 2, figsize = (13,9))
    axis[0].plot(s) 
    axis[1].plot(s[0:30])    
    fig.suptitle('Single Values of features', fontsize=16)
    plt.show()
    
    fig,axis=plt.subplots(3, 2, figsize = (13,9))

    for i in range (len(rank)):

    #Get the base. SVD ARRANJA A MELHOR BASE, COM MENOS ERRO

        base = u[:,:rank[i]]
        
    #Calculate orthogonal projection on to the Subspace and ortigonal projection on to the null subspace

        Pi = np.matmul(base, base.T)
        PiN = np.identity(len(Pi))-Pi

    # Calculate the angle between the vectors of the collumn and PI and PIN

        angle = (np.linalg.norm(np.matmul(Pi,data), axis =0))
        angle_null = (np.linalg.norm(np.matmul(PiN,data), axis =0))
        angle = angle/(np.linalg.norm(data, axis =0))
        angle_null = angle_null/(np.linalg.norm(data, axis =0))

    # Plot the normalization of the angles. Observe which are too far away from the null subspace
        
        axis[i, 0].hist(angle_null, bins =300)
        axis[i, 0].set_title(f'Angle projection on to the null subspace: Rank {rank[i]}', fontsize=12)
        #fig = plt.figure()
        axis[i, 1].hist(angle, bins = 300)
        axis[i, 1].set_title(f'Angle projection on to the subspace: Rank {rank[i]}', fontsize=12)

    # Consider outliers every collumn above err

        count = 0
        outl=[]
        for j in range (len(angle_null)):
            if angle_null[j] > err[i]:
                #ind_outliers = i
                outl.append(j)
                count = count +1

    # Watch the outliers,

        #plot_outliers_features_images(outl, rank, count)
        
    #Taking into account the outliers
        if i == 1 : 
            outliers = outl
    fig.suptitle('Features projections', fontsize=16)
    plt.show()
        
    
    # Delete the outliers form the features and their respective frames

    frames = np.delete(frames, outliers)
    data_delete = np.delete(data, outliers, 1)
    
    return data_delete, outliers, frames

def get_outliers_skeletons(data , f_outliers):
    # CLEANING DATA

    #Take out the the row of frames

    frames_skeletons = data[0,:]
    data_delete = np.delete(data, 0, 0) 
    
    #Take out frames in features which are outliers

    frame_outliers =[]
    for i in range (len(frames_skeletons)):
        for j in range (len(f_outliers)):
            if f_outliers[j] == frames_skeletons[i]:
                frame_outliers.append(i)

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

    #Which row has got less zeros 

    row_zeros = np.zeros((data_delete.shape[0]),int)
    for i in range (row_zeros.shape[0]):
        row_zeros[i] = np.count_nonzero(data_delete[i,:]==0)
    fig = plt.figure()
    fig.suptitle('zeros in Skeletons rows', fontsize=16)
    plt.bar(np.arange(36) ,row_zeros) # Concluimos que as cordennadas na pose 1, que tambem é o centroide, é a posição com menos zeros. Daí utilizarmos a posição 1 para puxar os esqueletos para o seu frame 
    plt.show()

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

    #Which collumn has got less zeros 

    fig = plt.figure()
    fig.suptitle('Zeros in skeletons collumns', fontsize=16)
    plt.bar(np.arange(20345) ,collumn_zeros)  
    plt.show()

    return data_delete, frames_skeletons, prob

def skeletons_centered(skeletons_clean, skeletons_prob, skeletons_frame):
    
    #Center skeletons in Pose 1

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
    skeletons_clean = np.delete(skeletons_clean, 2 ,0 )
    skeletons_clean = np.delete(skeletons_clean, 2 ,0 )
    skeletons_prob = np.delete(skeletons_prob, 1, 0)
    skeletons_prob = np.delete(skeletons_prob, index_zero_1, 1  )
    #pose_1 = np.zeros((2,skeletons_clean.shape[1]))
    #pose_1[0,:], pose_1[1,:] = skeletons_clean[2,:] , skeletons_clean[3,:]
    return skeletons_clean, skeletons_frame, skeletons_prob

def complete_missing_data(data_clean, err, r): 
    
    skell_fill = np.zeros((len(r), data_clean.shape[0], data_clean.shape[1]))
    for m in range(len(r)):

    # Create Mask--> 1 if visble 0 if not visible

        M = data_clean.astype(bool).astype(int) # Getting the Mask

    # Get the mean of rows without zero 

        mean = np.true_divide(data_clean.sum(1),(data_clean!=0).sum(1))

    # Replacing zeros by the mean 
    
        data = np.array(data_clean)
        for i in range (data_clean.shape[0]):
            for j in range(data_clean.shape[1]):
                if data_clean[i,j]==0 :
                    data[i,j]= mean[i]
        
    # PING PONG
        
        e = 100
        data_mask = np.multiply(M, data )+np.multiply((1-M),data)
        norm_data = np.linalg.norm(data)
        while e >= err[m]:
    
            u, s, v = np.linalg.svd(data_mask,full_matrices=False)
            s_lowrank = s[:r[m]]
            s_lowrank = np.diag(s_lowrank)
            vh_lowrank = v[:r[m]]
            u_lowrank = u[:,:r[m]]
            data_mask = np.matmul(np.matmul(u_lowrank,s_lowrank), vh_lowrank)
            data_i = np.multiply(M, data )+np.multiply((1-M),data_mask)
            e = np.linalg.norm(np.multiply(M,(data_clean-data_mask)))/norm_data 
            data_mask = data_i
            #print(e)

        skell_fill[m] = data_mask
    return skell_fill

def joining_skeletons_features(ske_data, feat_data, ske_frames, feat_frames, prob):
    
    data_shape_0 = ske_data.shape[0] + feat_data.shape[0]
    data = np.empty([data_shape_0, feat_data.shape[1]])
    zeros = np.zeros([ske_data.shape[0]])

    #Joining the most probable skeleton to the feature it belongs

    count=0
    count1=0
    for i in range(len(feat_frames)):
        
        count = np.count_nonzero(feat_frames[i] == ske_frames)
        
        if count > 0 :
            p=0
            index = 0
            for j in range (count):
                if p < np.sum(prob[:,count1+j], axis=0):
                    p = np.sum(prob[:,count1+j], axis=0)
                    index = count1+j 
            
            data[:,i]=np.append(feat_data[:,i],ske_data[:,index], axis=0)
        else:
            data[:,i]=np.append(feat_data[:,i],zeros, axis=0)

        count1 = count1+count

    return data

def plot_clusters(data, kmeans, labels, max_ind, m, frames, min_ind):
    
    centroids = kmeans.cluster_centers_
    u_labels = np.unique(labels)

    for i in u_labels:
        axis[m].scatter(data.T[labels == i , 0] , data.T[labels == i , 1] ,s=10, label = i)
    axis[m].scatter(centroids[:,0] , centroids[:,1] , s = 20,marker='^', color= 'black')
    axis[m].scatter(data.T[min_ind, 0], data.T[min_ind, 1] , s = 30,marker='x', color= 'black')
    axis[m].scatter(data.T[max_ind, 0], data.T[max_ind, 1] , s = 30,marker='o', color= 'black')
    axis[m].legend()

def Kmeans_clustering(data, frames, n_clust, m):

    
    # Reduce the dimension to 100
    u, s, v = np.linalg.svd(data, full_matrices=False)
    s_lowrank = s[:100]
    s_lowrank = np.diag(s_lowrank)
    vh_lowrank = v[0:100,:]
    lowdim = np.matmul(s_lowrank , vh_lowrank)



    #Calculate the clusters

    kmeans = KMeans( n_clusters=n_clust, random_state=0)
    labels = kmeans.fit_predict(lowdim.T)
    clustering = list(zip(frames, labels))
    
    #Get the fardest point from centroid 

    lowdim_dist = kmeans.transform(lowdim.T)**2
    max_indices = []
    for label in np.unique(kmeans.labels_):
        data_label_indices = np.where(labels==label)[0]
        max_label_idx = data_label_indices[np.argmax(lowdim_dist[labels==label].sum(axis=1))]
        max_indices.append(max_label_idx)
    
    #Get centroids indexes

    min_indices = []
    for i in range(kmeans.cluster_centers_.shape[0]):
        min_indices.append(np.argmin(np.linalg.norm(lowdim.T - kmeans.cluster_centers_[i], axis=1)))

    #Plot Clustering

    plot_clusters(lowdim,  kmeans , labels, max_indices, m, frames, min_indices)

    return clustering

#--------------------------MAIN---------------------

file_path_features = 'girosmallveryslow2.mp4_features.mat'
file_path_skeletons = 'esqueletosveryslow.mat'
file_path_skeletons_complete ='esqueletosveryslow_complete.mat'

#---------------------Get Data Matrix-------------------------------------

features  =  get_matrix(file_path_features)
skeletons = get_matrix(file_path_skeletons)
skeletons_complete = get_matrix(file_path_skeletons_complete)
#img = cv2.imread('/home/goncalo/Desktop/ist/PBDat_Project/frames/frame0.jpg')
#draw_poses(skeletons ,img, False)

#---------------------Parameters-----------------------------------

features_frames = np.arange(features.shape[1])
features_rank = np.array((2,10,15))
skeletons_rank = np.array([4, 8 , 12])
erros_rank_features =  np.array([0.67, 0.58 , 0.55])
erros_skeletons =  np.array([0.092, 0.050 , 0.047])
n_clusters = np.array([5, 7, 11])

#--------------------Get Base and watch rank-----------------------

features_clean, features_outliers,features_frames = get_outliers_features(features, features_rank, features_frames, erros_rank_features )
skeletons_clean, skeletons_frames, skeletons_probabilities = get_outliers_skeletons(skeletons, features_outliers)

#-------------------Center Skeletons on Pose 1 -----------------------

skeletons_cent,skeletons_frames,skeletons_probabilities= skeletons_centered(skeletons_clean, skeletons_probabilities , skeletons_frames)

#----------------------Missing Data replaced--------------------------

skeletons_fill = complete_missing_data(skeletons_cent, erros_skeletons, skeletons_rank) #We chose the one with rank 8 because when we did the plot of the skeletons in matlab it was the best. That is why we use skeletons_fill[1]

#-----------------------KMeans Clustering Features and skeletons----------------------

fig,axis=plt.subplots(1, 2, figsize = (15,5))
Kmeans_clustering(features_clean, features_frames, 4, 0)
Kmeans_clustering(skeletons_fill[1], features_frames, 4, 1)
axis[0].set_title('Kmeans Clustering for features')
axis[1].set_title('Kmeans Clustering for skeletons')
plt.show()

#---------------------------Joining Skeletons and Features--------------------

all_data = joining_skeletons_features(skeletons_fill[1], features_clean, skeletons_frames, features_frames, skeletons_probabilities)

#-----------------------KMeans Clustering ----------------------

fig,axis=plt.subplots(1, 3, figsize = (15,5))
Kmeans_clustering(all_data, features_frames, 5, 0)
Kmeans_clustering(all_data, features_frames, 7, 1)
Kmeans_clustering(all_data, features_frames, 11, 2)
axis[0].set_title('5 clusters')
axis[1].set_title('7 clusters')
axis[2].set_title('11 clusters')
fig.suptitle('Kmeans for features with skeletons', fontsize=16)
plt.show()

#print(cluster)
sio.savemat('Features_Outliers_rank10.mat', mdict={'features_outliers':features_outliers })

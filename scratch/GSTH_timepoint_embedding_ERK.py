from os import listdir

import numpy as np
import h5py
from numpy import linalg as LA
from sklearn.decomposition import PCA
import scipy
import phate
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram

import scipy.io as sio
import matplotlib.pyplot as plt
import phate

def lazy_random_walk(adj):
    #if d == 0, P = 0
    P_array = []
    d = adj.sum(0)
    P_t = adj/d
    P_t[np.isnan(P_t)] = 0
    P = 1/2*(np.identity(P_t.shape[0])+P_t)
    #for i in range(0,t,2):
    #for i in range(t+1):
    #for i in [0]:
    #P_array.append(LA.matrix_power(P,i))
    #return P_array,P
    return P

def graph_wavelet(P):
    psi = []
    #for d1 in range(1,t,2):
    #for d1 in range(1,t+1):
    for d1 in [1,2,4,8,16]:
        W_d1 = LA.matrix_power(P,d1) - LA.matrix_power(P,2*d1)
        psi.append(W_d1)
    return psi

def zero_order_feature(A,ro):
    F0= np.matmul(LA.matrix_power(A,16),ro)
    
    return F0

def first_order_feature(A,u):
    F1 = np.matmul(LA.matrix_power(A,16),np.abs(u))
    #F = []
    #F.append(np.sum(np.matmul(P,u[0]),1))
    #for i in range(1,t):
    F1 = np.concatenate(F1,0)
    return F1

def selected_second_order_feature(A,W,u):
    u1 = np.einsum('ij,ajt ->ait',W[1],u[0:1])
    for i in range(2,len(W)):
        u1 = np.concatenate((u1,np.einsum('ij,ajt ->ait',W[i],u[0:i])),0)
    u1 = np.abs(u1)
    F2 = np.matmul(LA.matrix_power(A,16),u1)
    #F2.append(np.sum(np.einsum('ij,ajt ->ait',W[i],u[0:i]),1).reshape(len(u[0:i])*F,1))
    #F2 = np.sum(np.einsum('ijk,akt ->iajt',P,F1),2).reshape(len(P)*len(F1)*F,1)
    F2 = np.concatenate(F2,0)
    return F2

def generate_timepoint_feature(adj,ro):
    #with zero order, first order and second order features
    #shall consider only zero and first order features
    P = lazy_random_walk(adj)
    
    
    W = graph_wavelet(P)
    u = np.abs(np.matmul(W,ro))
    
    F0 = zero_order_feature(P,ro)
    F1 = first_order_feature(P,u)
    #F2 = second_order_feature(W,u,P[0],t,F)
    F2 = selected_second_order_feature(P,W,u)
    #F3 = selected_third_order_feature(W,u,P[0],t,F)
    F = np.concatenate((F0,F1),axis=0)
    F = np.concatenate((F,F2),axis=0)
    #F = np.concatenate((F1,F2),axis=0)
    #F = np.concatenate((F,F3),axis=0)
    return F

folders = ['/home/dhanajayb/Downloads/ERK_dataset/Output/Pos3Registration/processed']

phate_data_all = []
time_point_all = []
files_all = []
phate_operator_all = []

for folder in folders:

    files = listdir(folder)

    for file in files:

        print('processing file',file)
        files_all.append(file)
        new_data = sio.loadmat(folder+'/'+file)
        cells_mean = np.transpose(new_data['cells_mean'])
        cells_mean = np.nan_to_num(cells_mean)
        graph_size = cells_mean.shape[1]
        
        node_pool = []
        adj = np.zeros((graph_size,graph_size))
        for i,l in enumerate(new_data['nbList']):
            for j in l[0][0]:
                node_pool.append(j)
                adj[i][j-1] = 1
                adj[j-1][i] = 1

        features = cells_mean[:]
        normalzied_features = (features-np.min(features,0).reshape(1,-1))/np.min(features,0).reshape(1,-1)
        normalzied_features_transpose = normalzied_features.transpose()
        feature_z = scipy.stats.mstats.zscore(normalzied_features_transpose,0)

        print(np.min(features,0))
        
        print('calculate scattering')
        scattering_feature = generate_timepoint_feature(adj,feature_z)
        time_point = np.arange(scattering_feature.shape[1])
        scattering_feature_transpose = scattering_feature.transpose()

        print(scattering_feature_transpose)

        print('calculate phate')
        phate_operator = phate.PHATE(n_components=3,knn=20)
        phate_data = phate_operator.fit_transform(scattering_feature_transpose)
        
        phate_data_all.append(phate_data)
        time_point_all.append(time_point)
        phate_operator_all.append(phate_operator)

#calculate VietorisRipsPersistence
VR = VietorisRipsPersistence(homology_dimensions=[0, 1, 2])
diagrams = VR.fit_transform(phate_data_all)
np.save('diagram_all',diagrams)

#plot phate trajectory
fig = plt.figure(figsize=(16,8))
ax1 = plt.subplot2grid(shape=(1,1), loc=(0,0),projection='3d')
im1 = ax1.scatter(phate_data_all[0][:,0],phate_data_all[0][:,1],phate_data_all[0][:,2],c=time_point_all[0],s=5,cmap='inferno')
fig.colorbar(im1, ax=ax1,fraction=0.015,pad=0.08)
plt.savefig('cpDataTracked',dpi=600)

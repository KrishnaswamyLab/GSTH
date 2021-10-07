from os import listdir
import numpy as np
import h5py
from numpy import linalg as LA
import scipy
import phate

from collections import defaultdict


import scipy.io as sio
from collections import Counter
import matplotlib.pyplot as plt
import phate

def lazy_random_walk(adj):
    #if d == 0, P = 0
    P_array = []
    d = adj.sum(0)
    P_t = adj/d
    P_t[np.isnan(P_t)] = 0
    P = 1/2*(np.identity(P_t.shape[0])+P_t)
 
    return P



def graph_wavelet(P):
    psi = []
    for d1 in [1,2,4,8,16]:
        W_d1 = LA.matrix_power(P,d1) - LA.matrix_power(P,2*d1)
        psi.append(W_d1)
    return psi



def zero_order_feature(A,ro):
    F0= np.matmul(LA.matrix_power(A,16),ro)
    
    return F0



def first_order_feature(A,u):
    F1 = np.matmul(LA.matrix_power(A,16),np.abs(u))
    F1 = np.concatenate(F1,1)
    return F1



def selected_second_order_feature(A,W,u):
    u1 = np.einsum('ij,ajt ->ait',W[1],u[0:1])
    for i in range(2,len(W)):
        u1 = np.concatenate((u1,np.einsum('ij,ajt ->ait',W[i],u[0:i])),0)
    u1 = np.abs(u1)
    F2 = np.matmul(LA.matrix_power(A,16),u1)
    F2 = np.concatenate(F2,1)
    return F2





def generate_celluar_feature(adj,ro):
    #with zero order, first order and second order features
    P = lazy_random_walk(adj)
    W = graph_wavelet(P)
    u = np.abs(np.matmul(W,ro))
    
    F0 = zero_order_feature(P,ro)
    F1 = first_order_feature(P,u)
    F2 = selected_second_order_feature(P,W,u)
    F = np.concatenate((F0,F1),axis=1)
    F = np.concatenate((F,F2),axis=1)
    
    return F


folders = ['Dataset']


phate_data_all = []
cell_cycle_all = []
lingering_time_all = []
neighbor_size_all = []
phate_operator_all = []
features_all = []
n_flash_all = []
for folder in folders:
    files = listdir(folder)
    for file in files:
        print('processing file',file)
        new_data = sio.loadmat(folder+'/'+file)
        graph_size = new_data['cells_mean'].shape[1]
        cell_cycle = new_data['events_info'][:,-1]
        n_flash = new_data['events_info'][:,1]
        print(cell_cycle)
        
        neighbor_size = []
        for i in new_data['bins4']:
            neighbor_size.append(i[0])
        
        
        
        linging_time_info = defaultdict(list)
        for i,v in enumerate(new_data['updatedEvents'][:,0]):
            linging_time_info[int(v)].append(new_data['updatedEvents'][i,2])
        max_linging_time_info = []
        ave_linging_time_info = []
        for i in range(len(new_data['events_info'])):
            if i+1 in linging_time_info:
                max_linging_time_info.append(max(linging_time_info[i+1]))
                ave_linging_time_info.append(np.mean(linging_time_info[i+1]))
            else:
                max_linging_time_info.append(0)
                ave_linging_time_info.append(0)
        
        
        
        node_pool = []
        adj = np.zeros((graph_size,graph_size))
        for i,l in enumerate(new_data['nbList']):
            for j in l[0][0]:
                node_pool.append(j)
                adj[i][j-1] = 1
                adj[j-1][i] = 1
                    
        features = new_data['cells_mean'][:]
        normalzied_features = (features-np.min(features,0).reshape(1,-1))/np.min(features,0).reshape(1,-1)
        normalzied_features_transpose = normalzied_features.transpose()
        feature_z = scipy.stats.mstats.zscore(normalzied_features_transpose,0)
        print('calculate scattering')
        scattering_feature = generate_celluar_feature(adj,feature_z)
        print(scattering_feature.shape)
        
        print('calculate phate')
        phate_operator = phate.PHATE(n_components=3,knn=50)
        phate_data = phate_operator.fit_transform(scattering_feature)
        
        phate_data_all.append(phate_data)
        cell_cycle_all.append(cell_cycle)
        phate_operator_all.append(phate_operator)
        features_all.append(normalzied_features)
        neighbor_size_all.append(neighbor_size)
        lingering_time_all.append(max_linging_time_info)
        n_flash_all.append(n_flash)




#an example of plotting phate trajectory of celluar embeddings according to the cell cycle information
cols = []
for v in cell_cycle_all[0]:
    if v==1:
        cols.append('navy')
    else:
        cols.append('orange')


fig = plt.figure(figsize=(16,8))
ax1 = plt.subplot2grid(shape=(1,1), loc=(0,0),projection='3d')
ax1.view_init(30, 110)
im1 = ax1.scatter(phate_data_all[0][:,0],phate_data_all[0][:,1],phate_data_all[0][:,2],c=cols,s=5)
#plt.savefig('cell_clustering',dpi=600)

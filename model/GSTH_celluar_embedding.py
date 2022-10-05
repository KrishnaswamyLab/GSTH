from os import listdir

import numpy as np

import phate

import h5py
import scipy
import scipy.io as sio

from collections import Counter
from collections import defaultdict

import matplotlib.pyplot as plt

from scattering import *


folders = ['/home/dhanajayb/Downloads/CaSignalingDataset/K10/']

phate_data_all = []
cell_cycle_all = []
lingering_time_all = []
neighbor_size_all = []
phate_operator_all = []
features_all = []
n_flash_all = []

for folder in folders[:2]:

    files = listdir(folder)
    
    for file in files:

        print('processing file',file)
        new_data = sio.loadmat(folder+'/'+file)
        graph_size = new_data['cells_mean'].shape[1]
        cell_cycle = new_data['events_info'][:,7]
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

        # plot phate trajectory of celluar embeddings according to the cell cycle information
        cols = []
        for v in cell_cycle:
            if v==1:
                cols.append('navy')
            else:
                cols.append('orange')

        fig = plt.figure(figsize=(16,8))
        ax1 = plt.subplot2grid(shape=(1,1), loc=(0,0),projection='3d')
        ax1.view_init(30, 110)
        im1 = ax1.scatter(phate_data[:,0],phate_data[:,1],phate_data[:,2],c=cols,s=5)
        plt.savefig('cell_embedding_'+file+'.png', dpi=600)
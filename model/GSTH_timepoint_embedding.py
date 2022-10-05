import os, sys, time

from os import listdir

import numpy as np

import h5py
import scipy
import scipy.io as sio

import phate
from sklearn.decomposition import PCA

from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram

import matplotlib.pyplot as plt

from scattering import *

SHUFFLE = False
SAMPLE = True
DIST = "normal"
rng_seed = int(time.time())

folders = ['/home/dhanajayb/Downloads/GSTH_test/mat']

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
        graph_size = new_data['cells_mean'].shape[1]
        
        node_pool = []
        adj = np.zeros((graph_size,graph_size))
        for i,l in enumerate(new_data['nbList']):
            for j in l[0][0]:
                node_pool.append(j)
                adj[i][j-1] = 1
                adj[j-1][i] = 1
                    
        features = new_data['cells_mean'][:]

        if SHUFFLE:
            np.random.seed(rng_seed)
            features = features[:, np.random.permutation(features.shape[1])]

        if SAMPLE:
            np.random.seed(rng_seed)
            if DIST == "uniform":
                sig_max = np.percentile(features[:], 0.8)
                sig_min = np.percentile(features[:], 0.2)
                features = np.random.uniform(sig_min, sig_max, features.shape)
            elif DIST == "normal":
                sig_mean = np.mean(features[:])
                sig_std = np.std(features[:])
                features = np.random.normal(sig_mean, sig_std, features.shape)      
            else:
                print("ERROR: Unknown distribution.")
                sys.exit(1)

            features = features[:, np.random.permutation(features.shape[1])]

        normalzied_features = (features-np.min(features,0).reshape(1,-1))/np.min(features,0).reshape(1,-1)
        normalzied_features_transpose = normalzied_features.transpose()
        feature_z = scipy.stats.mstats.zscore(normalzied_features_transpose,0)

        print('calculate scattering')
        scattering_feature = generate_timepoint_feature(adj,feature_z)
        time_point = np.arange(scattering_feature.shape[1])
        scattering_feature_transpose = scattering_feature.transpose()

        print('calculate phate')
        phate_operator = phate.PHATE(n_components=3,knn=20)
        phate_data = phate_operator.fit_transform(scattering_feature_transpose)
        
        phate_data_all.append(phate_data)
        time_point_all.append(time_point)
        phate_operator_all.append(phate_operator)

# calculate VietorisRipsPersistence
VR = VietorisRipsPersistence(homology_dimensions=[0, 1, 2])
diagrams = VR.fit_transform(phate_data_all)
if SHUFFLE:
    np.save('diagram_shuffle_'+str(rng_seed),diagrams)
elif SAMPLE:
    np.save('diagram_sample_'+DIST+'_'+str(rng_seed),diagrams)
else:
    np.save('diagram_all',diagrams)

# plot phate trajectory
fig = plt.figure(figsize=(16,8))
ax1 = plt.subplot2grid(shape=(1,1), loc=(0,0),projection='3d')
im1 = ax1.scatter(phate_data_all[0][:,0],phate_data_all[0][:,1],phate_data_all[0][:,2],c=time_point_all[0],s=5,cmap='inferno')
fig.colorbar(im1, ax=ax1,fraction=0.015,pad=0.08)
if SHUFFLE:
    plt.savefig('timelapse_embedding_shuffle_'+str(rng_seed),dpi=600)
elif SAMPLE:
    plt.savefig('timelapse_embedding_sample_'+DIST+'_'+str(rng_seed),dpi=600)
else:
    plt.savefig('timelapse_embedding',dpi=600)
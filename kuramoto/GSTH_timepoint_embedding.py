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
SAMPLE = False
DIST = "normal"
rng_seed = int(time.time())

folders = ['/home/dhanajayb/Downloads/Kuramoto/']

for folder in folders:

    files = listdir(folder)

    for file in files:

        fname = os.path.splitext(os.path.basename(file))[0]
        print('processing file', fname)

        dat = sio.loadmat(folder+'/'+file)
        new_data = dat['theta'].T
        graph_size = new_data.shape[1]
        
        node_pool = []
        adj = np.ones((graph_size,graph_size))
                    
        features = new_data[:]

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

        #normalized_features = (features-np.min(features,0).reshape(1,-1))/np.min(features,0).reshape(1,-1)
        #normalized_features_transpose = normalized_features.transpose()
        normalized_features_transpose = features.transpose()
        feature_z = scipy.stats.mstats.zscore(normalized_features_transpose,0)

        print('calculate scattering')
        scattering_feature = generate_timepoint_feature(adj,feature_z)
        time_point = np.arange(scattering_feature.shape[1])
        scattering_feature_transpose = scattering_feature.transpose()

        print('calculate phate')
        phate_operator = phate.PHATE(n_components=3,knn=20)
        phate_data = phate_operator.fit_transform(scattering_feature_transpose)

        # calculate VietorisRipsPersistence
        VR = VietorisRipsPersistence(homology_dimensions=[0, 1])   # DB: removed 2
        diagrams = VR.fit_transform([phate_data])
        if SHUFFLE:
            np.save(fname + 'diagram_shuffle_' + str(rng_seed), diagrams)
        elif SAMPLE:
            np.save(fname + 'diagram_sample_' + DIST + '_' + str(rng_seed), diagrams)
        else:
            np.save(fname + '_diagram_all', diagrams)

        # plot phate trajectory
        fig = plt.figure(figsize=(16,8))
        ax1 = plt.subplot2grid(shape=(1,1), loc=(0,0), projection='3d')
        im1 = ax1.scatter(phate_data[:,0], phate_data[:,1], phate_data[:,2], c=time_point, s=5, cmap='inferno')
        fig.colorbar(im1, ax=ax1, fraction=0.015, pad=0.08)
        if SHUFFLE:
            plt.savefig(fname + '_timelapse_embedding_shuffle_' + str(rng_seed), dpi=600)
        elif SAMPLE:
            plt.savefig(fname + '_timelapse_embedding_sample_' + DIST + '_' + str(rng_seed), dpi=600)
        else:
            plt.savefig(fname + '_timelapse_embedding', dpi=600)
        plt.close()
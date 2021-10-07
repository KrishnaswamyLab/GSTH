import umap
import numpy as np
import h5py
import scipy.io as sio
import networkx as nx
from collections import Counter

import h5py
from numpy import linalg as LA
from sklearn.decomposition import PCA
import scipy


from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import phate



new_data = sio.loadmat('cdkn1b/positive/hyperstack_200301_15-23-52_green_crop2_ALL_CELLS_plus.mat')




node_pool = []
adj = np.zeros((1867,1867))
for i,l in enumerate(new_data['nbList']):
    if i==1024:
        print(l)
    for j in l[0][0]:
        #print(j)
        #if j not in node_pool:
        node_pool.append(j)
        adj[i][j-1] = 1
        adj[j-1][i] = 1


toy_feature = np.zeros(1867)
toy_feature[256] = 1
G = nx.from_numpy_matrix(adj)
L = nx.normalized_laplacian_matrix(G)
M = L.toarray()


simulate_signal = []
simulate_signal.append(toy_feature)
temp = toy_feature
for i in range(300):
    temp = np.matmul(M,temp)
    simulate_signal.append(temp)


simulate_signal = np.array(simulate_signal)





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
    for d1 in [1,2,4,8]:
        W_d1 = LA.matrix_power(P,d1) - LA.matrix_power(P,2*d1)
        psi.append(W_d1)
    return psi



def zero_order_feature(A,ro):
    F0= np.matmul(LA.matrix_power(A,8),ro)
    
    return F0



def first_order_feature(A,u):
    F1 = np.matmul(LA.matrix_power(A,8),np.abs(u))
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
    F2 = np.matmul(LA.matrix_power(A,8),u1)
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



simulate_signal_ = simulate_signal+np.random.normal(0,0.001,(301,1867))
combined_signal = np.concatenate((simulate_signal,simulate_signal_),0)
combined_signal = combined_signal.transpose()
feature_z = scipy.stats.mstats.zscore(combined_signal,0)

scattering_feature = generate_timepoint_feature(adj,feature_z)
scattering_feature = scattering_feature.transpose()



phate_operator1= phate.PHATE(n_components=3,knn=20)
phate_data1 = phate_operator3.fit_transform(scattering_feature) 


fig = plt.figure(figsize=(16,8))
ax1 = plt.subplot2grid(shape=(1,1), loc=(0,0),projection='3d')
im1 = ax1.scatter(phate_data1[:,0],phate_data1[:,1],phate_data1[:,2],c=np.arange(602),s=5,cmap='inferno')
fig.colorbar(im1, ax=ax1,fraction=0.015,pad=0.08)



feature_tsne= TSNE(n_components=3).fit_transform(scattering_feature)
fig = plt.figure(figsize=(16,8))
ax1 = plt.subplot2grid(shape=(1,1), loc=(0,0),projection='3d')
im1 = ax1.scatter(feature_tsne[:,0],feature_tsne[:,1],feature_tsne[:,2],c=np.arange(602),s=5,cmap='inferno')
fig.colorbar(im1, ax=ax1,fraction=0.015,pad=0.08)
#plt.savefig('toy2_tsne',dpi=600)


phate_operator2 = phate.PHATE(n_components=3,knn=20)
phate_data2 = phate_operator2.fit_transform(combined_signal.transpose()) 
fig = plt.figure(figsize=(16,8))
ax1 = plt.subplot2grid(shape=(1,1), loc=(0,0),projection='3d')
im1 = ax1.scatter(phate_data2[:,0],phate_data2[:,1],phate_data2[:,2],c=np.arange(602),s=5,cmap='inferno')
fig.colorbar(im1, ax=ax1,fraction=0.015,pad=0.08)
#plt.savefig('toy2_phate',dpi=600)



pca_feature = PCA(n_components=3).fit_transform(scattering_feature)
fig = plt.figure(figsize=(16,8))
ax1 = plt.subplot2grid(shape=(1,1), loc=(0,0),projection='3d')
im1 = ax1.scatter(pca_feature[:,0],pca_feature[:,1],pca_feature[:,2],c=np.arange(602),s=5,cmap='inferno')
fig.colorbar(im1, ax=ax1,fraction=0.015,pad=0.08)



umap_operator= umap.UMAP(n_components=3)
umap_feature = umap_operator.fit_transform(scattering_feature)
fig = plt.figure(figsize=(16,8))
ax1 = plt.subplot2grid(shape=(1,1), loc=(0,0),projection='3d')
im1 = ax1.scatter(umap_feature[:,0],umap_feature[:,1],umap_feature[:,2],c=np.arange(602),s=5,cmap='inferno')
fig.colorbar(im1, ax=ax1,fraction=0.015,pad=0.08)
#plt.savefig('toy1_umap',dpi=600)

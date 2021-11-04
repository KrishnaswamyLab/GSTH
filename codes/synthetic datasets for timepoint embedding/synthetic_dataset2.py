import umap
import numpy as np
import h5py
import scipy.io as sio
from scipy.stats import pearsonr
import networkx as nx
from collections import Counter

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





def generate_mol_feature(adj,ro):
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




sin_signal1 = [np.sin(1/20*np.pi*x) for x in range(11)]
sin_signal2 = [np.sin(1/10*np.pi*x) for x in range(11)]
sin_signal3 = [np.sin(1/5*np.pi*x) for x in range(11)]

sin_signal = [sin_signal1,sin_signal2,sin_signal3]
tp = np.arange(50,400,11)
np.random.shuffle(tp)
cp = np.arange(1867)
np.random.shuffle(cp)
selected_cell = cp[:100]
simulate_signal = simulate_signal.transpose()


simulate_cell_signal = []
for i in range(100):
    tp = np.arange(50,600,11)
    np.random.shuffle(tp)
    selected_tp = tp[:12]
    signal_cell = np.zeros(600)
    for j in selected_tp:
        signal_cell[j:j+11] = sin_signal[np.random.randint(3)]
    simulate_cell_signal.append(signal_cell)



simulate_signal_com_new = simulate_signal[:,:50]



new_signal_matrix =np.zeros((1867,600))
for i,v in enumerate(selected_cell):
    new_signal_matrix[v] = simulate_cell_signal[i]



new_signal_matrix[:,:50] = simulate_signal_com_new



phate_operator5 = phate.PHATE(n_components=3,knn=20)
phate_data5 = phate_operator5.fit_transform(new_signal_matrix.transpose()) 

fig = plt.figure(figsize=(16,8))
ax1 = plt.subplot2grid(shape=(1,1), loc=(0,0),projection='3d')
im1 = ax1.scatter(phate_data5[:,0],phate_data5[:,1],phate_data5[:,2],c=np.arange(600),s=5,cmap='inferno')
fig.colorbar(im1, ax=ax1,fraction=0.015,pad=0.08)


feature_z_disrupt = scipy.stats.mstats.zscore(new_signal_matrix,0)
feature_z_disrupt[np.isnan(feature_z_disrupt)] = 0
scattering_feature_disrupt = generate_mol_feature(adj,feature_z_disrupt)
scattering_feature_disrupt = scattering_feature_disrupt.transpose()
phate_operator6 = phate.PHATE(n_components=3,knn=20)
phate_data6 = phate_operator6.fit_transform(scattering_feature_disrupt) 
fig = plt.figure(figsize=(16,8))
ax1 = plt.subplot2grid(shape=(1,1), loc=(0,0),projection='3d')
im1 = ax1.scatter(phate_data6[:,0],phate_data6[:,1],phate_data6[:,2],c=np.arange(600),s=5,cmap='inferno')
fig.colorbar(im1, ax=ax1,fraction=0.015,pad=0.08)




feature_disrupt_tsne= TSNE(n_components=3).fit_transform(scattering_feature_disrupt)
fig = plt.figure(figsize=(16,8))
ax1 = plt.subplot2grid(shape=(1,1), loc=(0,0),projection='3d')
im1 = ax1.scatter(feature_disrupt_tsne[:,0],feature_disrupt_tsne[:,1],feature_disrupt_tsne[:,2],c=np.arange(600),s=5,cmap='inferno')
fig.colorbar(im1, ax=ax1,fraction=0.015,pad=0.08)



pca_feature_disrupt = PCA(n_components=3).fit_transform(scattering_feature_disrupt)
fig = plt.figure(figsize=(16,8))
ax1 = plt.subplot2grid(shape=(1,1), loc=(0,0),projection='3d')
im1 = ax1.scatter(pca_feature_disrupt[:,0],pca_feature_disrupt[:,1],pca_feature_disrupt[:,2],c=np.arange(600),s=5,cmap='inferno')
fig.colorbar(im1, ax=ax1,fraction=0.015,pad=0.08)
plt.savefig('toy3_pca1002',dpi=600)



umap_operator= umap.UMAP(n_components=3)
umap_feature = umap_operator.fit_transform(scattering_feature_disrupt)

fig = plt.figure(figsize=(16,8))
ax1 = plt.subplot2grid(shape=(1,1), loc=(0,0),projection='3d')
im1 = ax1.scatter(umap_feature[:,0],umap_feature[:,1],umap_feature[:,2],c=np.arange(600),s=5,cmap='inferno')
fig.colorbar(im1, ax=ax1,fraction=0.015,pad=0.08)
plt.savefig('toy3_umap1002',dpi=600)



from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram



VR = VietorisRipsPersistence(homology_dimensions=[0, 1, 2])
umap_feature_ = umap_feature.reshape(1,-1,3)
diagrams = VR.fit_transform(umap_feature_)
plot_diagram(diagrams[0])

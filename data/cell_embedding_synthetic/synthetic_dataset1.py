import numpy as np
import h5py
import scipy.io as sio
from scipy.stats import pearsonr
import networkx as nx
from collections import Counter
from numpy import linalg as LA
from sklearn.decomposition import PCA
import phate
import matplotlib.pyplot as plt

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
for i in range(20):
    temp = np.matmul(M,temp)
    simulate_signal.append(temp)

simulate_signal_ = np.array(simulate_signal).transpose()
distance_to_center = []

for i in range(1867):
    try: 
        d = nx.shortest_path_length(G,source=i,target=256)
    except:
        d = 45
    distance_to_center.append(d)

wave_pool = []

for i in range(1867):
    if sum(simulate_signal_[i])!=0:
        wave_pool.append(i)

single_pool = []
single_num = 1867 - len(wave_pool)
for i in range(1867):
    #n = np.random.randint(1867)
    #while sum(simulate_signal_[n])!=0:
    #    n = np.random.randint(1867)
    if i not in wave_pool:
        n = i
        single_pool.append(n)
        sin_signal1 = [np.sin(1/20*np.pi*x) for x in range(5)]
        p = np.random.randint(14)
        simulate_signal_[n][p:p+5] = np.array(sin_signal1)



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
    F0 = np.matmul(LA.matrix_power(A,16),ro)    
    return F0



def first_order_feature(A,u):
    F1 = np.matmul(LA.matrix_power(A,16),np.abs(u))
    #F = []
    #F.append(np.sum(np.matmul(P,u[0]),1))
    #for i in range(1,t):
    F1 = np.concatenate(F1,1)
    return F1



def selected_second_order_feature(A,W,u):
    u1 = np.einsum('ij,ajt ->ait',W[1],u[0:1])
    for i in range(2,len(W)):
        u1 = np.concatenate((u1,np.einsum('ij,ajt ->ait',W[i],u[0:i])),0)
    u1 = np.abs(u1)
    F2 = np.matmul(LA.matrix_power(A,16),u1)
    #F2.append(np.sum(np.einsum('ij,ajt ->ait',W[i],u[0:i]),1).reshape(len(u[0:i])*F,1))
    #F2 = np.sum(np.einsum('ijk,akt ->iajt',P,F1),2).reshape(len(P)*len(F1)*F,1)
    F2 = np.concatenate(F2,1)
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
    F = np.concatenate((F0,F1),axis=1)
    F = np.concatenate((F,F2),axis=1)
    #F = np.concatenate((F1,F2),axis=0)
    #F = np.concatenate((F,F3),axis=0)
    return F



feature_z = scipy.stats.mstats.zscore(simulate_signal_,0)
scattering_feature = generate_mol_feature(adj,feature_z)
phate_operator = phate.PHATE(n_components=3,knn=20)
phate_data = phate_operator.fit_transform(scattering_feature) 
fig = plt.figure(figsize=(16,8))
ax1 = plt.subplot2grid(shape=(1,1), loc=(0,0),projection='3d')
ax1.view_init(30, 110)
im1 = ax1.scatter(phate_data[:,0],phate_data[:,1],phate_data[:,2],c=distance_to_center,s=5)
fig.colorbar(im1, ax=ax1,fraction=0.015,pad=0.08)
#plt.savefig('cell_toy1',dpi=600)

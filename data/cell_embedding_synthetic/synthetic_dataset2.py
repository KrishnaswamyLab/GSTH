import numpy as np
import h5py
import scipy.io as sio
import networkx as nx

from numpy import linalg as LA
import scipy

import phate
import matplotlib.pyplot as plt

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
toy_feature[128] = 1
G = nx.from_numpy_matrix(adj)
L = nx.normalized_laplacian_matrix(G)
index_nan = np.argwhere(np.isnan(L.toarray()))

M = L.toarray()
for v in index_nan:
    M[v[0]][v[1]] = 0

simulate_signal = []
simulate_signal.append(toy_feature)
temp = toy_feature
flag = 1
for i in range(20):
    temp = np.matmul(M,temp)
    simulate_signal.append(temp)

simulate_signal_ = np.array(simulate_signal).transpose()

label = []
for i in range(1867):
    if sum(simulate_signal_[i])!=0:
        label.append(i)


toy_feature = np.zeros(1867)
toy_feature[128] = 1
toy_feature[1356] = 1
G = nx.from_numpy_matrix(adj)
L = nx.normalized_laplacian_matrix(G)
index_nan = np.argwhere(np.isnan(L.toarray()))

M = L.toarray()
for v in index_nan:
    M[v[0]][v[1]] = 0

simulate_signal = []
simulate_signal.append(toy_feature)
temp = toy_feature
flag = 1
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

d1 = []
d2 = []

for i in range(1867):
    if i in label:
        d1.append(distance_to_center[i])
    else:
        d2.append(distance_to_center[i])



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


phate_operator1 = phate.PHATE(n_components=3,knn=20)
phate_data1 = phate_operator1.fit_transform(simulate_signal_)

pd1_1 = []
pd2_1 = []

for i in range(1867):
    if i in label:
        pd1_1.append(phate_data1[i])
    else:
        pd2_1.append(phate_data1[i])
pd1_1 = np.array(pd1_1)
pd2_1 = np.array(pd2_1)

pd1 = []
pd2 = []

for i in range(1867):
    if i in label:
        pd1.append(phate_data[i])
    else:
        pd2.append(phate_data[i])
pd1 = np.array(pd1)
pd2 = np.array(pd2)

fig = plt.figure(figsize=(16,8))
ax1 = plt.subplot2grid(shape=(1,1), loc=(0,0),projection='3d')
ax1.view_init(30, 110)
im1 = ax1.scatter(pd1_1[:,0],pd1_1[:,1],pd1_1[:,2],c=d1,s=5,cmap='OrRd')
im2 = ax1.scatter(pd2_1[:,0],pd2_1[:,1],pd2_1[:,2],c=d2,s=5,cmap='YlGn')
fig.colorbar(im1, ax=ax1,fraction=0.015,pad=0.08)
fig.colorbar(im2, ax=ax1,fraction=0.015,pad=0.08)
#plt.savefig('cell_phate_toy2',dpi=600)

fig = plt.figure(figsize=(16,8))
ax1 = plt.subplot2grid(shape=(1,1), loc=(0,0),projection='3d')
ax1.view_init(30, 110)
im1 = ax1.scatter(pd1[:,0],pd1[:,1],pd1[:,2],c=d1,s=5,cmap='OrRd')
im2 = ax1.scatter(pd2[:,0],pd2[:,1],pd2[:,2],c=d2,s=5,cmap='YlGn')
fig.colorbar(im1, ax=ax1,fraction=0.015,pad=0.08)
fig.colorbar(im2, ax=ax1,fraction=0.015,pad=0.08)

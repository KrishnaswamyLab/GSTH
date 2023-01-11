import numpy as np
from numpy import linalg as LA

def lazy_random_walk(adj):

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

    F0 = np.matmul(LA.matrix_power(A,16),ro)

    return F0

def first_order_feature(A,u,ax):

    F1 = np.matmul(LA.matrix_power(A,16),np.abs(u))
    F1 = np.concatenate(F1,ax)

    return F1

def second_order_feature(A,W,u,ax):

    u1 = np.einsum('ij,ajt ->ait',W[1],u[0:1])
    for i in range(2,len(W)):
        u1 = np.concatenate((u1,np.einsum('ij,ajt ->ait',W[i],u[0:i])),0)
    u1 = np.abs(u1)
    F2 = np.matmul(LA.matrix_power(A,16),u1)
    F2 = np.concatenate(F2,ax)
    
    return F2

def generate_timepoint_feature(adj,ro):

    P = lazy_random_walk(adj)
    
    W = graph_wavelet(P)
    u = np.abs(np.matmul(W,ro))
    
    F0 = zero_order_feature(P,ro)
    F1 = first_order_feature(P,u,0)
    F2 = second_order_feature(P,W,u,0)
    F = np.concatenate((F0,F1),axis=0)
    F = np.concatenate((F,F2),axis=0)

    return F

def generate_celluar_feature(adj,ro):

    P = lazy_random_walk(adj)
    W = graph_wavelet(P)
    u = np.abs(np.matmul(W,ro))
    
    F0 = zero_order_feature(P,ro)
    F1 = first_order_feature(P,u,1)
    F2 = second_order_feature(P,W,u,1)
    F = np.concatenate((F0,F1),axis=1)
    F = np.concatenate((F,F2),axis=1)
    
    return F
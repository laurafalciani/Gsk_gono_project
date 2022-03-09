#!/usr/bin/env python
# coding: utf-8

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
import os
import pandas as pd
import torch
from scipy.sparse import csr_matrix,coo_matrix,save_npz,load_npz, dok_matrix, lil_matrix
import pickle as pkl
from typing import Union
import numpy as np

def from_scipy_sparse_matrix(A):
    r"""Converts a scipy sparse matrix to edge indices and edge attributes.

    Args:
        A (scipy.sparse): A sparse matrix.
    """

    #print(A)
    div = 1
    if(A.max()!=0):
        div = A.max()
    norm_A = A/div 
    row = torch.from_numpy(A.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(A.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([col, row], dim=0)
    
    #edge_weight = torch.from_numpy(A.data)
    edge_weight = torch.from_numpy(A.data)
    return edge_index, edge_weight

def compareMatrix(A,B):
    A = A.todense()
    
    B = B.todense()
    E = np.zeros(A.shape)
    rows = A.shape[0]
    cols = A.shape[1]
    for i in range(0, rows):
        for j in range(0, cols):            
            #print(A[i,j])
            if (A[i,j]) != 0:
                E[i,j] = 1-B[i,j]
                if(E[i,j] == 0 and (A[i,j]) == 1):
                    print("B", B[i,j])
                    print("A",A[i,j])#print(E[i,j])
    return E

def load_data(g_name):
    
    y = pd.read_csv('/hpc/scratch/hdd1/lf481323/label_bin/lin.{}'.format(g_name), index_col = 0)
    y = y['lineage']
    A = load_npz("/hpc/scratch/hdd1/lf481323/distMatCOO_bin/{}.npz".format(g_name))
    B = load_npz("/hpc/scratch/hdd1/lf481323/EdgeFeatures/{}.npz".format(g_name))
    print(g_name)
    c = lil_matrix(A)
    c.setdiag(0)
    A = c.tocoo()
    
   
    E = compareMatrix(A,B)
    E = lil_matrix(E)
    # build graph
    edge_index,a =  from_scipy_sparse_matrix(A)
    #e, edge_attr =  from_scipy_sparse_matrix(E)
    
    #E = scaling(E)
    E = E.tocoo()
   
    edge_attr = torch.from_numpy(E.data)   
    
    #values = E.data
    #indices = np.vstack((E.row, E.col))

    #i = torch.LongTensor(indices)
    #v = torch.FloatTensor(values)
    #shape = E.shape

    #features = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
    features = torch.eye(len(y), dtype=torch.float)
    
    y = torch.LongTensor(y.values)
    
    
    
    return features, edge_index, edge_attr, y



mymat = os.listdir('/hpc/scratch/hdd1/lf481323/distMat_1/')




def dataset(mymat):
    dataset = []
    num_features = 1

    for file  in mymat:
        file_path = f"/hpc/scratch/hdd1/lf481323/distMatCOO_balanced/{file}"
        g_name = file.replace('.npz','')

        #if(g_name in sil200.index):
           # print(g_name)
        x, edge_index, edge_attr, y = load_data(g_name)
        dataset.append(Data(x=x, edge_index=edge_index, edge_attr = edge_attr, y=y))

    return dataset

msk=AddTrainValTestMask(split="train_rest", num_splits = 1, num_val = 0.3, num_test= 0.6)

for data in dataset:
    msk(data)
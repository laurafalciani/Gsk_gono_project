#!/usr/bin/env python
# coding: utf-8
#SBATCH --job-name=example
#SBATCH --ntasks=10
#SBATCH --time=5-00:05:00
#SBATCH --cpus-per-task=20


import pip as th
#from Bio.Phylo.TreeConstruction import DistanceCalculator
#from Bio import AlignIO
import os
import numpy as np
import scipy as sc
from scipy.sparse import csr_matrix,coo_matrix,save_npz,load_npz, dok_matrix
from scipy.linalg import block_diag
import pickle as pkl
import pandas as pd
import multiprocessing


lin = pd.read_csv('~/Lineage_differences_PresenceAbsence/FeatureSelection/lineages.csv', sep = '\t')
lin['ID'] = 'g' + lin['ID'].astype(str)

#one hot labels outuput
def add_label_toMatrix(df,lineages):
    genes = df.index
    names = [x.split('_')[0] for x in genes]
    df = df.set_index([names])
    lineages = lin.set_index('ID')
    y = pd.merge(df,lineages,left_index=True,
    right_index=True)
    y = y.set_index([genes])
    return y
#binarizzazione matrice
def oneMatrix(x, q):
    if x <= q:
        x = 1
    else:
        x = 0
    return x
def distMatrix(x, q):
    if x <= q:
        x = x
    else:
        x = 0
    return x
def norm(df):
    div = 1
    if(df.max(axis = 0)!=0):
        div = df.max(axis = 0)
    norm_df = df.div()
    return norm_df

mydfmat = os.listdir('/hpc/scratch/hdd1/lf481323/dataframeMatrix/')
stat = pd.read_csv('/home/lf481323/matrixDistance/gcn/dm_statistics.csv')
def binaryMatrix(file):
    file_path = f"/hpc/scratch/hdd1/lf481323/dataframeMatrix/{file}"
    g_name = file.replace('df.','')
    q = float(stat[stat['Gene'] == g_name]['q3'])
    df = pd.read_csv(file_path, index_col = 0)
    

    if(len(df) > 6):
       #df_bin = df.applymap(lambda x: oneMatrix(x, q))
        df_bin = df.applymap(lambda x: oneMatrix(x, q))
        df_dist = df
        gene = df.index.tolist()
        gene = [x.split('_')[0] for x in gene]
        B = set(lin['ID'].tolist())
        x = list(set(B).difference(gene))
        #print(x)
        if x:
            ukn = [s + '_x'+ g_name for s in x]
            index = df.columns.tolist() + ukn
            df2 = pd.DataFrame([[0]*len(index)], columns= index, index = ukn)    
            df2[ukn] = 1
            df_bin = pd.concat([df_bin, df2[ukn]], axis=1).replace(np.nan, 1)
            df2d = pd.DataFrame([[1]*len(index)], columns= index, index = ukn)   
            df2d[ukn] = 0
            df_dist = pd.concat([df_dist, df2d[ukn]], axis=1).replace(np.nan, 0)
            #df_dist.to_csv('/hpc/scratch/hdd1/lf481323/distMatCOO/df.{}'.format(g_name))
            y =  add_label_toMatrix(df_bin,lin)
        else:

            y =  add_label_toMatrix(df,lin)
        mask1 = y[y['lineage']==1].sample(n=121)
        mask0 = y[y['lineage']==0].sample(n=121)
        ybal = pd.concat([mask0,mask1])
        ybal = ybal.sort_index(axis = 0)

        df_bin = df_bin[ybal.index]
        df_bin = df_bin.loc[ybal.index]
        df_dist = df_dist[ybal.index]
        df_dist = df_dist.loc[ybal.index]
        
        A = csr_matrix(df_bin.astype(pd.SparseDtype("float", np.nan)))
        E = csr_matrix(df_dist.astype(pd.SparseDtype("float", np.nan)))
        
        ybal.to_csv('/hpc/scratch/hdd1/lf481323/label_bin/lin.{}'.format(g_name))
        df_bin.to_csv('/hpc/scratch/hdd1/lf481323/dataframeMatrix_bin/df.{}'.format(g_name))
        save_npz("/hpc/scratch/hdd1/lf481323/EdgeFeatures/{}.npz".format(g_name), E)
        save_npz("/hpc/scratch/hdd1/lf481323/distMatCOO_bin/{}.npz".format(g_name), A)
    return 0


num_proc = 10                           # specify number to use (to be nice)
p = multiprocessing.Pool(num_proc)
result = p.map(binaryMatrix, mydfmat)
p.close()
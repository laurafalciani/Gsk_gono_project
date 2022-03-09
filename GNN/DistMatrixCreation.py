#!/usr/bin/env python
# coding: utf-8
#SBATCH --job-name=example
#SBATCH --ntasks=10
#SBATCH --time=5-00:05:00
#SBATCH --cpus-per-task=20



#Script per creare matrici di distanze dai file di allineamento e statistiche
#import dgl
from Bio.Phylo.TreeConstruction import DistanceCalculator
from Bio import AlignIO
import os
import numpy as np
import scipy as sc
from scipy.sparse import csr_matrix,coo_matrix,save_npz,load_npz
from scipy.linalg import block_diag
#from dgl.data.utils import save_graphs
#from dgl.data.utils import load_graphs
import pickle as pkl
import pandas as pd
#get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# In[ ]:





# In[62]:


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

# In[68]:


mylist = os.listdir('pan_genome_sequences')


# In[69]:


out = '/hpc/scratch/hdd1/lf481323/gcnData'


import multiprocessing


# In[86]:


def distanceCalculator(file):
    
    calculator = DistanceCalculator('identity')
    #statistiche
    mean = []
    std = []
    quant1 = []
    quant2 = []
    quant3 = []
    quant4 = []
    count = []
       
    g_name = file.replace('.fa.aln','')

    #creo matrice distanza
    file_path = f"pan_genome_sequences/{file}"
    aln = AlignIO.read(open(file_path), "fasta")
    dm = calculator.get_distance(aln)
    #gene.add(dm.names) 
    if(len(dm) > 1):
        dms = coo_matrix(dm)
        mean.append(np.mean(dm))
        std.append(np.std(dm))
        quant1.append(np.percentile(dm, 25))
        quant3.append(np.percentile(dm, 75))
        quant2.append(np.percentile(dm, 50))
        quant4.append(np.percentile(dm, 100))
        count.append(len(dm))
        #salvo nomi dei geni per ogni cluster
        with open("/hpc/scratch/hdd1/lf481323/geneListperMat/{}.gnames".format(g_name), "wb") as fp:
               pkl.dump(dm.names, fp)
        #matrice distanza in formato COO 
        
        save_npz("/hpc/scratch/hdd1/lf481323/distMatCOO/{}.npz".format(g_name), dms)
        #matrice distanza come dataframe
        df = pd.DataFrame.sparse.from_spmatrix(dms, index = dm.names, columns = dm.names)
        df.to_csv('/hpc/scratch/hdd1/lf481323/dataframeMatrix/df.{}'.format(g_name))
        #label
        y =  add_label_toMatrix(df,lin)  
        df.to_csv('/hpc/scratch/hdd1/lf481323/dataframeMatrix/df.{}'.format(g_name))
        y[['lineage']].to_csv('/hpc/scratch/hdd1/lf481323/label/{}_label'.format(g_name))
       
    return mean,std,quant1,quant2,quant3,quant4, g_name, count





num_proc = 20                           # specify number to use (to be nice)
p = multiprocessing.Pool(num_proc)
result = p.map(distanceCalculator, mylist)
p.close()


# In[96]:


result = np.array(result)


# In[116]:


d = dict({'Gene':result[:,6].reshape(len(result)),'Count':result[:,7].reshape(len(result)), 'Mean' : result[:,0].reshape(len(result)) , 'Std' : result[:,1].reshape(len(result)), 'q1':result[:,2].reshape(len(result)), 'q2':result[:,3].reshape(len(result)), 'q3': result[:,4].reshape(len(result)),'q4': result[:,5].reshape(len(result))})
df = pd.DataFrame.from_dict(d)
df.to_csv('dm_statistics')


# In[ ]:





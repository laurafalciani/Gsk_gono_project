import numpy as np
import scipy as sc
from scipy.sparse import csr_matrix,coo_matrix,save_npz,load_npz, dok_matrix
from scipy.linalg import block_diag
import pickle as pkl
import pandas as pd
import os

mydfmat = os.listdir('/hpc/scratch/hdd1/lf481323/dataframeMatrix/')

lin = pd.read_csv('~/Lineage_differences_PresenceAbsence/FeatureSelection/lineages.csv', sep = '\t')
lin['ID'] = 'g' + lin['ID'].astype(str)


#one hot labels outuput
def add_label_toMatrix(df,lineages):
    names = [x.split('_')[0] for x in df.index]
    df = df.set_index([names])
    lineages = lin.set_index('ID')
    y = pd.merge(df,lineages,left_index=True,
    right_index=True)
    #y = y.set_index(df.index)
    return  y.set_index(df.index)


cols = [1,2,3,4,5,6,7,8,9,10,11,12,13]
df_A = pd.read_csv('~/roary_output_linA/gene_presence_absence_linA.csv',sep=',',error_bad_lines=False)

df_A.drop(df_A.columns[cols], 1, inplace=True)
df_B = pd.read_csv('~/roary_output_linB/gene_presence_absence.csv',sep=';')

df_B.drop(df_B.columns[cols], 1, inplace=True)



df_A = df_A.set_index('Gene')
df_B = df_B.set_index('Gene')

#salvo geni diversificati per lineage
lin_A = lin['ID'].loc[(lin['lineage'] ==1)]
lin_A = lin_A.to_list()
lin_B = lin['ID'].loc[(lin['lineage'] ==0)]
lin_B = lin_B.to_list()





#silhouette_score
from sklearn.metrics import silhouette_score
s = dict()
for file in mydfmat:
    file_path = f"/hpc/scratch/hdd1/lf481323/dataframeMatrix/{file}"
    g_name = file.replace('df.','')
    df = pd.read_csv(file_path, index_col = 0)
    #differenzio COGs per lineage
    df_A = df.loc[:,df.rename(columns=lambda x: x.split('_')[0]).columns.isin(lin_A)]
    df_B = df.loc[:,df.rename(columns=lambda x: x.split('_')[0]).columns.isin(lin_B)]
    yA = add_label_toMatrix(df_A,lin)
    yB = add_label_toMatrix(df_B,lin)
    #Conto proporzione lineage per COGs
    count_B[g_name] = len(np.array(yB[(yB['lineage'] == 0)]))
    count_A[g_name] = len(np.array(yA[(yA['lineage'] ==1)]))
    #aggingo colonna lineage a matrice COGs
    y = add_label_toMatrix(df,lin)
    if(len(df) > 2):
        if (count_A[g_name] <=1 or count_B[g_name] <= 1):
            s[g_name] = 0
        else:
            s[g_name] = silhouette_score(np.array(df, dtype= float),np.array(y['lineage']), metric="precomputed")
            
            
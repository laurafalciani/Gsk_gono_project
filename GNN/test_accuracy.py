import numpy as np
import pickle as pkl
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
import os
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from scipy.sparse import csr_matrix,coo_matrix,save_npz,load_npz, dok_matrix, lil_matrix
import sys
import mydata
from utils import *
import pandas as pd


mymat = os.listdir('/hpc/scratch/hdd1/lf481323/distMatCOO_bin/')

dataset = []

for file  in mymat:
    file_path = f"/hpc/scratch/hdd1/lf481323/distMatCOO_balanced/{file}"
    g_name = file.replace('.npz','')
    print(g_name)
    x, edge_index, edge_attr, y = mydata.load_data(g_name)
    dataset.append(Data(x=x, edge_index=edge_index, edge_attr = edge_attr, y=y))
    
    
#load
import shutil
import model
from itertools import chain
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer, checkpoint['epoch'], checkpoint['loss']


model = model.GCN(120)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
ckp_path ="checkpoint_lr001_q3balanced_distances.pth"
model, optimizer, start_epoch, loss = load_ckp(ckp_path, model, optimizer)





#Test
y_pred_list = []
y_true_list = []
model.eval()
report = dict()
with torch.no_grad():
    for file in mymat:
        file_path = f"/hpc/scratch/hdd1/lf481323/distMatCOO_balanced/{file}"
        g_name = file.replace('.npz','')
       
        x, edge_index, edge_attr, y = mydata.load_data(g_name)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        print(data)
        pred = model(data)
        #y_test_pred = pred[data]
        label = data.y.flatten()
        #label = label[data]
        #print('true', label)
        y_test_pred = torch.sigmoid(pred)
        y_pred_tag = torch.round(y_test_pred)    
        
        y_true_list.append(np.array(label))    
        y_pred_list.append(y_pred_tag.numpy())
       
        print(confusion_matrix(np.array(label), (y_pred_tag.numpy())))
        
        report[g_name] = classification_report(np.array(label), (y_pred_tag.numpy()),output_dict=True)
        df_test = pd.DataFrame.from_dict(report).T

df_test.sort_values(by = ['accuracy'], ascending = False)

gcn_gene = df_test[df_test['accuracy'] > 0.78]

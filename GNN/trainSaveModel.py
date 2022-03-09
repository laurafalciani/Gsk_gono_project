#!/usr/bin/env python
# coding: utf-8
#SBATCH --mem-per-cpu=20G
#SBATCH --ntasks=5
#SBATCH --time=5-00:05:00
#SBATCH --cpus-per-task=10
#export PYTHONPATH="${PYTHONPATH}:/home/lf481323/matrixDistance/gcn/gcn_train/utils"
from __future__ import division
from __future__ import print_function
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv,GraphConv
from torch_geometric.loader import DataLoader
import pickle as pkl 
import matplotlib.pyplot as plt
import numpy as np

with open("dataset_num_test0.3val0.1_bin.txt", "rb") as fp:   # Unpickling
    dataset = pkl.load(fp)



output_dim = 1
num_node_features = 1
hidden_dim= 64

class GCN(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv1 = GraphConv(num_node_features, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        
        x = self.conv1(x, edge_index,edge_weight.float())
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index,edge_weight.float())
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        x = self.conv3(x.float(), edge_index, edge_weight.float())
        
        return x
    
    def loss(self, pred, label):
        return F.nll_loss(pred, label)
    def CE_loss(self, pred, label,class_weights):
        #loss = torch.nn.CrossEntropyLoss(class_weights)
        loss = torch.nn.BCEWithLogitsLoss(pos_weight = class_weights)
        return loss(pred.float().flatten(),label.float().flatten())   
    
class_weights= [1/4]
class_weights= torch.tensor(class_weights,dtype=torch.double)

def train(dataset,num_epochs):
    
    test_loader = loader = DataLoader(dataset, batch_size=100, shuffle=True)
    model = GCN(hidden_dim)
    loss_values = []   
    
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    #train
    for epoch in range(1,num_epochs + 1):
        total_loss = 0
        epoch_acc = 0
        model.train()
        for batch in loader:
            #print(batch.train_mask, '----')
            opt.zero_grad()
            pred = model(batch)
            label = batch.y.flatten()
            pred = pred[batch.train_mask]
            label = label[batch.train_mask]
            loss = model.CE_loss(pred, label, class_weights)
            loss.backward()
            opt.step()
            acc = binary_acc(pred, label.unsqueeze(1))
            total_loss += loss.item() * batch.num_graphs
            acc = binary_acc(pred, label.unsqueeze(1))
            epoch_acc += acc.item() * batch.num_graphs
            
        
        epoch_acc /= len(loader.dataset)
        total_loss /= len(loader.dataset)
        loss_values.append(total_loss)
        print("loss", total_loss, epoch)
        print('Acc', epoch_acc )
        if epoch % 10 == 0:
            val_acc = validation(test_loader, model)
            print("Epoch {}. Loss: {:.4f}. Val accuracy: {:.4f}".format(
                epoch, total_loss, val_acc))
            print("Validation accuracy", val_acc, epoch)
    plt.plot(range(1,num_epochs + 1), np.array(loss_values), 'g', label='Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('Training_loss.png')
    return model, opt, loss
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

def validation(loader, model):
    model.eval()

    correct = 0
    total = 0
    for data in loader:
        with torch.no_grad():
            pred = model(data)
            pred = torch.round(torch.sigmoid(pred))
            label = data.y


        mask = data.val_mask
        
        pred = pred[mask]
        
        
        y_true = label[mask].unsqueeze(1)
        
        correct += torch.sum(pred == y_true).item()
        #correct += pred.eq(label).sum().item()
        total += y_true.shape[0]
        #print("corret: ",correct)
       
        #total += torch.sum(data.val_mask).item()
        
       # for data in loader:
        #    total += torch.sum(data.val_mask).item()
        #print("total",total)
        
    return correct / total

def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            pred = model(data)
            #pred = pred.argmax(dim=1)
            pred = torch.round(torch.sigmoid(pred))
            label = data.y


        mask = data.val_mask if is_validation else data.test_mask
        
        pred = pred[mask]
        label = data.y[mask].flatten()
            
        correct += pred.eq(label).sum().item()
    
    
        total = 0
        for data in loader.dataset:
            total += torch.sum(data.test_mask).item()
    return correct / total


model, opt, loss = train(dataset,100)

FILE = "checkpoint.pth"
torch.save({
            'epoch': 100,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss,
            }, FILE)

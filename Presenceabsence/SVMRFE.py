#!/usr/bin/env python
# coding: utf-8

# In[50]:


from sklearn import svm
import pandas as pd
import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv('/home/lf481323/Roary_output_all_last_touse/gene_presence_absence.csv', index_col = 0, sep = ',')
df = df.rename(columns=lambda x: x.split('_')[0])
#df.to_csv("/home/lf481323/Roary_output_all_last_touse/gene_presence_absence.csv")


# In[5]:


df.head()


# In[6]:


#tolgo geni in meno di 4 genomi e del core
df = df[df['No. isolates'] > 4]
df = df[df['No. isolates'] <= 415]
df.rename(columns=lambda x: x.split('_')[0])


# In[7]:


#tolgo colonne informative, mantengo solo geni e genomi
cols = [0,1,2,3,4,5,6,7,8,9,10,11,12]
df.columns[cols]
df.drop(df.columns[cols], 1, inplace=True)
gene = df.fillna(0)


# In[9]:


#creo matrice one hot encoding
gene = gene.astype(bool).astype(int)
gene = gene.rename(columns=lambda x: x.split('_')[0])
gene.head()


# In[10]:


#traspongo: genomi in riga, geni in colonna
geneT = gene.T.astype(int)
geneT


# In[14]:


#carico dataset che associa genoma a lineage
lineage = pd.read_csv('~/Lineage_differences_PresenceAbsence/FeatureSelection/lineages.csv',sep='\t', index_col = 0)
lineage = lineage.set_index(geneT.index)
lineage.sort_index(axis= 0, ascending = True)


# In[15]:


data = geneT.join(lineage)

#df2 = df2.join(lineage)
data


# In[17]:


#SVM linear kernel con tutte le variabili
linear_svc = svm.SVC(kernel='linear')
Xf = data.iloc[:,:-1].to_numpy()
yf= data['lineage'].to_numpy()
scores = cross_val_score(linear_svc, Xf, yf, cv=5)


# In[19]:


print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[20]:


def f_importances(coef, names,top=-1):
    imp = coef
    imp,names = zip(*sorted(list(zip(imp,names))))
     # Show all features
    if top == -1:
        top = len(names)
    plt.barh(range(top), imp[::-1][0:top], align='center')
    plt.yticks(range(top),  names[::-1][0:top])
    plt.show()
    return names[::-1][0:top]
features_names = geneT.columns


# # SVM Recursive Feature Elimination

# In[21]:


#Preparo il training set
X = data.iloc[:,:-1].to_numpy()
y = data['lineage'].to_numpy()


# In[23]:


#ALL
linear_svc = svm.SVC(kernel='linear')
linear_svc.fit(X, y)
sorted_feat_All = f_importances(abs(linear_svc.coef_[0]), features_names, top=20)


# In[24]:


#algortimo recursive feature elimination basato sui pesi del classificatore svm
def svm_feature_selection_linear(X_0, y_0, f_names):  
    r = []
    
    s = [*range(0,len(X_0[0])-1,1)]
    X_train = X_0[:,s]
    y_train = y_0
    features = np.array(f_names)
    i = 0
    while (len(s) > 0):
        svc_linear = svm.SVC(kernel='linear')
        svc_linear.fit(X_train, y_train)
        f_importance = abs(svc_linear.coef_[0])
        sorted_idx = f_importance.argsort()
    
        
        r.append(features[sorted_idx[0]])
        X_train = np.delete(X_train,sorted_idx[0],axis = 1)
        features = np.delete(features,sorted_idx[0])       
        s = [*range(0,len(X_train[0])-1,1)]
       
    return r


# In[37]:


ranked_f= svm_feature_selection_linear(X, y, geneT.columns)


# In[35]:


geneRanked_arr = np.array(ranked_f)[::-1]
#pd.DataFrame(geneRanked_arr).to_csv('sorted_feat_SVM.csv')


# In[44]:



geneRanked = pd.read_csv('sorted_feat_SVM.csv')
geneRanked_arr = np.array(geneRanked['COG'])


# In[45]:


sortedRecursive = geneRanked_arr[:180]


# In[46]:


import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
svc = SVC(kernel="linear")
n = [30,60,90,120,150,180,240,270,300,330,360,390,420,450,480,510,540]
grid_scores =np.zeros(len(n))
scores.mean()
for i in range(len(n)):
# Create the RFE object and compute a cross-validated score.
    data_f = data[ranked_f[-n[i]:]].join(lineage)
# The "accuracy" scoring shows the proportion of correct classifications
    Xf = data_f.iloc[:,:-1].to_numpy()
    yf= data_f['lineage'].to_numpy()
    scores = cross_val_score(linear_svc, Xf, yf, cv=10).mean()
    grid_scores[i] = scores.mean()
    
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (accuracy)")
plt.plot(n,grid_scores)
plt.show()


# In[47]:


print("Optimal number of features : %d" % n[grid_scores.argmax()])


# In[48]:


#filtro il dataset sulle features selezionate
n = 180
data_f = data[ranked_f[-n:]].join(lineage)


# In[51]:


trainf, testf = train_test_split(data_f, test_size = 0.4)
X_trainf = trainf.iloc[:,:n-1].to_numpy()
X_testf = testf.iloc[:,:n-1].to_numpy()
y_trainf = trainf['lineage'].to_numpy()
y_testf = testf['lineage'].to_numpy()


# In[52]:


#train con le feature selezionate
linear_svc = svm.SVC(kernel='linear')
linear_svc.fit(X_trainf, y_trainf)


# In[53]:


y_predf = linear_svc.predict(X_testf)
metrics.confusion_matrix(y_testf, y_predf)


# In[54]:


metrics.accuracy_score(y_testf, y_predf)


# In[55]:


metrics.precision_score(y_testf, y_predf,average='weighted')


# In[56]:


from sklearn.metrics import f1_score
f1_score(y_testf, y_predf, average='macro')


# In[57]:


#Cross Validation
from sklearn.model_selection import cross_val_score
Xf = data_f.iloc[:,:-1].to_numpy()
yf= data_f['lineage'].to_numpy()
scores = cross_val_score(linear_svc, Xf, yf, cv=5)


# In[58]:


print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[ ]:





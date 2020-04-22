#!/usr/bin/env python
# coding: utf-8

# In[12]:


import glob
import pandas as pd
import os
import sklearn.datasets as skdata
import numpy as np
#import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.metrics import confusion_matrix
#%matplotlib inline


# In[30]:


from sklearn.metrics import f1_score


# In[13]:


files=glob.glob("./Stocks/*.txt")
print(np.shape(files))
print(files[:20])


# In[81]:


# Solamente uso las columnas  x="high" y="nombre del archivo". Ejm


# In[15]:


data = pd.read_csv("{}".format(files[0]),delimiter=",")
labels=data.keys()
print(labels)
print(np.shape(data))


# In[80]:


#las dimensiones son 1249 (cada compañía)


# In[94]:


#tomo todos los elementos que no tengan celdas vacías
data[label[0]][1]


# In[102]:


n_max=1200#number of files taken
n_data=120# last days taken
#n_max=len(files)
X=[]
Y=[]
date=[]
cnt=0
for f in files[:n_max]:
    if(os.stat("{}".format(f)).st_size != 0):
        data = pd.read_csv("{}".format(f),delimiter=",")
        label=data.keys()
        if(len(data[label[0]])>119):
            X=np.append(X,data[label[2]][-n_data:])#toma todos los datos con high
            if(cnt==0):
                date=np.append(date,data[label[0]][-n_data:])#toma todos los datos con high
            cnt+=1
#        Y=np.append(Y,data[label[4]][-n_data:])#toma todos los datos con high
X=X.reshape(cnt,n_data)
for i in range(len(date)):
    for j in [8,9,10,11]:
        if("-{}-".format(j) in date[i]):
            Y=np.append(Y,i)
#Y=np.arange(cnt)
#Y=Y.reshape(n_max,n_data)
#print(20*3201)
print(np.shape(X))
print(np.shape(Y))


# In[104]:


print(len(date))
31863/120
print((date))


# In[34]:


#Y
#plt.plot(np.arange(50),X[0,:])


# In[82]:


# Vamos a hacer un split training test
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.5)


# In[83]:


x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#y_train = scaler.fit_transform(y_train.reshape(-1, 1))
#y_test = scaler.transform(y_test.reshape(-1, 1))
print(np.shape(x_train))
print(np.shape(x_test))
print(np.shape(y_train))


# ## para l1

# In[68]:


# Turn up tolerance for faster convergence
train_samples = int(np.shape(Y)[0]*0.5)
#regresión logística sobre los dígitos
for i in np.log(np.arange(1,1000,10)):
    clf = LogisticRegression(
        C=i, penalty='l1', solver='saga', tol=0.1)
        #C=50. / train_samples, penalty='l1', solver='saga', tol=0.1)#,multi_class='multinomial'
    clf.fit(x_train, y_train)
    y_pred=clf.predict(x_test)
    print(f1_score(y_test,y_pred, average='weighted'))
#     sparsity = np.mean(clf.coef_ == 0) * 100
#     score = clf.score(x_test, y_test)
#     # print('Best C % .4f' % clf.C_)
#     print("Sparsity with L1 penalty: %.2f%%" % sparsity)
#     print("Test score with L1 penalty: %.4f" % score)


# ## para l2

# In[78]:


# Turn up tolerance for faster convergence
train_samples = int(np.shape(Y)[0]*0.5)
#regresión logística sobre los dígitos
for i in np.log(np.arange(1,1000,10)):
    clf = LogisticRegression(
        C=i, penalty='l2', solver='saga', tol=0.1)
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    print(f1_score(y_test,y_pred, average='weighted'))
#     sparsity = np.mean(clf.coef_ == 0) * 100
#     score = clf.score(x_test, y_test)
#     # print('Best C % .4f' % clf.C_)
#     print("Sparsity with L1 penalty: %.2f%%" % sparsity)
#     print("Test score with L1 penalty: %.4f" % score)


# In[37]:


print(np.shape(y_test))
print(np.shape(y_pred))


# In[41]:





# # Discusión

# Al aplicar umap se observa el agrupamiento de los elementos en un conjunto de líneas, en las cuales cada número está relacionado con el nombre de un mercado de la siguiente forma:

# In[10]:


print("Mercado"," ","Número")
for i,f in enumerate(files[:n_max]):
    if(os.stat("{}".format(f)).st_size != 0):
        print(f[9:-4]," ",i)
 


# Se observa que al aumentar el número de vecinos cercanos se van formando líneas más "nítidas", de hecho para el caso neighbors=2 no se encuentra un agrupamiento, por otro lado, al aumentar la distancia "min_dist" se van haciendo más gruesas algunas partes de las líneas. Si miramos diferentes métricas, se observa que la métrica euclidiana separa los datos en líneas para un numero más pequeño de neighbors que en los otros casos.

# In[ ]:





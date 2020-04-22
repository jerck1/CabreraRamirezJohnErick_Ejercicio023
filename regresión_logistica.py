#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


from sklearn.metrics import f1_score


# In[3]:


files=glob.glob("./Stocks/*.txt")
print(np.shape(files))
print(files[:20])


# In[4]:


# Solamente uso las columnas  x="high" y="nombre del archivo". Ejm


# In[5]:


data = pd.read_csv("{}".format(files[0]),delimiter=",")
labels=data.keys()
print(labels)
print(np.shape(data))


# In[13]:


#las dimensiones son 1249 (cada compañía)
print(labels[0])


# In[8]:


#tomo todos los elementos que no tengan celdas vacías


# In[15]:


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
            X=np.append(X,data[labels[2]][-n_data:])#toma todos los datos con high
            if(cnt==0):
                date=np.append(date,data[labels[0]][-n_data:])#toma todos los dates
            cnt+=1
X=(X.reshape(cnt,n_data)).transpose()
#las categorías son los meses del año
for i in range(len(date)):
    for j in range(5,12):
        if("-0{}-".format(j) in date[i]):
            Y=np.append(Y,j)
        elif("-{}-".format(j) in date[i]):
            Y=np.append(Y,j)
print(np.shape(X))
print(np.shape(Y))


# In[16]:


print(len(date))
print((Y))


# In[17]:


plt.scatter(Y,X[:,1])


# In[18]:


# Vamos a hacer un split training test
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.5)


# In[19]:


x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#y_train = scaler.fit_transform(y_train.reshape(-1, 1))
#y_test = scaler.transform(y_test.reshape(-1, 1))
print(np.shape(x_train))
print(np.shape(x_test))
print(np.shape(y_train))


# ## para l1

# In[20]:


# Turn up tolerance for faster convergence
train_samples = int(np.shape(Y)[0]*0.5)
f1_av_1=[]
#regresión logística sobre los dígitos
for i in np.log(np.arange(1,1000,10)):
    clf = LogisticRegression(
        C=i, penalty='l1', solver='saga', tol=0.1)
        #C=50. / train_samples, penalty='l1', solver='saga', tol=0.1)#,multi_class='multinomial'
    clf.fit(x_train, y_train)
    y_pred=clf.predict(x_test)
    f1_av_1=np.append(f1_av_1,f1_score(y_test,y_pred, average='weighted'))
#     sparsity = np.mean(clf.coef_ == 0) * 100
#     score = clf.score(x_test, y_test)
#     # print('Best C % .4f' % clf.C_)
#     print("Sparsity with L1 penalty: %.2f%%" % sparsity)
#     print("Test score with L1 penalty: %.4f" % score)


# ## para l2

# In[21]:


# lab_enc = preprocessing.LabelEncoder()
# y_train = lab_enc.fit_transform(y_train)
# y_test = lab_enc.fit_transform(y_test)
# # Turn up tolerance for faster convergence
# train_samples = int(np.shape(Y)[0]*0.5)
# #regresión logística sobre los dígitos
f1_av_2=[]
for i in np.log(np.arange(np.e,1000,10)):
    clf = LogisticRegression(
        C=i, penalty='l2', solver='saga', tol=0.1)
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    f1_av_2=np.append(f1_av_2,f1_score(y_test,y_pred, average='weighted'))
#     sparsity = np.mean(clf.coef_ == 0) * 100
#     score = clf.score(x_test, y_test)
#     # print('Best C % .4f' % clf.C_)
#     print("Sparsity with L1 penalty: %.2f%%" % sparsity)
#     print("Test score with L1 penalty: %.4f" % score)


# In[30]:


plt.figure()
#plt.scatter(np.log(np.arange(np.e,1000,10)),f1_av_1,label="f1")
plt.scatter(np.log(np.arange(np.e,1000,10)),f1_av_2)#,label="f2")


# In[33]:


print(f1_av_1)
print(np.log(np.arange(np.e,1000,10)))


# # Discusión

# * Se usan los precio más alto de diferentes mercados en los últimos 120 días y la idea es clasificarlas en meses, 

# In[23]:


print("Mercado"," ","Número")
for i,f in enumerate(files[:n_max]):
    if(os.stat("{}".format(f)).st_size != 0):
        print(f[9:-4]," ",i)
 


# 

# In[ ]:





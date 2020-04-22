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


# In[13]:


files=glob.glob("./Stocks/*.txt")
print(np.shape(files))
print(files[:20])


# In[14]:


# Analizo todas los  x="high" y="nombre del archivo". Ejm


# In[15]:


data = pd.read_csv("{}".format(files[0]),delimiter=",")
labels=data.keys()
print(labels)
print(np.shape(data))


# In[16]:


#tomo todos los elementos que no tengan celdas vacías


# In[17]:


n_max=1200#number of files taken
n_data=50# last days taken
#n_max=len(files)
X=[]
cnt=0
for f in files[:n_max]:
    if(os.stat("{}".format(f)).st_size != 0):
        data = pd.read_csv("{}".format(f),delimiter=",")
        label=data.keys()
        if(len(data[label[0]])>49):
            X=np.append(X,data[label[2]][-n_data:])#toma todos los datos con high
            cnt+=1
#        Y=np.append(Y,data[label[4]][-n_data:])#toma todos los datos con high
X=X.reshape(cnt,n_data)
Y=np.arange(cnt)
#Y=Y.reshape(n_max,n_data)
#print(20*3201)
print(np.shape(X))
print(np.shape(Y))


# In[ ]:





# In[ ]:





# # Discusión

# Al aplicar umap se observa el agrupamiento de los elementos en un conjunto de líneas, en las cuales cada número está relacionado con el nombre de un mercado de la siguiente forma:

# In[10]:


print("Mercado"," ","Número")
for i,f in enumerate(files[:n_max]):
    if(os.stat("{}".format(f)).st_size != 0):
        print(f[9:-4]," ",i)
 


# Se observa que al aumentar el número de vecinos cercanos se van formando líneas más "nítidas", de hecho para el caso neighbors=2 no se encuentra un agrupamiento, por otro lado, al aumentar la distancia "min_dist" se van haciendo más gruesas algunas partes de las líneas. Si miramos diferentes métricas, se observa que la métrica euclidiana separa los datos en líneas para un numero más pequeño de neighbors que en los otros casos.

# In[ ]:





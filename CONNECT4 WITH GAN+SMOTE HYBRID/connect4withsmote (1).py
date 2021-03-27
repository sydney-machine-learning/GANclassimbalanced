#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


data = pd.read_csv("/kaggle/input/connect-4/c4_game_database.csv", sep=",", header='infer' )


# In[3]:


data


# In[4]:


X = data.values[:,0:42].astype(float)
Y = data.values[:,42]


# In[5]:


preX = data[:,0:42]
preY = data[:,42]


# In[6]:


from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


# In[7]:


encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)


# In[8]:


ysi=pd.Series(encoded_Y) 
ysi.value_counts()


# In[9]:


yk=[]
for i in encoded_Y:
    if i==1:
        yk.append(1)
    else:
        yk.append(0)


# In[10]:


ysi=pd.Series(yk) 
ysi.value_counts()


# In[11]:


y = np.asarray(yk, dtype=np.float32)
y.shape


# In[12]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[13]:


print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))
from imblearn.over_sampling import SMOTE
X_train_res,y_train_res = SMOTE().fit_resample(X_train,y_train)

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))


# In[14]:


# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
from sklearn.metrics import f1_score
from statistics import stdev


# In[15]:


def callf1(xx,yy,xt,yt):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam',
                  loss='mean_absolute_error',
                  metrics=['accuracy'])
    model.fit(xx, yy , epochs=3)
    ls=[]
    test_loss, test_acc = model.evaluate(xt,  yt, verbose=2)
    print('\nTest accuracy:', test_acc)
    tr_loss, tr_acc = model.evaluate(xx,  yy, verbose=2)
    ls.append(test_acc)
    ls.append(tr_acc)
    ypr=model.predict(xt)
    ypr=(ypr>0.5)*1
    ypre= np.ravel(ypr)
    ls.append(f1_score(yt, ypre))
    return ls


# In[16]:


r=callf1(X_train, y_train.ravel(),X_test,y_test.ravel())


# In[17]:


r


# In[18]:


r=callf1(X_train_res,y_train_res.ravel(),X_test,y_test.ravel())


# In[19]:


r


# In[20]:


sta=[]
ste=[]
sf =[]
for i in range(30):
    r=callf1(X_train, y_train.ravel(),X_test,y_test.ravel())
    ste.append(r[0])
    sta.append(r[1])
    sf.append(r[2])


# In[21]:


sum(sf)/30


# In[54]:


sum(sta)/30


# In[55]:


sum(ste)/30


# In[56]:


max(sf)


# In[57]:


max(sta)


# In[58]:


max(ste)


# In[59]:


stdev(sf)


# In[60]:


stdev(sta)


# In[61]:


stdev(ste)


# In[22]:


sta2=[]
ste2=[]
sf2 =[]
for i in range(30):
    r=callf1(X_train_res,y_train_res.ravel(),X_test,y_test.ravel())
    ste2.append(r[0])
    sta2.append(r[1])
    sf2.append(r[2])


# In[53]:


sum(sf2)/30


# In[62]:


max(sf2)


# In[63]:


stdev(sf2)


# In[64]:


sum(ste2)/30


# In[65]:


max(ste2)


# In[66]:


stdev(ste2)


# In[67]:


sum(sta2)/30


# In[68]:


max(sta2)


# In[69]:


stdev(sta2)


# In[72]:


save=1


# In[73]:


progress=1


# In[ ]:


save2=1


# In[ ]:





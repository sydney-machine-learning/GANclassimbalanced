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


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder


# In[3]:


data = pd.read_csv("/kaggle/input/glass/glass.csv",delimiter=",")


# In[4]:


data.head()


# In[5]:


from sklearn.utils import shuffle
data=shuffle(data)
data


# In[6]:


# split into input (X) and output (Y) variables

X = data.values[0:,0:9].astype(float)
Y = data.values[0:,9]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)


# In[7]:


X


# In[8]:


data["Type"].value_counts()


# In[9]:


from sklearn.preprocessing import StandardScaler

Xn = StandardScaler().fit_transform(X)

Xn.shape


# In[10]:


Xn


# In[11]:


from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=0)

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)


# In[12]:




sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))


# In[13]:


# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


# In[31]:


model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(6,activation='softmax')
])


# In[32]:


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[33]:


from keras.utils import np_utils
dummy_y = np_utils.to_categorical(y_train_res)


# In[34]:


model.fit(X_train_res,dummy_y , epochs=250)


# In[35]:


dummy_y2 = np_utils.to_categorical(y_test)


# In[36]:


test_loss, test_acc = model.evaluate(X_test,  dummy_y2, verbose=2)

print('\nTest accuracy:', test_acc)


# In[ ]:


y_test


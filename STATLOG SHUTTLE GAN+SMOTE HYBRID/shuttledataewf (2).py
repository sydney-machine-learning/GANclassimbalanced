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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


# In[3]:


data = pd.read_csv("/kaggle/input/shuttledat/shuttle (1).trn", sep=",", header='infer' )


# In[109]:


dt2=pd.read_csv("/kaggle/input/shuttledat/shuttle (1).tst", sep=",", header='infer' )


# In[110]:


dt2


# In[4]:


data


# In[5]:


dn=data.to_numpy()


# In[111]:


dnn=dt2.to_numpy()


# In[6]:


l=list(map(int,dn[0][0].split()))


# In[7]:


ln=[]
yn=[]


# In[8]:


for i in range(43499):
    l=list(map(int,dn[i][0].split()))
    ln.append(l[:9])
    yn.append(l[9])


# In[112]:


lnn=[]
ynn=[]


# In[113]:


for i in range(14499):
    l=list(map(int,dnn[i][0].split()))
    lnn.append(l[:9])
    ynn.append(l[9])


# In[114]:


yok=np.array(ynn)
ysi=pd.Series(yok) 


# In[115]:


ysi.value_counts()


# In[9]:


X = np.asarray(ln, dtype=np.float32)


# In[116]:


Xtes = np.asarray(lnn, dtype=np.float32)


# In[117]:


Xtes.shape


# In[11]:


y=np.array(yn)
ys=pd.Series(y) 


# In[12]:


ys.value_counts()


# In[13]:


yk=[]
for i in yn:
    if i==3:
        yk.append(1)
    else:
        yk.append(0)


# In[118]:


ykk=[]
for i in ynn:
    if i==3:
        ykk.append(1)
    else:
        ykk.append(0)


# In[14]:


ynew=np.array(yk)


# In[119]:


ytes=np.array(ykk)


# In[120]:


ytes.shape


# In[121]:


sum(ytes)


# In[15]:


ys=pd.Series(ynew) 


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X,ynew, random_state=10, test_size=0.2)


# In[18]:


sum(y_train)


# In[19]:


from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)


# In[20]:


y_resampled.shape


# In[21]:


sum(y_resampled)


# In[122]:


ok=100


# In[123]:


pk=1000


# In[22]:



# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
from sklearn.metrics import f1_score


# In[27]:


def callf1(xx,yy,xt,yt):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam',
                  loss='mean_absolute_error',
                  metrics=['accuracy'])
    model.fit(xx, yy , epochs=30)
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


# In[28]:


callf1(X_train,y_train.ravel(),X_test,y_test.ravel())


# In[29]:


sta=[]
ste=[]
sf =[]
for i in range(30):
    r=callf1(X_train,y_train.ravel(),X_test,y_test.ravel())
    ste.append(r[0])
    sta.append(r[1])
    sf.append(r[2])


# In[33]:


sta2=[]
ste2=[]
sf2 =[]
for i in range(30):
    r=callf1(X_resampled, y_resampled.ravel(),X_test,y_test.ravel())
    ste2.append(r[0])
    sta2.append(r[1])
    sf2.append(r[2])


# In[42]:


kag=1


# In[43]:


from statistics import stdev


# In[44]:


sum(sta)/30


# In[45]:


sum(sta2)/30


# In[46]:


max(sta)


# In[47]:


max(sta2)


# In[48]:


stdev(sta)


# In[49]:


stdev(sta2)


# In[50]:


sum(ste)/30


# In[51]:


sum(ste2)/30


# In[52]:


max(ste)


# In[53]:


max(ste2)


# In[54]:


stdev(ste)


# In[55]:


stdev(ste2)


# In[57]:


sum(sf)/30


# In[58]:


sum(sf2)/30


# In[59]:


max(sf)


# In[60]:


max(sf2)


# In[62]:


stdev(sf)


# In[63]:


stdev(sf2)


# In[64]:


stan=[]
sten=[]
sfn =[]
for i in range(30):
    r=callf1(X_train,y_train.ravel(),X_test,y_test.ravel())
    sten.append(r[0])
    stan.append(r[1])
    sfn.append(r[2])


# In[94]:


kwow2=2


# In[96]:


sum(stan)/30


# In[97]:


max(stan)


# In[98]:


stdev(stan)


# In[99]:


sum(sten)/30


# In[100]:


max(sten)


# In[101]:


stdev(sten)


# In[102]:


sum(sfn)/30


# In[104]:


max(sfn)


# In[105]:


stdev(sfn)


# In[108]:


save=1


# In[124]:


from imblearn.over_sampling import SMOTE
X_resampled2, y_resampled2 = SMOTE().fit_resample(X, ynew)


# In[126]:


y_resampled2.shape


# In[127]:


sum(y_resampled2)


# In[128]:


strr=[]
stes=[]
sfo =[]
for i in range(30):
    r=callf1(X,ynew.ravel(),Xtes,ytes.ravel())
    stes.append(r[0])
    strr.append(r[1])
    sfo.append(r[2])


# In[156]:


well=1


# In[158]:


strr2=[]
stes2=[]
sfo2 =[]
for i in range(30):
    r=callf1(X_resampled2,y_resampled2.ravel(),Xtes,ytes.ravel())
    stes2.append(r[0])
    strr2.append(r[1])
    sfo2.append(r[2])


# In[179]:


ok1=1


# In[180]:


sum(sfo2)/30


# In[181]:



max(sfo2)


# In[182]:


stdev(sfo2)


# In[183]:


sum(strr2)/30


# In[184]:


max(strr2)


# In[185]:


stdev(strr2)


# In[187]:


sum(stes2)/30


# In[188]:


max(stes2)


# In[189]:


stdev(stes2)


# In[190]:


saveit=1


# In[ ]:


plssaveit=1


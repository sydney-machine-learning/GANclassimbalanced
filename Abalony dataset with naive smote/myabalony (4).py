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


# In[3]:


data = pd.read_csv("/kaggle/input/abalone/abalone_csv.csv", sep=",", header='infer' )


# In[4]:


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


# In[5]:


data.head()


# In[6]:


category = np.repeat("empty000", data.shape[0])
category.size


# In[7]:


for i in range(0, data["Class_number_of_rings"].size):
    if(data["Class_number_of_rings"][i] <= 7):
        category[i] = int(1)
    elif(data["Class_number_of_rings"][i] > 7):
        category[i] = int(0)


# In[8]:


category[10:25]


# In[9]:


data.Class_number_of_rings.value_counts()


# In[10]:


label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(data['Sex'])
print(integer_encoded[0:7])


# In[11]:


onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
onehot_encoded[0:7]
#onehot_encoded.shape


# In[12]:


data = data.drop(['Sex'], axis=1)
#print((onehot_encoded))
#size(np.array(onehot_encoded.tolist()))
#data['Gender'] = np.array(onehot_encoded.tolist())
data['category_size'] = category
data = data.drop(['Class_number_of_rings'], axis=1)


# In[13]:


data.head()


# In[14]:


features = data.iloc[:,np.r_[0:7]]
labels = data.iloc[:,7]


# In[15]:


data.category_size.value_counts()


# In[16]:


features.head()
#labels.head()


# In[17]:


data.describe()


# In[18]:


X_train, X_test, y_train, y_test, X_gender, X_gender_test = train_test_split(features, labels, onehot_encoded, random_state=10, test_size=0.2)


# In[19]:


X_train.shape


# In[20]:


X_gender.shape


# In[21]:


y_train[0:3]


# In[22]:


X_test.shape


# In[23]:


X_test[1:3]


# In[24]:


temp = X_train.values
X_train_gender = np.concatenate((temp, X_gender), axis=1)
X_train_gender.shape


# In[25]:


temp = X_test.values
X_test_gender = np.concatenate((temp, X_gender_test), axis=1)
X_test_gender.shape


# In[28]:


X_train_gender


# In[29]:


X_test_gender


# In[30]:


from imblearn.over_sampling import SMOTE


# In[31]:


y_train


# In[32]:


class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(data['category_size']), y=data['category_size'])
class_weights_dict = dict(enumerate(class_weights))
class_weights_dict


# In[33]:


old_keys = np.r_[0:27]
new_keys = np.unique(data['category_size'])
weights_final = dict(zip(new_keys, list(class_weights_dict.values()))) 
#class_weights_dict[new_keys] = class_weights_dict.pop(old_keys)
#class_weights_dict.keys()
weights_final


# In[34]:


X_train_gender[0:5]


# In[35]:


y_train.value_counts()


# In[36]:


X_resampled, y_resampled = SMOTE().fit_sample(X_train_gender, y_train)


# In[37]:


X_resampled.shape


# In[38]:


X_train_gender.shape


# In[39]:


y_resampled.value_counts()


# In[40]:


y_resampled


# In[41]:


rf_classifier = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                       max_features='auto', n_estimators=30, oob_score=False,n_jobs=-1)
rf_classifier.fit(X_train_gender, y_train)


# In[42]:


rf_classifier_train = rf_classifier.predict(X_train_gender)
accuracy_score(y_train, rf_classifier_train)


# In[43]:


rf_classifier_test = rf_classifier.predict(X_test_gender)
accuracy_score(y_test, rf_classifier_test)


# In[44]:


rf_classifiert = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                       max_features='auto', n_estimators=30, oob_score=False,n_jobs=-1)
rf_classifiert.fit(X_resampled, y_resampled)


# In[45]:


rf_classifier_tran = rf_classifiert.predict(X_resampled)
accuracy_score(y_resampled, rf_classifier_tran)


# In[46]:


rf_classifier_tes = rf_classifiert.predict(X_test_gender)
accuracy_score(y_test, rf_classifier_tes)


# In[47]:


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


# In[48]:


Xn,yn=shuffle_in_unison(X_resampled, y_resampled)


# In[54]:


rf_classifiert = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                       max_features='auto', n_estimators=30, oob_score=False,n_jobs=-1)
rf_classifiert.fit(Xn, yn)


# In[55]:


rf_classifier_tran = rf_classifiert.predict(Xn)
accuracy_score(yn, rf_classifier_tran)


# In[56]:


rf_classifier_tes = rf_classifiert.predict(X_test_gender)
accuracy_score(y_test, rf_classifier_tes)


# In[57]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train_gender, y_train)


# In[58]:


y_pred = svclassifier.predict(X_test_gender)
accuracy_score(y_test, y_pred)


# In[59]:


svclassifier2 = SVC(kernel='linear')
svclassifier2.fit(Xn, yn)


# In[60]:


y_pred2 = svclassifier2.predict(X_test_gender)
accuracy_score(y_test, y_pred2)


# In[61]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,rf_classifier_test))
print(classification_report(y_test,rf_classifier_test))


# In[62]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,rf_classifier_tes))
print(classification_report(y_test,rf_classifier_tes))


# In[63]:


# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


# In[74]:


model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
])


# In[75]:


model.compile(optimizer='adam',
              loss='BinaryCrossentropy',
              metrics=['accuracy'])


# In[68]:


for i in range(5332):
    if(yn[i] == "1"):
        yn[i]=1
    elif(yn[i]=="0"):
        yn[i] = 0


# In[70]:


yn.shape


# In[71]:


y_test


# In[81]:


Xn


# In[93]:


l=[]
for i in range(len(yn)):
    l.append(1*yn[i])
l    


# In[97]:


ynn=np.array(l)
ynn.shape


# In[98]:


model.fit(Xn,ynn, epochs=300)


# In[100]:


type(y_test)


# In[101]:


yt=y_test.to_numpy()


# In[102]:


yt


# In[107]:


int(yt[0])


# In[108]:


l2=[]
for j in range(len(yt)):
    l2.append(1*int(yt[j]))
l2    


# In[109]:


ytn=np.array(l2)
ytn.shape


# In[110]:


X_test_gender.shape


# In[111]:


Xtest_loss, test_acc = model.evaluate(X_test_gender, ytn , verbose=2)

print('\nTest accuracy:', test_acc)


# In[112]:


ypre=model.predict(X_test_gender)


# In[113]:


ypre= np.ravel(ypre)
ypre.shape


# In[131]:


ypr=(ypre>0.565)*1
ypr


# In[132]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(ytn,ypr))
print(classification_report(ytn,ypr))


# In[134]:


ytn.shape


# In[ ]:


ypr.shape


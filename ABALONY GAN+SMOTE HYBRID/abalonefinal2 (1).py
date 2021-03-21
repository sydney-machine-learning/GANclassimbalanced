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
data = pd.read_csv("/kaggle/input/abalone/abalone_csv.csv", sep=",", header='infer' )
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
category = np.repeat("empty000", data.shape[0])
category.size
for i in range(0, data["Class_number_of_rings"].size):
    if(data["Class_number_of_rings"][i] <= 7):
        category[i] = int(1)
    elif(data["Class_number_of_rings"][i] > 7):
        category[i] = int(0)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(data['Sex'])
print(integer_encoded[0:7])
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
onehot_encoded[0:7]
#onehot_encoded.shape


# In[3]:


from statistics import stdev


# In[4]:


data = data.drop(['Sex'], axis=1)
#print((onehot_encoded))
#size(np.array(onehot_encoded.tolist()))
#data['Gender'] = np.array(onehot_encoded.tolist())
data['category_size'] = category
data = data.drop(['Class_number_of_rings'], axis=1)
features = data.iloc[:,np.r_[0:7]]
labels = data.iloc[:,7]
X_train, X_test, y_train, y_test, X_gender, X_gender_test = train_test_split(features, labels, onehot_encoded, random_state=10, test_size=0.2)
temp = X_train.values
X_train_gender = np.concatenate((temp, X_gender), axis=1)
X_train_gender.shape
temp = X_test.values
X_test_gender = np.concatenate((temp, X_gender_test), axis=1)
X_test_gender.shape
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X_train_gender, y_train)


# In[5]:


l=[]
for i in y_resampled[3341:]:
    l.append(int(i))


# In[6]:


X_oversampled=X_resampled[3341:]


# In[7]:


y_oversampled=np.array(l)
y_oversampled.shape


# In[8]:



# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
from sklearn.metrics import f1_score


# THE NEURAL NETWORK FUNCTION :

# In[9]:


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


# In[10]:


test_list = [int(i) for i in y_test.ravel()] 


# In[11]:


Y_test=np.array(test_list)
Y_test.shape


# In[12]:


train_list = [int(i) for i in y_train.ravel()] 


# In[13]:


Y_train=np.array(train_list)
Y_train.shape


# In[14]:


yr= [int(i) for i in y_resampled.ravel()] 
Y_resam=np.array(yr)
Y_resam.shape


# SIMPLE NEURAL NETWORK:

# In[15]:


sta=[]
ste=[]
sf =[]
for i in range(30):
    r=callf1(X_train_gender,Y_train.ravel(),X_test_gender,Y_test.ravel())
    ste.append(r[0])
    sta.append(r[1])
    sf.append(r[2])


# In[16]:


sum(sta)/30


# In[17]:


sum(ste)/30


# In[18]:


sum(sf)/30


# In[19]:


max(sta)


# In[20]:


max(ste)


# In[21]:


max(sf)


# In[22]:


stdev(sta)


# In[23]:


stdev(ste)


# In[24]:


stdev(sf)


# NEURAL NETWORKS WITH SMOTE:

# In[25]:


s2=0
sta2=[]
ste2=[]
sf2 =[]
for i in range(30):
    r=callf1(X_resampled,Y_resam.ravel(),X_test_gender,Y_test.ravel())
    ste2.append(r[0])
    sta2.append(r[1])
    sf2.append(r[2])


# In[26]:


sum(sta2)/30


# In[27]:


sum(ste2)/30


# In[28]:


sum(sf2)/30


# In[29]:


max(sta2)


# In[30]:


max(ste2)


# In[31]:


max(sf2)


# In[32]:


stdev(sta2)


# In[33]:


stdev(ste2)


# In[34]:


stdev(sf2)


# GAN MODEL:

# In[35]:


import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# In[36]:


def get_generator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True),
    )
class Generator(nn.Module):

    def __init__(self, z_dim=10, im_dim=10, hidden_dim=128):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()
        )
    def forward(self, noise):
        return self.gen(noise)
    
    # Needed for grading
    def get_gen(self):

        return self.gen
def get_discriminator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(0.2, inplace=True)        
    )
class Discriminator(nn.Module):
    def __init__(self, im_dim=10, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image):

        return self.disc(image)
    
    def get_disc(self):

        return self.dis


# In[37]:


X_oversampled = torch.from_numpy(X_oversampled)


# In[38]:


criterion = nn.BCEWithLogitsLoss()
n_epochs = 400
z_dim = 10
batch_size = 128
lr = 0.00001
display_step = 500
device = 'cpu'


# In[39]:


gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device) 
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)


# In[40]:


def get_disc_loss(gen, disc, criterion, real, device):

    fake = gen(X_oversampled.float())
    disc_fake_pred = disc(fake.detach())
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    disc_real_pred = disc(real)
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    return disc_loss


# In[41]:


def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    fake_images = gen(X_oversampled.float())
    
    disc_fake_pred = disc(fake_images)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
    return gen_loss


# In[42]:


ytr=Y_train.ravel()


# In[43]:


li=[]
for i in range(len(ytr)):
    if int(ytr[i])==1:
        li.append(X_train_gender[i])


# In[44]:


X_real=np.array(li)


# In[45]:


X_real.shape


# In[46]:


Y_train.sum()


# In[47]:


li2=[1]*675


# In[48]:


y_real=np.array(li2)
y_real.shape


# In[49]:


from torch.utils.data import TensorDataset, DataLoader
tensor_x = torch.Tensor(X_real) 
tensor_y = torch.Tensor(y_real)
my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset


# In[50]:


dataloader = DataLoader(
    my_dataset,
    batch_size=batch_size,
    shuffle=True)


# In[51]:


cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
test_generator = True 
gen_loss = False
error = False
for epoch in range(n_epochs):
  
    # Dataloader returns the batches
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)

        # Flatten the batch of real images from the dataset
        real = real.view(cur_batch_size, -1).to(device)

        ### Update discriminator ###
        # Zero out the gradients before backpropagation
        disc_opt.zero_grad()

        # Calculate discriminator loss
        disc_loss = get_disc_loss(gen, disc, criterion, real, device)

        # Update gradients#
        disc_loss.backward(retain_graph=True)

        # Update optimizer
        disc_opt.step()

        # For testing purposes, to keep track of the generator weights
        if test_generator:
            old_generator_weights = gen.gen[0][0].weight.detach().clone()

        ### Update generator ###
        #     Hint: This code will look a lot like the discriminator updates!
        #     These are the steps you will need to complete:
        #       1) Zero out the gradients.
        #       2) Calculate the generator loss, assigning it to gen_loss.
        #       3) Backprop through the generator: update the gradients and optimizer.
        #### START CODE HERE ####
        gen_opt.zero_grad()
        gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
        gen_loss.backward()
        gen_opt.step()        
        #### END CODE HERE ####

        # For testing purposes, to check that your code changes the generator weights
        if test_generator:
            try:
                assert lr > 0.0000002 or (gen.gen[0][0].weight.grad.abs().max() < 0.0005 and epoch == 0)
                assert torch.any(gen.gen[0][0].weight.detach().clone() != old_generator_weights)
            except:
                error = True
                print("Runtime tests have failed")

        # Keep track of the average discriminator loss
        mean_discriminator_loss += disc_loss.item() / display_step

        # Keep track of the average generator loss
        mean_generator_loss += gen_loss.item() / display_step

        ### Visualization code ###
        if cur_step % display_step == 0 and cur_step > 0:
            print(f"Epoch {epoch}, step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            #fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            #fake = gen(fake_noise)
            #show_tensor_images(fake)
            #show_tensor_images(real)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1


# In[52]:


res=gen(X_oversampled.float())


# In[53]:


res


# In[54]:


X_oversampled


# In[55]:


fres=res.detach().numpy()
fres.shape


# CREATING THE GAN AND SMOTE UPSAMPLED DATASET:

# In[57]:


fin=np.concatenate((X_resampled[:3341], fres), axis=0)


# In[58]:


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


# In[59]:


Xnew,ynew=shuffle_in_unison(fin, Y_resam)


# NEURAL NETWORK ON GAN + SMOTE DATASET:

# In[60]:


s3=0
sta3=[]
ste3=[]
sf3 =[]
for i in range(30):
    r=callf1(Xnew,ynew,X_test_gender,Y_test.ravel())
    ste3.append(r[0])
    sta3.append(r[1])
    sf3.append(r[2])


# In[61]:


sum(sta3)/30


# In[62]:


sum(ste3)/30


# In[63]:


sum(sf3)/30


# In[64]:


max(sta3)


# In[65]:


max(ste3)


# In[66]:


max(sf3)


# In[67]:


stdev(sta3)


# In[68]:


stdev(ste3)


# In[69]:


stdev(sf3)


# In[70]:


save=1


# In[ ]:


final=1


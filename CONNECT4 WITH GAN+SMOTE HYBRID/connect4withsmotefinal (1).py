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


# In[ ]:


preX = data[:,0:42]
preY = data[:,42]


# In[5]:


from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


# In[6]:


encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)


# In[7]:


ysi=pd.Series(encoded_Y) 
ysi.value_counts()


# In[8]:


yk=[]
for i in encoded_Y:
    if i==1:
        yk.append(1)
    else:
        yk.append(0)


# In[9]:


ysi=pd.Series(yk) 
ysi.value_counts()


# In[10]:


y = np.asarray(yk, dtype=np.float32)
y.shape


# In[11]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[12]:


print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))
from imblearn.over_sampling import SMOTE
X_train_res,y_train_res = SMOTE().fit_resample(X_train,y_train)

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))


# In[13]:


# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
from sklearn.metrics import f1_score
from statistics import stdev


# In[ ]:


model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam',
              loss='mean_absolute_error',
              metrics=['accuracy'])
model.fit(X_train_res,y_train_res.ravel() , epochs=3)


# In[ ]:


test_loss, test_acc = model.evaluate(X_test,  y_test.ravel(), verbose=2)

print('\nTest accuracy:', test_acc)


# In[ ]:


ypre=model.predict(X_test)
ypre= np.ravel(ypre)
ypre.shape
ypr=(ypre>0.5)*1
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test.ravel(),ypr))
print(classification_report(y_test.ravel(),ypr))


# In[ ]:





# In[ ]:





# In[14]:


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


# In[15]:


X_train.shape


# In[ ]:


r=callf1(X_train, y_train.ravel(),X_test,y_test.ravel())


# In[ ]:


r


# In[ ]:


r=callf1(X_train_res,y_train_res.ravel(),X_test,y_test.ravel())


# In[ ]:


r


# In[ ]:


sta=[]
ste=[]
sf =[]
for i in range(30):
    r=callf1(X_train, y_train.ravel(),X_test,y_test.ravel())
    ste.append(r[0])
    sta.append(r[1])
    sf.append(r[2])


# In[ ]:


sum(sf)/30


# In[ ]:


sum(sta)/30


# In[ ]:


sum(ste)/30


# In[ ]:


max(sf)


# In[ ]:


max(sta)


# In[ ]:


max(ste)


# In[ ]:


stdev(sf)


# In[ ]:


stdev(sta)


# In[ ]:


stdev(ste)


# In[ ]:


sta2=[]
ste2=[]
sf2 =[]
for i in range(30):
    r=callf1(X_train_res,y_train_res.ravel(),X_test,y_test.ravel())
    ste2.append(r[0])
    sta2.append(r[1])
    sf2.append(r[2])


# In[ ]:


sum(sf2)/30


# In[ ]:


max(sf2)


# In[ ]:


stdev(sf2)


# In[ ]:


sum(ste2)/30


# In[ ]:


max(ste2)


# In[ ]:


stdev(ste2)


# In[ ]:


sum(sta2)/30


# In[ ]:


max(sta2)


# In[ ]:


stdev(sta2)


# In[ ]:


save=1


# In[ ]:


progress=1


# In[ ]:


save2=1


# In[16]:


X_oversampled=X_train_res[301312:]


# In[43]:


import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# In[44]:


torch.cuda.empty_cache()


# In[45]:


def get_generator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True),
    )


# In[46]:


class Generator(nn.Module):

    def __init__(self, z_dim=42, im_dim=42, hidden_dim=128):
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


# In[47]:


def get_discriminator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(0.2, inplace=True)        
    )


# In[48]:


class Discriminator(nn.Module):
    def __init__(self, im_dim=42, hidden_dim=128):
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


# In[49]:


X_oversampled = torch.from_numpy(X_oversampled)


# In[50]:


X_oversampled.shape


# In[51]:


criterion = nn.BCEWithLogitsLoss()
n_epochs = 5
z_dim = 42
batch_size = 128
lr = 0.00001
display_step = 1
device = 'cuda'


# In[52]:


gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device) 
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)


# In[53]:


def get_disc_loss(gen, disc, criterion, real, device):

    fake = gen(X_oversampled.float().to(device))
    disc_fake_pred = disc(fake.detach())
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    disc_real_pred = disc(real)
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    return disc_loss


# In[54]:


def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    fake_images = gen(X_oversampled.float().to(device))
    
    disc_fake_pred = disc(fake_images)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
    return gen_loss


# In[55]:


y_tr=y_train.ravel()


# In[56]:


li=[]
for i in range(len(y_tr)):
    if int(y_tr[i])==1:
        li.append(X_train[i])


# In[57]:


X_real=np.array(li)


# In[58]:


X_real.shape


# In[59]:


li2=[1]*11562


# In[60]:


y_real=np.array(li2)
y_real.shape


# In[61]:


from torch.utils.data import TensorDataset, DataLoader
tensor_x = torch.Tensor(X_real) 
tensor_y = torch.Tensor(y_real)
my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset


# In[62]:


dataloader = DataLoader(
    my_dataset,
    batch_size=batch_size,
    shuffle=True)


# In[63]:



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


# In[64]:


res=gen(X_oversampled.float().to(device))


# In[65]:


res


# In[66]:


X_oversampled


# In[67]:


fres=res.cpu().detach().numpy()
fres.shape


# In[68]:


fin=np.concatenate((X_train_res[:301312], fres), axis=0)


# In[69]:


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


# In[70]:


y_train_res.shape


# In[71]:


Xn,yn=shuffle_in_unison(fin, y_train_res)


# In[ ]:


callf1(Xn, yn.ravel(),X_test,  y_test.ravel())


# In[72]:


ste3=[]
sta3=[]
sf3=[]
for i in range(30):
    r=callf1(Xn, yn.ravel(),X_test,  y_test.ravel())
    ste3.append(r[0])
    sta3.append(r[1])
    sf3.append(r[2])


# In[116]:


sum(sf3)/30


# In[117]:


sum(sta3)/30


# In[118]:


sum(ste3)/30


# In[119]:


stdev(sf3)


# In[120]:


stdev(sta3)


# In[121]:


stdev(ste3)


# In[122]:


max(sf3)


# In[123]:


max(sta3)


# In[124]:


max(ste3)


# In[127]:


progress=1


# In[128]:


save=1


# In[ ]:





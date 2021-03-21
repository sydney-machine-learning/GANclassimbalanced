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


# In[10]:


df=pd.read_csv('/kaggle/input/spamdat/realspambase.data')


# In[11]:


df.columns


# In[12]:


df["1"].value_counts()


# In[13]:


df.info()


# In[14]:


dict_1={}

dict_1=dict(df.corr()['1'])


# In[15]:


list_features=[]
for key,values in dict_1.items():
    if abs(values)<0.2:
        list_features.append(key)


# In[16]:


#lets drop the features which have coorelation less than <0.2
df=df.drop(list_features,axis=1)


# In[17]:


y=df['1']
x=df.drop(['1'],axis=1)


# In[18]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)


# In[20]:


from imblearn.over_sampling import SMOTE
print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_resample(x_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))


# In[21]:


# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
from sklearn.metrics import f1_score


# In[26]:


from statistics import stdev


# In[44]:


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


# In[45]:


Xnb=X_train_res.to_numpy()


# In[46]:


Xnk,ynk=shuffle_in_unison(Xnb, y_train_res)


# NEURAL NETWORK FUNCTION:

# In[185]:


def callf2(xx,yy,xt,yt):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam',
                  loss='mean_absolute_error',
                  metrics=['accuracy'])
    model.fit(xx, yy , epochs=90)
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


# NEURAL NETWORK ON NON SMOTE DATA:

# In[186]:


ste=[]
sta=[]
sf=[]
for i in range(30):
    r=callf2(x_train,y_train,x_test,y_test)
    ste.append(r[0])
    sta.append(r[1])
    sf.append(r[2])


# In[187]:


sum(sta)/30


# In[188]:


sum(ste)/30


# In[189]:


sum(sf)/30


# In[190]:


max(sta)


# In[191]:


max(ste)


# In[192]:


max(sf)


# In[193]:


stdev(sta)


# In[194]:


stdev(ste)


# In[195]:


stdev(sf)


# NEURAL NETWORK WITH SMOTE OVERSAMPLE:

# In[196]:


ste2=[]
sta2=[]
sf2=[]
for i in range(30):
    r=callf2(Xnk,ynk,x_test,y_test)
    ste2.append(r[0])
    sta2.append(r[1])
    sf2.append(r[2])


# In[197]:


sum(sta2)/30


# In[198]:


sum(ste2)/30


# In[199]:


sum(sf2)/30


# In[200]:


max(sta2)


# In[201]:


max(ste2)


# In[202]:


max(sf2)


# In[203]:


stdev(sta2)


# In[204]:


stdev(ste2)


# In[205]:


stdev(sf2)


# GAN MODEL:

# In[71]:


import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# In[72]:


def get_generator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True),
    )
class Generator(nn.Module):

    def __init__(self, z_dim=19, im_dim=19, hidden_dim=128):
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
    def __init__(self, im_dim=19, hidden_dim=128):
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
    


# In[73]:


Xtb=x_train.to_numpy()


# In[74]:


X_oversampled=Xnb[3680:]


# In[75]:


X_oversampled.shape


# In[76]:


l=[1]*820
y_oversampled=np.array(l)
y_oversampled.shape


# In[77]:


X_oversampled = torch.from_numpy(X_oversampled)


# In[78]:


criterion = nn.BCEWithLogitsLoss()
n_epochs = 400
z_dim = 19
batch_size = 128
lr = 0.00001
display_step = 500
device = 'cpu'


# In[79]:


gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device) 
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)


# In[80]:


def get_disc_loss(gen, disc, criterion, real, device):

    fake = gen(X_oversampled.float())
    disc_fake_pred = disc(fake.detach())
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    disc_real_pred = disc(real)
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    return disc_loss


# In[81]:


def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    fake_images = gen(X_oversampled.float())
    
    disc_fake_pred = disc(fake_images)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
    return gen_loss


# In[83]:


ytr=y_train.ravel()


# In[84]:


li=[]
for i in range(len(ytr)):
    if int(ytr[i])==1:
        li.append(Xtb[i])


# In[85]:


X_real=np.array(li)


# In[86]:


X_real.shape


# In[87]:


li2=[1]*1430


# In[88]:


y_real=np.array(li2)
y_real.shape


# In[89]:


from torch.utils.data import TensorDataset, DataLoader
tensor_x = torch.Tensor(X_real) 
tensor_y = torch.Tensor(y_real)
my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset


# In[90]:


dataloader = DataLoader(
    my_dataset,
    batch_size=batch_size,
    shuffle=True)


# In[91]:


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


# In[92]:


res=gen(X_oversampled.float())


# In[93]:


res


# In[94]:


X_oversampled


# In[95]:


fres=res.detach().numpy()
fres.shape


# GAN+SMOTE DATA:

# In[102]:


fin=np.concatenate((Xnb[:3680], fres), axis=0)


# In[103]:


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


# In[104]:


Xnew,ynew=shuffle_in_unison(fin, y_train_res)


# In[206]:


ste3=[]
sta3=[]
sf3=[]
for i in range(30):
    r=callf2(Xnew,ynew,x_test,y_test)
    ste3.append(r[0])
    sta3.append(r[1])
    sf3.append(r[2])


# In[238]:


sum(sta3)/30


# In[239]:


sum(ste3)/30


# In[240]:


sum(sf3)/30


# In[241]:


max(sta3)


# In[242]:


max(ste3)


# In[243]:


max(sf3)


# In[244]:


stdev(sta3)


# In[245]:


stdev(ste3)


# In[246]:


stdev(sf3)


# In[247]:


save=1


# In[ ]:


final=1


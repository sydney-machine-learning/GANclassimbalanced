#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import os


# In[ ]:


urlist=['https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/page-blocks-1-3_vs_4.dat','https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/ecoli4.dat','https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/poker-8_vs_6.dat','https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/winequality-red-8_vs_6.dat','https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/yeast-2_vs_4.dat']
option=int(input("Enter 0 for page-blocks\nEnter 1 for ecoli\nEnter 2 for poker\nEnter 3 for winequality\nEnter 4 for yeast\n"))


# In[ ]:


import pandas as pd
url=urlist[option]
data = pd.read_csv(url, sep=",", header='infer' )
data


# In[ ]:


t=()
t=data.shape
X = data.values[:,0:(t[1]-1)].astype(float)
Y = data.values[:,(t[1]-1)]
print(X)


# In[ ]:


if option==0 or option==3 or option==2 :
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    print(X)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)
ysi=pd.Series(encoded_Y) 
ysi.value_counts()


# In[ ]:


if option==1:
    yk=[]
    for i in encoded_Y:
        if i==2:
            yk.append(1)
        else:
            yk.append(0)
    encoded_Y = np.asarray(yk, dtype=np.float32)
    encoded_Y.shape


# In[ ]:


if option==0:
    rs=1
    ep=10
    ne=2000
if option==1:
    rs=0
    ep=10
    ne=700
if option==2:
    rs=1
    ep=30
    ne=1500
if option==3:
    rs=4
    ep=30
    ne=1500    
if option==4:
    rs=1
    ep=10
    ne=1000    
    
    


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,encoded_Y, test_size=0.2, random_state=rs)


# In[ ]:


print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))
from imblearn.over_sampling import SMOTE
X_train_res,y_train_res = SMOTE().fit_resample(X_train,y_train)

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))


# In[ ]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
from sklearn.metrics import f1_score
from statistics import stdev


# In[ ]:


def callf1(xx,yy,xt,yt):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam',
                  loss='mean_absolute_error',
                  metrics=['accuracy'])
    model.fit(xx, yy , epochs=ep)
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


sum(sta2)/30


# In[ ]:


sum(ste2)/30


# In[ ]:


max(sf2)


# In[ ]:


max(sta2)


# In[ ]:


max(ste2)


# In[ ]:


stdev(sf2)


# In[ ]:


stdev(sta2)


# In[ ]:


stdev(ste2)


# In[ ]:


t2=X_train.shape


# In[ ]:


X_oversampled=X_train_res[(t2[0]):]


# In[ ]:


import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# In[ ]:


torch.cuda.empty_cache()


# In[ ]:


def get_generator_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True),
    )
class Generator(nn.Module):

    def __init__(self, z_dim=t2[1], im_dim=t2[1], hidden_dim=128):
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
    def __init__(self, im_dim=t2[1], hidden_dim=128):
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
X_oversampled = torch.from_numpy(X_oversampled)
criterion = nn.BCEWithLogitsLoss()
n_epochs = ne
z_dim = t2[1]
batch_size = 128
lr = 0.00001
display_step = 1
device = 'cuda'
gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device) 
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
def get_disc_loss(gen, disc, criterion, real, device):

    fake = gen(X_oversampled.float().to(device))
    disc_fake_pred = disc(fake.detach())
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    disc_real_pred = disc(real)
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    return disc_loss
def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    fake_images = gen(X_oversampled.float().to(device))
    
    disc_fake_pred = disc(fake_images)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
    return gen_loss
y_tr=y_train.ravel()
li=[]
for i in range(len(y_tr)):
    if int(y_tr[i])==1:
        li.append(X_train[i])
X_real=np.array(li)
t3=X_real.shape
li2=[1]*(t3[0])
y_real=np.array(li2)
y_real.shape
from torch.utils.data import TensorDataset, DataLoader
tensor_x = torch.Tensor(X_real) 
tensor_y = torch.Tensor(y_real)
my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
dataloader = DataLoader(
    my_dataset,
    batch_size=batch_size,
    shuffle=True)
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
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1


# In[ ]:


res=gen(X_oversampled.float().to(device))


# In[ ]:


res


# In[ ]:


X_oversampled


# In[ ]:


fres=res.cpu().detach().numpy()
fres.shape


# In[ ]:


tu=X_train.shape


# In[ ]:


fin=np.concatenate((X_train_res[:(tu[0])], fres), axis=0)


# In[ ]:


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


# In[ ]:


y_train_res.shape


# In[ ]:


Xn,yn=shuffle_in_unison(fin, y_train_res)


# In[ ]:


ste3=[]
sta3=[]
sf3=[]
for i in range(30):
    r=callf1(Xn, yn.ravel(),X_test,  y_test.ravel())
    ste3.append(r[0])
    sta3.append(r[1])
    sf3.append(r[2])


# In[ ]:


sum(sf3)/30


# In[ ]:


sum(sta3)/30


# In[ ]:


sum(ste3)/30


# In[ ]:


max(sf3)


# In[ ]:


max(sta3)


# In[ ]:


max(ste3)


# In[ ]:


stdev(sf3)


# In[ ]:


stdev(sta3)


# In[ ]:


stdev(ste3)


# In[ ]:


save=1


# In[ ]:


progress=1


# In[ ]:


save=2


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





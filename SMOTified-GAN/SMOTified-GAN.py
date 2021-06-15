import numpy as np 
import pandas as pd 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
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
urlist=['https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/page-blocks-1-3_vs_4.dat','https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/ecoli4.dat','https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/poker-8_vs_6.dat','https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/winequality-red-8_vs_6.dat','https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/yeast-2_vs_4.dat','https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/abalone_csv.csv',
       'https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/ionosphere_data_kaggle.csv','https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/realspambase%20(1).data',['https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/SHUTTLE/shuttle%20(1).trn','https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/SHUTTLE/shuttle%20(1).tst']]
option=int(input("Enter 0 for page-blocks\nEnter 1 for ecoli\nEnter 2 for poker\nEnter 3 for winequality\nEnter 4 for yeast\nEnter 5 for abalone\nEnter 6 for ionosphere\nEnter 7 for spambase\nEnter 8 for shuttle\n"))

if option!=8:
    import pandas as pd
    url=urlist[option]
    data = pd.read_csv(url, sep=",", header='infer' )
if option==5:
    category = np.repeat("empty000", data.shape[0])
    for i in range(0, data["Class_number_of_rings"].size):
        if(data["Class_number_of_rings"][i] <= 7):
            category[i] = int(1)
        elif(data["Class_number_of_rings"][i] > 7):
            category[i] = int(0)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(data['Sex'])
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    data = data.drop(['Sex'], axis=1)
    data['category_size'] = category
    data = data.drop(['Class_number_of_rings'], axis=1)
    features = data.iloc[:,np.r_[0:7]]
    labels = data.iloc[:,7]
    X_train, X_test, y_train, y_test, X_gender, X_gender_test = train_test_split(features, labels, onehot_encoded, random_state=10, test_size=0.2)
    temp = X_train.values
    X_train = np.concatenate((temp, X_gender), axis=1)
    temp2 = X_test.values
    X_test = np.concatenate((temp2, X_gender_test), axis=1)
    test_list = [int(i) for i in y_test.ravel()] 
    y_test=np.array(test_list)
    train_list = [int(i) for i in y_train.ravel()] 
    y_train=np.array(train_list)
    ep=30
    ne=150
elif option==7:
    dict_1={}
    dict_1=dict(data.corr()['1'])
    list_features=[]
    for key,values in dict_1.items():
        if abs(values)<0.2:
            list_features.append(key)
    data=data.drop(list_features,axis=1)  
    X = df.values[:,0:19].astype(float)
    Y = df.values[:,19]
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=0,test_size=0.2)
    ep=90
    ne=1500
elif option==8:
    url1=urlist[option][0]
    url2=urlist[option][1]
    dtrn = pd.read_csv(url1, sep=",", header='infer' )
    dtes = pd.read_csv(url2, sep=",", header='infer' )
    dtr=dtrn.to_numpy()
    dte=dtes.to_numpy()
    ln=[]
    yn=[]
    for i in range(43499):
        l=list(map(int,dtr[i][0].split()))
        ln.append(l[:9])
        yn.append(l[9])
    lnn=[]
    ynn=[]
    for i in range(14499):
        l=list(map(int,dte[i][0].split()))
        lnn.append(l[:9])
        ynn.append(l[9])    
    X_train = np.asarray(ln, dtype=np.float32)
    X_test = np.asarray(lnn, dtype=np.float32) 
    yk=[]
    for i in yn:
        if i==3:
            yk.append(1)
        else:
            yk.append(0)
    ykk=[]
    for i in ynn:
        if i==3:
            ykk.append(1)
        else:
            ykk.append(0)  
    y_train=np.array(yk)
    y_test=np.array(ykk)  
    ep=10
    ne=1000
else:
    t=()
    t=data.shape
    X = data.values[:,0:(t[1]-1)].astype(float)
    Y = data.values[:,(t[1]-1)]
    
    if option==0 or option==3 or option==2 :
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    
    from sklearn.preprocessing import LabelEncoder
    from keras.utils import np_utils
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    dummy_y = np_utils.to_categorical(encoded_Y)
    if option==1:
        yk=[]
        for i in encoded_Y:
            if i==2:
                yk.append(1)
            else:
                yk.append(0)
        encoded_Y = np.asarray(yk, dtype=np.float32)
        encoded_Y.shape 
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
    if option==6:
        rs=11
        ep=100
        ne=200
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X,encoded_Y, test_size=0.2, random_state=rs)
if option!=6:
    print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
    print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))
    from imblearn.over_sampling import SMOTE
    X_train_res,y_train_res = SMOTE().fit_resample(X_train,y_train)

    print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
    print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

    print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
    print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0))) 
if option==6:
    lst=[]
    lst2=[]
    for j in y_train:
        if j==1:
            lst.append(0)
        else:
            lst.append(1)
    for j2 in y_test:
        if j2==1:
            lst2.append(0)
        else:
            lst2.append(1)
    y_train=np.array(lst)
    y_test=np.array(lst2)              
            
    from imblearn.over_sampling import SMOTE
    print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
    print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

    sm = SMOTE(random_state=2)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())

    print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
    print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

    print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
    print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
from sklearn.metrics import f1_score
from statistics import stdev
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
sta=[]
ste=[]
sf =[]
for i in range(30):
    r=callf1(X_train, y_train.ravel(),X_test,y_test.ravel())
    ste.append(r[0])
    sta.append(r[1])
    sf.append(r[2])

sta2=[]
ste2=[]
sf2 =[]
for i in range(30):
    r=callf1(X_train_res,y_train_res.ravel(),X_test,y_test.ravel())
    ste2.append(r[0])
    sta2.append(r[1])
    sf2.append(r[2])
t2=X_train.shape
X_oversampled=X_train_res[(t2[0]):]
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.cuda.empty_cache()
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
my_dataset = TensorDataset(tensor_x,tensor_y) 
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
  
    
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)

      
        real = real.view(cur_batch_size, -1).to(device)

  
       
        disc_opt.zero_grad()

      
        disc_loss = get_disc_loss(gen, disc, criterion, real, device)

        
        disc_loss.backward(retain_graph=True)

       
        disc_opt.step()

    
        if test_generator:
            old_generator_weights = gen.gen[0][0].weight.detach().clone()

        gen_opt.zero_grad()
        gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
        gen_loss.backward()
        gen_opt.step()        

        if test_generator:
            try:
                assert lr > 0.0000002 or (gen.gen[0][0].weight.grad.abs().max() < 0.0005 and epoch == 0)
                assert torch.any(gen.gen[0][0].weight.detach().clone() != old_generator_weights)
            except:
                error = True
                print("Runtime tests have failed")

    
        mean_discriminator_loss += disc_loss.item() / display_step
        mean_generator_loss += gen_loss.item() / display_step

    
        if cur_step % display_step == 0 and cur_step > 0:
            print(f"Epoch {epoch}, step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1
res=gen(X_oversampled.float().to(device))

fres=res.cpu().detach().numpy()
fres.shape

tu=X_train.shape
fin=np.concatenate((X_train_res[:(tu[0])], fres), axis=0)
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b
Xn,yn=shuffle_in_unison(fin, y_train_res)
ste3=[]
sta3=[]
sf3=[]
for i in range(30):
    r=callf1(Xn, yn.ravel(),X_test,  y_test.ravel())
    ste3.append(r[0])
    sta3.append(r[1])
    sf3.append(r[2])

t2=X_train.shape
X_oversampled=X_train_res[(t2[0]):]
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.cuda.empty_cache()
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
def get_noise(n_samples, z_dim, device='cuda'):

    return torch.randn(n_samples,z_dim,device=device) 
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

def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):

    fake_noise = get_noise(num_images, z_dim, device=device)
    fake = gen(fake_noise)
    disc_fake_pred = disc(fake.detach())
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    disc_real_pred = disc(real)
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
    disc_loss = (disc_fake_loss + disc_real_loss) / 2

    return disc_loss

def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):

    fake_noise = get_noise(num_images, z_dim, device=device)
    fake = gen(fake_noise)
    disc_fake_pred = disc(fake)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
    return gen_loss
from torch.utils.data import TensorDataset, DataLoader
tensor_x = torch.Tensor(X_real) 
tensor_y = torch.Tensor(y_real)
my_dataset = TensorDataset(tensor_x,tensor_y)
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
  
   
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)

       
        real = real.view(cur_batch_size, -1).to(device)

      
       
        disc_opt.zero_grad()

       
        disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)

        
        disc_loss.backward(retain_graph=True)

       
        disc_opt.step()

      
        if test_generator:
            old_generator_weights = gen.gen[0][0].weight.detach().clone()

        gen_opt.zero_grad()
        gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
        gen_loss.backward()
        gen_opt.step()

        if test_generator:
            try:
                assert lr > 0.0000002 or (gen.gen[0][0].weight.grad.abs().max() < 0.0005 and epoch == 0)
                assert torch.any(gen.gen[0][0].weight.detach().clone() != old_generator_weights)
            except:
                error = True
                print("Runtime tests have failed")

     
        mean_discriminator_loss += disc_loss.item() / display_step

  
        mean_generator_loss += gen_loss.item() / display_step

 
        if cur_step % display_step == 0 and cur_step > 0:
            print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")

            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1
t4=X_oversampled.shape
fake_noise = get_noise((t4[0]), z_dim, device=device)
res2=gen(fake_noise)
fres2=res2.cpu().detach().numpy()
fres2.shape
fin2=np.concatenate((X_train_res[:(t2[0])], fres2), axis=0)
Xnew,ynew=shuffle_in_unison(fin2, y_train_res)
s4=0
sta4=[]
ste4=[]
sf4 =[]
for i in range(30):
    r=callf1(Xnew,ynew,X_test,y_test.ravel())
    ste4.append(r[0])
    sta4.append(r[1])
    sf4.append(r[2])  
print(X_oversampled)    
print(res2)
print(res)      
print("Mean train accuracy of NON-OVERSAMPLED=",sta,"\nMean train accuracy of SMOTE=",sta2,
      "\nMean train accuracy of GAN",sta4,"\nMean train accuracy of SMOTified-GAN",sta3)   
print("Mean test accuracy of NON-OVERSAMPLED=",ste,"\nMean test accuracy of SMOTE=",ste2,
      "\nMean test accuracy of GAN",ste4,"\nMean test accuracy of SMOTified-GAN",ste3) 
print("Mean F1 score of NON-OVERSAMPLED=",sf,"\nMean F1 score of SMOTE=",sf2,
      "\nMean F1 score of GAN",sf4,"\nMean F1 score of SMOTified-GAN",sf3)  
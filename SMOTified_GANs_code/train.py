import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from torchvision.datasets.utils import download_url
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
from GANs_model import GANs_Discriminator, GANs_Generator
from model_trainer import train_discriminator, train_generator
from model_fit import SG_fit, G_fit
from test_model import test_model, test_model_lists
from Abalone_data_preprocessing import two_classes_Abalone, four_classes_Abalone, get_features, get_labels, GANs_two_class_real_data, GANs_four_class_real_data
from choose_device import get_default_device, to_device, DeviceDataLoader
from fit import f1

#print("This is train.py file")


def shuffle_in_unison(a, b):     #Shuffling the features and labels in unison.
    assert len(a) == len(b)       #In Python, the assert statement is used to continue the execute if the given condition evaluates to True.
    shuffled_a = np.empty(a.shape, dtype=a.dtype)       #Return a new array of given shape and type, without initializing entries.
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def Oversampled_data(X_train_SMOTE, y_train_SMOTE, y_train, device):
    X_oversampled_0 = []
    X_oversampled_2 = []
    X_oversampled_3 = []
    for i in y_train_SMOTE[(y_train.shape[0]):]:
        if i==0: 
            X_oversampled_0.append(X_train_SMOTE[i])
        elif i==2: 
            X_oversampled_2.append(X_train_SMOTE[i])
        elif i==3: 
            X_oversampled_3.append(X_train_SMOTE[i])
    X_oversampled_0 = torch.from_numpy(np.array(X_oversampled_0))
    X_oversampled_0 = to_device(X_oversampled_0.float(), device)
    X_oversampled_2 = torch.from_numpy(np.array(X_oversampled_2))
    X_oversampled_2 = to_device(X_oversampled_2.float(), device)
    X_oversampled_3 = torch.from_numpy(np.array(X_oversampled_3))
    X_oversampled_3 = to_device(X_oversampled_3.float(), device)

    return X_oversampled_0, X_oversampled_2, X_oversampled_3

def main():

    dataset_url =  'https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/abalone_csv.csv'
    download_url(dataset_url, '.')
    Abalone_df  = pd.read_csv('D:/Projects/Internship UNSW/abalone_csv.csv')

    #print(Abalone_df.Class_number_of_rings.size)
    option = int(input('Type the number of Abalone classes needed: '))
    if option == 2:

        Abalone_df = two_classes_Abalone(Abalone_df)

        label_encoder = LabelEncoder()
        onehot_encoder = OneHotEncoder(sparse=False) 
        Sex_labelencoded= label_encoder.fit_transform(Abalone_df['Sex']) 
        Sex_labelencoded = Sex_labelencoded.reshape(len(Sex_labelencoded), 1)
        Sex_onehotencoded = onehot_encoder.fit_transform(Sex_labelencoded)

        Abalone_df = Abalone_df.drop(['Sex'], axis=1)
        
        X_train, X_test = get_features(Abalone_df, Sex_onehotencoded, 0.2)
        y_train, y_test = get_labels(Abalone_df, 0.2)

        #### Calculating train and test accuracy and f1 score of non oversampled training data ####
        Normal_test_accuracy, Normal_train_accuracy, Normal_f1_score = test_model_lists(X_train, y_train.ravel(), X_test, y_test.ravel(), 30) 
        
        print("Before OverSampling, counts of label '0': {}".format(sum(y_train==0)))
        print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1))) 
        print("Before OverSampling, counts of label '2': {}".format(sum(y_train==2)))
        print("Before OverSampling, counts of label '3': {}".format(sum(y_train==3)))     

        X_train_SMOTE,y_train_SMOTE = SMOTE().fit_resample(X_train,y_train)

        print('After OverSampling, the shape of train_X: {}'.format(X_train_SMOTE.shape))
        print('After OverSampling, the shape of train_y: {} \n'.format(y_train_SMOTE.shape))

        print("After OverSampling, counts of label '0': {}".format(sum(y_train_SMOTE==0)))
        print("After OverSampling, counts of label '1': {}".format(sum(y_train_SMOTE==1))) 
        print("After OverSampling, counts of label '2': {}".format(sum(y_train_SMOTE==2))) 
        print("After OverSampling, counts of label '3': {}".format(sum(y_train_SMOTE==3))) 

        #### Calculating train and test accuracy and f1 score of SMOTE oversampled training data ####
        SMOTE_test_accuracy, SMOTE_train_accuracy, SMOTE_f1_score = test_model_lists(X_train_SMOTE, y_train_SMOTE.ravel(), X_test, y_test.ravel(), 30)

        device = get_default_device()
        #print(device)

        ##################### TWO CLASS ABALONE #####################
        ##### Oversampled data from SMOTE that is now to be passed in SMOTified GANs #####
        X_oversampled = X_train_SMOTE[(X_train.shape[0]):]
        X_oversampled = torch.from_numpy(X_oversampled)
        X_oversampled = to_device(X_oversampled.float(), device)

        #print(X_oversampled.shape)
        lr = 0.0002
        epochs = 150
        batch_size = 128

        X_real, y_real = GANs_two_class_real_data(X_train, y_train)   #Defining the real data to be put in GANs

        #Training our SMOTified GANs and GANs model and fetching their trained generators.
        generator_SG, generator_G = f1(X_train, y_train, X_train_SMOTE, y_train_SMOTE, X_real, y_real, X_oversampled, device, lr, epochs, batch_size, 1, 0)

        Trained_X_oversampled_SG = generator_SG(X_oversampled.float().to(device)).cpu().detach().numpy()
        Trained_SG_dataset = np.concatenate((X_train_SMOTE[:(X_train.shape[0])], Trained_X_oversampled_SG), axis=0)
        X_trained_SG, y_trained_SG = shuffle_in_unison(Trained_SG_dataset, y_train_SMOTE)

        #### Calculating train and test accuracy and f1 score of SMOTified GANs oversampled training data ####
        SG_test_accuracy, SG_train_accuracy, SG_f1_score = test_model_lists(X_trained_SG, y_trained_SG.ravel(), X_test, y_test.ravel(), 30)

        GANs_noise = torch.randn((X_oversampled.shape[0]), (X_oversampled.shape[1]), device=device)
        Trained_X_oversampled_G = generator_G(GANs_noise.float().to(device)).cpu().detach().numpy()
        Trained_G_dataset = np.concatenate((X_train_SMOTE[:(X_train.shape[0])], Trained_X_oversampled_G), axis=0)
        X_trained_G, y_trained_G = shuffle_in_unison(Trained_G_dataset, y_train_SMOTE)

        #### Calculating train and test accuracy and f1 score of SMOTified GANs oversampled training data ####
        G_test_accuracy, G_train_accuracy, G_f1_score = test_model_lists(X_trained_G, y_trained_G.ravel(), X_test, y_test.ravel(), 30)

        print(Normal_test_accuracy)
        print(Normal_train_accuracy)
        print(Normal_f1_score)
        print(SMOTE_test_accuracy)
        print(SMOTE_train_accuracy)
        print(SMOTE_f1_score)
        print(SG_test_accuracy)
        print(SG_train_accuracy)
        print(SG_f1_score)
        print(G_test_accuracy)
        print(G_train_accuracy)
        print(G_f1_score)
















    elif option == 4:

        Abalone_df = four_classes_Abalone(Abalone_df)

        label_encoder = LabelEncoder()
        onehot_encoder = OneHotEncoder(sparse=False) 
        Sex_labelencoded= label_encoder.fit_transform(Abalone_df['Sex']) 
        Sex_labelencoded = Sex_labelencoded.reshape(len(Sex_labelencoded), 1)
        Sex_onehotencoded = onehot_encoder.fit_transform(Sex_labelencoded)

        Abalone_df = Abalone_df.drop(['Sex'], axis=1)
        
        X_train, X_test = get_features(Abalone_df, Sex_onehotencoded, 0.2)
        y_train, y_test = get_labels(Abalone_df, 0.2)

        #### Calculating train and test accuracy and f1 score of non oversampled training data ####
        Normal_test_accuracy, Normal_train_accuracy, Normal_f1_score = test_model_lists(X_train, y_train.ravel(), X_test, y_test.ravel(), 30) 
        
        print("Before OverSampling, counts of label '0': {}".format(sum(y_train==0)))
        print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1))) 
        print("Before OverSampling, counts of label '2': {}".format(sum(y_train==2)))
        print("Before OverSampling, counts of label '3': {}".format(sum(y_train==3)))     

        X_train_SMOTE,y_train_SMOTE = SMOTE().fit_resample(X_train,y_train)

        print('After OverSampling, the shape of train_X: {}'.format(X_train_SMOTE.shape))
        print('After OverSampling, the shape of train_y: {} \n'.format(y_train_SMOTE.shape))

        print("After OverSampling, counts of label '0': {}".format(sum(y_train_SMOTE==0)))
        print("After OverSampling, counts of label '1': {}".format(sum(y_train_SMOTE==1))) 
        print("After OverSampling, counts of label '2': {}".format(sum(y_train_SMOTE==2))) 
        print("After OverSampling, counts of label '3': {}".format(sum(y_train_SMOTE==3))) 

        #### Calculating train and test accuracy and f1 score of SMOTE oversampled training data ####
        SMOTE_test_accuracy, SMOTE_train_accuracy, SMOTE_f1_score = test_model_lists(X_train_SMOTE, y_train_SMOTE.ravel(), X_test, y_test.ravel(), 30)

        device = get_default_device()
        #print(device)
        lr = 0.0002
        epochs = 150
        batch_size = 128

        ##################### FOUR CLASS ABALONE #####################

        ##### Oversampled data from SMOTE that is now to be passed in SMOTified GANs #####
        X_oversampled_0, X_oversampled_2, X_oversampled_3 = Oversampled_data(X_train_SMOTE, y_train_SMOTE, y_train, device)
        X_real_0, X_real_2, X_real_3, y_real_0, y_real_2, y_real_3 = GANs_four_class_real_data(X_train, y_train)

        generator_SG_0, generator_G_0 = f1(X_train, y_train, X_train_SMOTE, y_train_SMOTE, X_real_0, y_real_0, X_oversampled_0, device, lr, epochs, batch_size, 0, 1)
        generator_SG_2, generator_G_2 = f1(X_train, y_train, X_train_SMOTE, y_train_SMOTE, X_real_2, y_real_2, X_oversampled_2, device, lr, epochs, batch_size, 2, 1)
        generator_SG_3, generator_G_3 = f1(X_train, y_train, X_train_SMOTE, y_train_SMOTE, X_real_3, y_real_3, X_oversampled_3, device, lr, epochs, batch_size, 3, 1)

        Trained_X_oversampled_SG_0 = generator_SG_0(X_oversampled_0.float().to(device)).cpu().detach().numpy()
        Trained_X_oversampled_SG_2 = generator_SG_2(X_oversampled_2.float().to(device)).cpu().detach().numpy()
        Trained_X_oversampled_SG_3 = generator_SG_3(X_oversampled_3.float().to(device)).cpu().detach().numpy()

        Trained_SG_dataset_4_class = np.concatenate((X_train_SMOTE[:(X_train.shape[0])], Trained_X_oversampled_SG_0, Trained_X_oversampled_SG_2, Trained_X_oversampled_SG_3), axis=0)
        X_trained_SG_4_class, y_trained_SG_4_class = shuffle_in_unison(Trained_SG_dataset_4_class, y_train_SMOTE)

        #### Calculating train and test accuracy and f1 score of SMOTified GANs oversampled training data ####
        SG_test_accuracy_4_class, SG_train_accuracy_4_class, SG_f1_score_4_class = test_model_lists(X_trained_SG_4_class, y_trained_SG_4_class.ravel(), X_test, y_test.ravel(), 30)

        GANs_noise_0 = torch.randn((X_oversampled_0.shape[0]), (X_oversampled_0.shape[1]), device=device)
        GANs_noise_2 = torch.randn((X_oversampled_2.shape[0]), (X_oversampled_2.shape[1]), device=device)
        GANs_noise_3 = torch.randn((X_oversampled_3.shape[0]), (X_oversampled_3.shape[1]), device=device)

        Trained_X_oversampled_G_0 = generator_G_0(GANs_noise_0.float().to(device)).cpu().detach().numpy()
        Trained_X_oversampled_G_2 = generator_G_2(GANs_noise_2.float().to(device)).cpu().detach().numpy()
        Trained_X_oversampled_G_3 = generator_G_3(GANs_noise_3.float().to(device)).cpu().detach().numpy()

        Trained_G_dataset_4_class = np.concatenate((X_train_SMOTE[:(X_train.shape[0])], Trained_X_oversampled_G_0, Trained_X_oversampled_G_2, Trained_X_oversampled_G_3), axis=0)
        X_trained_G_4_class, y_trained_G_4_class = shuffle_in_unison(Trained_G_dataset_4_class, y_train_SMOTE)

        #### Calculating train and test accuracy and f1 score of GANs oversampled training data ####
        G_test_accuracy_4_class, G_train_accuracy_4_class, G_f1_score_4_class = test_model_lists(X_trained_G_4_class, y_trained_G_4_class.ravel(), X_test, y_test.ravel(), 30)  

        print(Normal_test_accuracy)
        print(Normal_train_accuracy)
        print(Normal_f1_score)
        print(SMOTE_test_accuracy)
        print(SMOTE_train_accuracy)
        print(SMOTE_f1_score)
        print(SG_test_accuracy_4_class)
        print(SG_train_accuracy_4_class)
        print(SG_f1_score_4_class)
        print(G_test_accuracy_4_class)
        print(G_train_accuracy_4_class)
        print(G_f1_score_4_class)
    


main()
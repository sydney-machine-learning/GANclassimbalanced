import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def two_classes_Abalone(Abalone_df):
  class_category = np.repeat("empty000", Abalone_df.shape[0])  
  for i in range(0, Abalone_df['Class_number_of_rings'].size):
    if(Abalone_df["Class_number_of_rings"][i] <= 7):
        class_category[i] = int(1)
    elif(Abalone_df["Class_number_of_rings"][i] > 7):
        class_category[i] = int(0)

  Abalone_df = Abalone_df.drop(['Class_number_of_rings'], axis=1)
  Abalone_df['class_category'] = class_category
  return Abalone_df


def four_classes_Abalone(Abalone_df):
  class_category = np.repeat("empty000", Abalone_df.shape[0])  
  for i in range(0, Abalone_df['Class_number_of_rings'].size):
    if(Abalone_df["Class_number_of_rings"][i] <= 7):
        class_category[i] = int(0)
    elif(Abalone_df["Class_number_of_rings"][i] > 7 and Abalone_df["Class_number_of_rings"][i] <= 10):
        class_category[i] = int(1)
    elif(Abalone_df["Class_number_of_rings"][i] > 10 and Abalone_df["Class_number_of_rings"][i] <= 15):
        class_category[i] = int(2)
    else:
        class_category[i] = int(3)

  Abalone_df = Abalone_df.drop(['Class_number_of_rings'], axis=1)
  Abalone_df['class_category'] = class_category
  return Abalone_df




def get_features(Abalone_df, Sex_onehotencoded, test_size):

    features = Abalone_df.iloc[:,np.r_[0:7]]
    X_train, X_test, X_gender, X_gender_test = train_test_split(features, Sex_onehotencoded, random_state=10, test_size=test_size)
    X_train = np.concatenate((X_train.values, X_gender), axis=1)
    X_test = np.concatenate((X_test.values, X_gender_test), axis=1)
    return X_train, X_test




def get_labels(Abalone_df, test_size):
    labels = Abalone_df.iloc[:,7]
    y_train, y_test = train_test_split(labels, random_state=10, test_size=test_size)
    train_list = [int(i) for i in y_train.ravel()] 
    y_train=np.array(train_list)
    test_list = [int(i) for i in y_test.ravel()]    #Flattening the matrix
    y_test=np.array(test_list)

    return y_train, y_test



def GANs_two_class_real_data(X_train, y_train):   #Defining the real data for GANs
  X_real = []
  y_train = y_train.ravel()
  for i in range(len(y_train)):
    if int(y_train[i])==1:
      X_real.append(X_train[i])
  X_real = np.array(X_real)
  y_real = np.ones((X_real.shape[0],))
  return X_real, y_real


def GANs_four_class_real_data(X_train, y_train):
    X_real_0 = []
    X_real_2 = []
    X_real_3 = []
    for i in range(len(y_train)):
        if int(y_train[i])==0:
            X_real_0.append(X_train[i])
        if int(y_train[i])==2:
            X_real_2.append(X_train[i])
        if int(y_train[i])==3:
            X_real_3.append(X_train[i])
    X_real_0 = np.array(X_real_0)
    X_real_2 = np.array(X_real_2)
    X_real_3 = np.array(X_real_3)

    y_real_0 = np.full((X_real_0.shape[0],), 0)
    y_real_2 = np.full((X_real_2.shape[0],), 2)
    y_real_3 = np.full((X_real_3.shape[0],), 3)

    return X_real_0, X_real_2, X_real_3, y_real_0, y_real_2, y_real_3

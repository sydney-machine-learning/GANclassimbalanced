import torch
from torch.utils.data import TensorDataset, DataLoader
from choose_device import get_default_device, to_device, DeviceDataLoader
from GANs_model import GANs_Discriminator, GANs_Generator
from model_fit import SG_fit, G_fit
#This file is used in train.py file


def f1(X_train, y_train, X_train_SMOTE, y_train_SMOTE, X_real, y_real, X_oversampled, device, lr, epochs, batch_size, minority_class, majority_class):   #Fetches us the trained generators
    # X_oversampled = X_train_SMOTE[(X_train.shape[0]):]
    # X_oversampled = torch.from_numpy(X_oversampled)
    # X_oversampled = to_device(X_oversampled.float(), device)

    #print(X_oversampled.shape)

    #X_real, y_real = GANs_two_class_real_data(X_train, y_train)

    ##### Wrapping all the tensors in a Tensor Dataset. #####
    # tensor_x = torch.Tensor(X_real) 
    # tensor_y = torch.Tensor(y_real)
    my_dataset = TensorDataset(torch.Tensor(X_real) ,torch.Tensor(y_real))

    # lr = 0.0002
    # epochs = 150
    # batch_size = 128

    ##### Loading our Tensor Dataset into a Dataloader. #####
    train_dl = DataLoader(my_dataset, batch_size=batch_size, shuffle=True) 
    train_dl = DeviceDataLoader(train_dl, device)
    
    ##### Initialising the generator and discriminator objects ######
    gen1 = GANs_Generator(X_train.shape[1], X_train.shape[1], 128)
    disc1 = GANs_Discriminator(X_train.shape[1], 128)

    gen2 = GANs_Generator(X_train.shape[1], X_train.shape[1], 128)
    disc2 = GANs_Discriminator(X_train.shape[1], 128)

    ##### Loading the model in GPU #####
    generator_SG = to_device(gen1.generator, device)      
    discriminator_SG = to_device(disc1.discriminator, device)

    generator_G = to_device(gen2.generator, device)      
    discriminator_G = to_device(disc2.discriminator, device)

    SG_fit_func = SG_fit(epochs, lr, discriminator_SG, generator_SG, X_oversampled, train_dl, device, minority_class, majority_class)   #fit function object initiated.
    history1 = SG_fit_func()    #Callable object

    G_fit_func = G_fit(epochs, lr, discriminator_G, generator_G, train_dl, device, minority_class, majority_class)
    history2 = G_fit_func()

    return generator_SG, generator_SG
    
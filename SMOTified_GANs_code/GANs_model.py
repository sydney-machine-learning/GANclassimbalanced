import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def get_generator_block(input_dim, output_dim):      #Generator Block
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True),
    )
class GANs_Generator(nn.Module):     #Generator Model

    def __init__(self, z_dim, im_dim, hidden_dim):
        super(GANs_Generator, self).__init__()
        self.generator = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()
        )
    def forward(self, noise):
        return self.generator(noise)    
   
    def get_generator(self):
        return self.generator

def get_discriminator_block(input_dim, output_dim):       #Discriminator Block
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(0.2, inplace=True)        
    )
class GANs_Discriminator(nn.Module):         #Discriminator Model
    def __init__(self, im_dim, hidden_dim):
        super(GANs_Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image):
        return self.discriminator(image)
    
    def get_disc(self):
        return self.discriminator
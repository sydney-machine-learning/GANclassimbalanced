import torch
from tqdm.auto import tqdm
from model_trainer import train_discriminator, train_generator
#This file is used in fit.py file




class SG_fit():
    def __init__(self, epochs, lr, discriminator, generator, X_oversampled, train_dl, device, minority_class, majority_class, start_idx=1):
        self.epochs = epochs
        self.lr = lr
        self.discriminator = discriminator
        self.generator = generator
        self.X_oversampled = X_oversampled
        self.train_dl = train_dl
        self.device = device
        self.minority_class = minority_class
        self.majority_class = majority_class

    def __call__(self):
        torch.cuda.empty_cache()
    
        # Losses & scores
        losses_g = []
        losses_d = []
        real_scores = []
        fake_scores = []
        
        # Create optimizers
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        
        for epoch in range(self.epochs):
            for real_data, _ in tqdm(self.train_dl):
                # Train discriminator
                train_disc = train_discriminator(real_data, self.X_oversampled, opt_d, self.generator, self.discriminator, self.device, self.minority_class, self.majority_class)
                loss_d, real_score, fake_score = train_disc()
                # Train generator
                train_gen = train_generator(self.X_oversampled, opt_g, self.generator, self.discriminator, self.device, self.minority_class)
                loss_g = train_gen()
                
            # Record losses & scores
            losses_g.append(loss_g)
            losses_d.append(loss_d)
            real_scores.append(real_score)
            fake_scores.append(fake_score)
            
            # Log losses & scores (last batch)
            print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
                epoch+1, self.epochs, loss_g, loss_d, real_score, fake_score))
        
            # Save generated images
            #save_samples(epoch+start_idx, fixed_latent, show=False)
        
        return losses_g, losses_d, real_scores, fake_scores





class G_fit():
    def __init__(self, epochs, lr, discriminator, generator, train_dl, device, minority_class, majority_class, start_idx=1):
        self.epochs = epochs
        self.lr = lr
        self.discriminator = discriminator
        self.generator = generator
        self.train_dl = train_dl
        self.device = device   
        self.minority_class = minority_class
        self.majority_class = majority_class

    def __call__(self):
        torch.cuda.empty_cache()
    
        # Losses & scores
        losses_g = []
        losses_d = []
        real_scores = []
        fake_scores = []
        
        # Create optimizers
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        
        for epoch in range(self.epochs):
            for real_data, _ in tqdm(self.train_dl):
                # Train discriminator
                latent_data = torch.randn(real_data.shape[0], real_data.shape[1], device=self.device)
                train_disc = train_discriminator(real_data, latent_data, opt_d, self.generator, self.discriminator, self.device, self.minority_class, self.majority_class)
                loss_d, real_score, fake_score = train_disc()                
                # Train generator
                train_gen = train_generator(latent_data, opt_g, self.generator, self.discriminator, self.device, self.minority_class)
                loss_g = train_gen()
                
            # Record losses & scores
            losses_g.append(loss_g)
            losses_d.append(loss_d)
            real_scores.append(real_score)
            fake_scores.append(fake_score)
            
            # Log losses & scores (last batch)
            print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
                epoch+1, self.epochs, loss_g, loss_d, real_score, fake_score))
        
            # Save generated images
            #save_samples(epoch+start_idx, fixed_latent, show=False)
        
        return losses_g, losses_d, real_scores, fake_scores
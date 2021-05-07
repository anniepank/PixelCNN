import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import pickle


class LabelToImageNet(nn.Module):
    def __init__(self, output_size=28*28):
        super().__init__()
        
        self.hidden_1 = nn.Linear(1, 20)
        self.hidden_2 = nn.Linear(20, 10)
        self.hidden_3 = nn.Linear(10, 10)
        self.hidden_4 = nn.Linear(10, 250)
        self.hidden_5 = nn.Linear(250, 500)
        self.hidden_6 = nn.Linear(500, 28*28)
        #self.hidden_3 = nn.Linear(500, 250)
        #self.hidden_4 = nn.Linear(250, 28*28)
        #self.hidden_5 = nn.Linear(*16, 28*28)        
    
    def forward(self, x):
        #x = torch.Tensor([[int(i == z) for i in range(10)] for z in x]).to(device).float()
        x = self.hidden_1(x)
        x =F.relu(x)
        x = self.hidden_2(x)
        x = F.relu(x)
        x = self.hidden_3(x)
        x = F.relu(x)
        x = self.hidden_4(x)
        x = F.relu(x)
        x = self.hidden_5(x)
        x = F.relu(x)
        x = self.hidden_6(x)
        #x = F.relu(x)
        x = torch.sigmoid(x)
        return x


class SimpleEncoder(nn.Module):
    def __init__(self, latent_space_size):
        super(SimpleEncoder, self).__init__()
        self.hidden_1 = nn.Linear(28 * 28, 512)
        self.hidden_2 = nn.Linear(512, 256)
        self.hidden_3 = nn.Linear(256, 128)

        self.mean = nn.Linear(128, latent_space_size)
        self.log_std = nn.Linear(128, latent_space_size)

    def forward(self, x):
        x = self.hidden_1(x)
        x = F.relu(x)
        x = self.hidden_2(x)
        x = F.relu(x)
        x = self.hidden_3(x)
        x = F.relu(x)

        return self.mean(x), self.log_std(x)

class SimpleDecoder(nn.Module):
    def __init__(self, latent_space_size):
        super(SimpleDecoder, self).__init__()
        self.hidden_1 = nn.Linear(latent_space_size, 128)
        self.hidden_2 = nn.Linear(128, 256)
        self.hidden_3 = nn.Linear(256, 512)
        self.result = nn.Linear(512, 28 * 28)

    def forward(self, x):
        x = self.hidden_1(x)
        x = F.relu(x)
        x = self.hidden_2(x)
        x = F.relu(x)
        x = self.hidden_3(x)
        x = F.relu(x)
        return torch.sigmoid(self.result(x))#, self.log_std(x)


class Encoder(nn.Module):
    def __init__(self, latent_space_size):
        super(Encoder, self).__init__()
        self.hidden_1 = nn.Linear(28 * 28, 500)
        self.hidden_2 = nn.Linear(500, 250)
        self.hidden_3 = nn.Linear(250, 10)
        self.hidden_4 = nn.Linear(10, 10)
        self.hidden_5 = nn.Linear(10, 20)
        self.hidden_6 = nn.Linear(20, 10)

        self.mean = nn.Linear(10, latent_space_size)
        self.log_std = nn.Linear(10, latent_space_size)

    def forward(self, x):
        x = self.hidden_1(x)
        x = F.relu(x)
        x = self.hidden_2(x)
        x = F.relu(x)
        x = self.hidden_3(x)
        x = F.relu(x)
        x = self.hidden_4(x)
        x = F.relu(x)
        x = self.hidden_5(x)
        x = F.relu(x)
        x = self.hidden_6(x)
        x = F.relu(x)

        return self.mean(x), self.log_std(x)


class Decoder(nn.Module):
    def __init__(self, latent_space_size):
        super(Decoder, self).__init__()
        self.hidden_1 = nn.Linear(latent_space_size, 10)
        self.hidden_2 = nn.Linear(10, 20)
        self.hidden_3 = nn.Linear(20, 10)
        self.hidden_4 = nn.Linear(10, 10)
        self.hidden_5 = nn.Linear(10, 250)
        self.hidden_6 = nn.Linear(250, 500)
        self.result = nn.Linear(500, 28 * 28)
        # self.mean = nn.Linear(500, 28 * 28)

        # self.log_std = nn.Linear(500, 28 * 28)

    def forward(self, x):
        x = self.hidden_1(x)
        x = F.relu(x)
        x = self.hidden_2(x)
        x = F.relu(x)
        x = self.hidden_3(x)
        x = F.relu(x)
        x = self.hidden_4(x)
        x = F.relu(x)
        x = self.hidden_5(x)
        x = F.relu(x)
        x = self.hidden_6(x)
        x = F.relu(x)

        return torch.sigmoid(self.result(x))#, self.log_std(x)


class VAE(nn.Module):
    def __init__(self, n_inputs, latent_space_size, simple=False):
        super(VAE, self).__init__()
        self.latent_space_size = latent_space_size
        if simple:
            self.encoder = SimpleEncoder(latent_space_size)
            self.decoder = SimpleDecoder(latent_space_size)
        else:
            self.encoder = Encoder(latent_space_size)
            self.decoder = Decoder(latent_space_size)
    
    def sample_latent(self, m_z, log_std_z):
        if self.training:
            #print('in training')
            eps = torch.randn_like(m_z)
            sample = eps * log_std_z.exp() + m_z
            return sample
        else:
            return m_z

    def forward(self, x):
        m_z, log_std_z = self.encoder(x)
        z = self.sample_latent(m_z, log_std_z)
        # mu_x, log_std_x = self.decoder(z)
        reconstruction = self.decoder(z)
        
        #return m_z, log_std_z, mu_x, log_std_x
        return m_z, log_std_z, reconstruction

    def sample(self, n, noise=True):
        with torch.no_grad():
            z = torch.randn(n, self.latent_space_size)
            mu, log_std = self.decoder(z)
            if noise:
                z = torch.randn_like(mu) * log_std.exp() + mu
            else:
                z = mu
        return z.cpu().numpy()


def vae_loss(image, m_z, log_std_z, reconstruction):
    BCE = F.binary_cross_entropy(input=reconstruction.view(-1, 28 * 28), target=image.view(-1, 28 * 28), reduction='sum').sum()
    KL_divergence = -(0.5 * (1 + log_std_z - m_z.pow(2) - torch.exp(2 * log_std_z))).sum(1).sum()

    return (BCE + KL_divergence, BCE, KL_divergence)



def get_sequential_model():
    return nn.Sequential(
        nn.Linear(1, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 250),
        nn.ReLU(),
        nn.Linear(250, 500),
        nn.ReLU(),
        nn.Linear(500, 28*28),
        nn.Sigmoid()
    )


def sample_from_autoencoder(model, n_classes, device):
    outputs = []
    for i in range(n_classes):
        print()
        outputs.append(
            torch.reshape(
                model(torch.tensor([i]).to(device).float()),
                (1, 28, 28)
                )
            )
    return torch.stack(outputs, dim=0)


def crossentropy_for_gray(output, target):
    return torch.sum(-target * torch.log(output) - (1 - target) * torch.log(1 - output)) / output.size()[0]


def sample_images_from_vae(model, n, device, noise=False):
    plt.figure(figsize=(8, 8), dpi=80)

    with torch.no_grad():
        z = torch.randn(n, 2).to(device)
        reconstructed = model.decoder(z)
        if noise:
            z = torch.randn_like(mu) * log_std.exp() + mu
        else:
            z = reconstructed
    imgs = (torch.reshape(z, (-1, 1, 28, 28)) * 255)

    nrows = 10
    grid = torchvision.utils.make_grid(imgs, nrows)
    grid = grid.detach().cpu().numpy()[1, :, :]  # from CHW to HWC
    plt.imshow(1 - grid, cmap='Greys')


def display_latents(vae, val_loader, device):
    color_mapping = {
        0: 'red',
        1: 'blue',
        2: 'green',
        3: 'yellow',
        4: 'pink',
        5: 'blueviolet',
        6: 'black',
        7: 'grey',
        8: 'salmon',
        9: 'peru'
    }

    scatter = {color:[[], []] for color in color_mapping.values()}

    for sample in val_loader:
        img = sample[0]
        labels = sample[1].detach().cpu().numpy()
        img = torch.reshape(img, (-1, 28 * 28)).to(device).float()
        
        mu_z, log_std_z = vae.encoder(img)
        z = (torch.randn_like(mu_z) * log_std_z.exp() + mu_z).detach().cpu().numpy()
        for i in range(z.shape[0]):
            plt.scatter([z[i][0]], [z[i][1]], color=color_mapping[labels[i]],s=8)
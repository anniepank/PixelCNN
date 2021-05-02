# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
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


# %%
writer = SummaryWriter()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


# %%
def to_binary(x):
    zero_mask = x < 0.5
    x[zero_mask] = 0.
    x[~zero_mask] = 1.
    return x

train_dataset_gray = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor()
]))

test_dataset_gray = datasets.MNIST('./data', train=False, transform=transforms.Compose([
    transforms.ToTensor()
]))

train_dataset_binary = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    to_binary
]))

test_dataset_binary = datasets.MNIST('./data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    to_binary
]))


N_val = 1000
N_train = len(train_dataset_binary) - N_val
N_test = len(test_dataset_binary)


# %%
def make_image_grid(images, nrow=8):
    images = images.float()  # from {0,1,2,3} ints to [0,1] floats
    grid = torchvision.utils.make_grid(images, nrow)
    grid = grid.permute(1, 2, 0).cpu().numpy()  # from CHW to HWC
    return grid

plt.figure(figsize=(10,10))
plt.title('Samples from gray dataset')
it = iter(train_dataset_gray)

imgs = []
labels = []
for i in range(100):
    element = next(it)
    imgs.append(element[0])
    labels.append(element[1])

labels = np.array(labels).reshape((10, 10))
    
plt.imshow(make_image_grid(torch.stack([next(it)[0] for _ in range(100)], dim=0), nrow=10))
del it


# %%
plt.figure(figsize=(10,10))
plt.title('Samples from binary dataset')
it = iter(train_dataset_binary)

imgs = []
labels = []
for i in range(100):
    element = next(it)
    imgs.append(element[0])
    labels.append(element[1])

labels = np.array(labels).reshape((10, 10))
    
plt.imshow(make_image_grid(torch.stack(imgs, dim=0), nrow=10))
print(imgs[0].size())
del it


# %%
class LabelToImageNet(nn.Module):
    
    def __init__(self, input_size=1, output_size=28*28):
        super().__init__()
        
        self.hidden_1 = nn.Linear(input_size, 20)
        self.hidden_2 = nn.Linear(20, 10)
        self.hidden_3 = nn.Linear(10, 10)
        self.hidden_4 = nn.Linear(10, 250)
        self.hidden_5 = nn.Linear(250, 500)
        self.hidden_6 = nn.Linear(500, 28*28)
        #self.hidden_3 = nn.Linear(500, 250)
        #self.hidden_4 = nn.Linear(250, 28*28)
        #self.hidden_5 = nn.Linear(*16, 28*28)        
    
    def forward(self, x):
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


# %%
lr = 1e-3
reg = 1e-4
batch_size = 256
hidden_size = 64  # h variable in PixelCNN
num_residual_blocks = 12
num_feature_maps = 128  # size of the hidden output layer
batch_norm = True
num_input_channels = 1
num_output_classes = 2
flatten_img_size = 28 * 28


def sample_mnist_pixel_cnn(model, n_classes):
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


# %%
input_size = 1
model = LabelToImageNet(input_size=input_size).to(device)
# alternatively model = torch.nn.Sequential( ?? )

optimizer = optim.Adam(model.parameters(), lr, weight_decay=reg)
criterion = nn.CrossEntropyLoss()

train_loader = torch.utils.data.DataLoader(train_dataset_binary, batch_size=batch_size,
                                           sampler=SubsetRandomSampler(list(range(N_train))))
val_loader = torch.utils.data.DataLoader(train_dataset_binary, batch_size=batch_size,
                                         sampler=SubsetRandomSampler(list(range(N_train, N_train + N_val))))

train_nll_history = []
val_nll_history = []

num_epochs = 100
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs} ', end='')
    
    model.train()
    for i, minibatch in enumerate(train_loader):
        optimizer.zero_grad()
        labels = torch.reshape(minibatch[1].to(device), (-1, input_size)).float()
        imgs = torch.flatten(torch.reshape(minibatch[0].to(device), (-1, flatten_img_size)))

        output = model(labels)

        outputs = torch.stack(
            [torch.flatten(1-output), torch.flatten(output)], dim=1
            )#.reshape(1, -1, num_output_classes)
        
        #nll = nn.MSELoss()(torch.flatten(output), imgs.float())
        nll = criterion(outputs, imgs.long())
        nll.backward()
        optimizer.step()
        
        train_nll_history.append(nll.item() / np.log(2.) / 2.)
        
        if i % 50 == 0:
            print('.', end='')
    
    train_loss = np.mean(train_nll_history) / np.log(2.) / 2.
    writer.add_scalar("Loss/train", train_loss, epoch)


    # compute nll on validation set
    val_nlls = []
    model.eval()
    with torch.no_grad():
        for val_minibatch in val_loader:
            val_labels = torch.reshape(minibatch[1].to(device), (-1, input_size)).float()
            val_imgs = torch.flatten(torch.reshape(minibatch[0].to(device), (-1, flatten_img_size)))

            output = model(val_labels)
            val_outputs = torch.stack(
                [torch.flatten(1-output), torch.flatten(output)], dim=1
            )#.reshape(-1, num_output_classes)
        
            #val_nll = nn.MSELoss()(torch.flatten(output), val_imgs.float())
            val_nll = criterion(val_outputs, imgs.long())
            val_nlls.append(val_nll.item())
    val_loss = np.mean(val_nlls) / np.log(2.) / 2.

    writer.add_scalar("Loss/val", val_loss, epoch)
    val_nll_history.append(val_loss)
    print("loss: ", val_loss)
    
    samples = sample_mnist_pixel_cnn(model, 10).detach()
    plt.figure(figsize=(6,3))
    img = make_image_grid(samples, nrow=5)
    plt.imsave(f'./data/epoch{epoch}.jpg', img)
    #plt.show()
    

writer.flush()


# %%
img = torch.reshape(model(torch.tensor([1]).to(device).float()),(1, 28, 28)) 
img


# %%




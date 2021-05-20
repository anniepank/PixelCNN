import numpy as np   
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from torch.distributions import Categorical
from torchvision import datasets, transforms

device = torch.device('cuda')
print(device)

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())

print(len(train_dataset), len(test_dataset))

x_plot, y_plot = zip(*torch.utils.data.Subset(test_dataset, range(16)))
x_plot, y_plot = torch.stack(x_plot), torch.as_tensor(y_plot, device=device)
x_plot = x_plot.to(device)

class VQ(nn.Module):
    
    def __init__(self, num_embeddings, embedding_size, commitment_cost):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.embedding.weight.data.uniform_(-1. / num_embeddings, 1. / num_embeddings)
    
    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()  # from BCHW to BHWC
        x_flat = x.view(-1, self.embedding_size)
        
        w = self.embedding.weight
        distances = torch.sum(x_flat ** 2, dim=1, keepdim=True) + torch.sum(w ** 2, dim=1) - 2 * (x_flat @ w.T)

        indices_flat = torch.argmin(distances, dim=1, keepdim=True)
        quantized_flat = self.embed(indices_flat)
        
        quantized = quantized_flat.view(x.shape)
        indices = indices_flat.view(*x.shape[:3]).unsqueeze(dim=1)  # BHW to BCHW
        
        # To implemtent the stop gradients and pass through of gradients you have to put detach() at the right positions
        if self.training:

            e_latent_loss = F.mse_loss(quantized.detach(), x)
            q_latent_loss = F.mse_loss(quantized, x.detach())

            loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
            quantized = x + (quantized - x).detach()
        else:
            loss = 0.
        
        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # from BHWC to BCHW
        
        return quantized, indices, loss
    
    def embed(self, indices):
        quantized = self.embedding(indices)
        return quantized

class ResBlock(nn.Module):
    
    def __init__(self, in_channels, hidden_channels, stride=1, padding_1=1, padding_2=0):
        super().__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=stride, padding=padding_1, bias=False)
        self.conv2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, stride=stride, padding=padding_2, bias=False)

    def forward(self, x):
        input_x = x
        x = self.conv1(F.relu(x))
        x = self.conv2(F.relu(x))

        return x + input_x

class VQVAE(nn.Module):
    
    def __init__(self, in_channels, num_embeddings, embedding_size, res_hidden_channels, decoder_bn,
                 commitment_cost, res_stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        
        h = embedding_size
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, h, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(h),
            nn.ReLU(inplace=True),
            nn.Conv2d(h, h, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(h),
            ResBlock(h, res_hidden_channels, stride=res_stride),
            nn.BatchNorm2d(h),
            ResBlock(h, res_hidden_channels, stride=1),
            nn.BatchNorm2d(h)
        )
        
        self.vq = VQ(num_embeddings, embedding_size, commitment_cost)
        
        if decoder_bn:
            self.decoder = nn.Sequential(
                ResBlock(h, res_hidden_channels, stride=1),
                nn.BatchNorm2d(h),
                ResBlock(h, res_hidden_channels, stride=res_stride),
                nn.BatchNorm2d(h),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(h, h, kernel_size=4, stride=1, padding=0),
                nn.BatchNorm2d(h),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(h, in_channels, kernel_size=4, stride=2, padding=1)
            )
        else:
            self.decoder = nn.Sequential(
                ResBlock(h, res_hidden_channels, stride=1),
                ResBlock(h, res_hidden_channels, stride=res_stride),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(h, h, kernel_size=4, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(h, in_channels, kernel_size=4, stride=2, padding=1)
            )
    
    def forward(self, x):
        z = self.encode(x)
        quantized, indices, vq_loss = self.quantize(z)
        x_recon = self.decode(quantized)
        return x_recon, quantized, indices, vq_loss
    
    def encode(self, x):
        z = self.encoder(x)
        return z
    
    def quantize(self, z):
        quantized, indices, vq_loss = self.vq(z)
        return quantized, indices, vq_loss
    
    def decode(self, quantized):
        x_recon = self.decoder(quantized)
        return x_recon
    
    def embed(self, indices):
        return self.vq.embed(indices)

def plot_latents(model):
    model.eval()
    with torch.no_grad():
        x_recon, quantized, indices, _ = model(x_plot)
    
    fig, axes = plt.subplots(1, 3, figsize=(16,4))
    
    axes[0].set_title('Original images')
    axes[0].imshow(torchvision.utils.make_grid(1 - x_plot, nrow=4, pad_value=1).permute(1, 2, 0).cpu().numpy())
    axes[0].axis('off')
    
    axes[1].set_title('Latent representation')
    axes[1].imshow(torchvision.utils.make_grid(
        indices.float() / (model.num_embeddings - 1),
        nrow=4, pad_value=1).permute(1, 2, 0).cpu().numpy())
    axes[1].axis('off')
    
    axes[2].set_title('Reconstructed images')
    axes[2].imshow(torchvision.utils.make_grid(
        torch.clamp(1 - x_recon, 0., 1.), nrow=4, pad_value=1).permute(1, 2, 0).cpu().numpy())
    axes[2].axis('off')

def train_vqvae(num_embeddings, embedding_size, res_hidden_channels, decoder_bn,
                commitment_cost, num_epochs, batch_size, lr, plot_final_only=False, res_stride=1, model=None):
    
    if model is None:
        model = VQVAE(1, num_embeddings, embedding_size, res_hidden_channels, decoder_bn, commitment_cost, res_stride=res_stride)
        model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr)
    
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    shown = 0
    hist = []
    for epoch in range(num_epochs):
        model.train()
        epoch_hist = []
        for x, y in dataloader:
            x = x.to(device)
            
            optimizer.zero_grad()
            x_recon, quantized, encoding_indices, vq_loss = model(x)
            
            if shown == 0:
                print("Quantized: ", quantized.size())
                print("Indices: ", encoding_indices.size())
                shown = 1
            
            recon_loss = F.mse_loss(x_recon, x)
            loss = vq_loss + recon_loss
            loss.backward()
            optimizer.step()
            
            epoch_hist.append((loss.item(), recon_loss.item(), vq_loss.item()))
        
        losses, recon_losses, vq_losses = zip(*epoch_hist)
        hist.append((np.mean(losses), np.mean(recon_losses), np.mean(vq_losses)))
        
        if not plot_final_only and (epoch % 5 == 0 or epoch == num_epochs - 1):
            print(f'Epoch {epoch + 1}')
            plot_latents(model)
            plt.savefig('/home/syslink/Documents/gm/ex4/pic.png')
    
    if plot_final_only:
        plot_latents(model)
        plt.savefig('/home/syslink/Documents/gm/ex4/latents.png')

    
    xs = np.arange(1, num_epochs + 1)
    losses, recon_losses, vq_losses = zip(*hist)
    
    plt.title('Loss history')
    plt.xticks(xs)
    plt.plot(xs, losses, marker='o', label='total')
    plt.plot(xs, recon_losses, marker='o', label='reconstruction')
    plt.plot(xs, vq_losses, marker='o', label='vq')
    plt.legend()
    
    return model


class MaskedConv(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv, self).__init__(*args, **kwargs)
        # use two type of masks. Type A for the first conv layer where center pixel we want to predict is masked on input. Type B for susequent conv2d-layers where center pixel is shown
        self.mask_type = mask_type   
        # use register_buffer so the mask is also transferred to cuda (GPU)
        self.register_buffer('mask', self.weight.data.clone())  # https://discuss.pytorch.org/t/copy-deepcopy-vs-clone/55022

        in_channels, out_channels, height, width = self.weight.size()

        self.mask.fill_(1)
        
        y_center, x_center = height // 2, width // 2

        if mask_type =='A':
            # center pixel masked
            self.mask[:, :, y_center + 1:, :] = 0.0
            self.mask[:, :, y_center:, x_center:] = 0.0
        else: 
            # center pixel not masked
            self.mask[:, :, y_center + 1:, :] = 0.0
            self.mask[:, :, y_center:, x_center + 1:] = 0.0

        
    def forward(self, x):
        self.weight.data *= self.mask.to(device) 
        return super(MaskedConv, self).forward(x)

embedding_size = 8
num_embeddings = 4

class PixelCNN(nn.Module):
    def __init__(self, cfg):
        super(PixelCNN, self).__init__()
        
        self.kernel_size = cfg.kernel_size
        self.hidden_layers = cfg.hidden_layers
        n_channels = cfg.channel_dimention
        
        self.conv2d_A = MaskedConv('A', 1, n_channels, self.kernel_size, 1, self.kernel_size // 2, bias=False)
        self.BatchNorm2d = nn.BatchNorm2d(n_channels)
        self.relu_1 = nn.ReLU(True)

        self.convs2d_B  = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.relus = nn.ModuleList()
        for i in range(self.hidden_layers): 
            self.convs2d_B.append(MaskedConv('B', n_channels, n_channels, self.kernel_size, 1, self.kernel_size // 2, bias=False))
            self.batch_norms.append(nn.BatchNorm2d(n_channels))
            self.relus.append(nn.ReLU(True))
            
        self.out = nn.Conv2d(n_channels, num_embeddings, 1)
    
    def forward(self, x):
        x = self.conv2d_A(x)
        x = self.BatchNorm2d(x)
        x = self.relu_1(x)

        for i in range(self.hidden_layers):
            x = self.convs2d_B[i](x)
            x = self.batch_norms[i](x)
            x = self.relus[i](x)

        x = self.out(x)
        return x

compressed_image_size = 11

def sample_pixel_cnn(model, n, num_input_channels):
    model.eval()
    with torch.no_grad():
        samples = torch.zeros((n, num_input_channels, compressed_image_size, compressed_image_size), dtype=torch.float, device=device)
        for i in range(compressed_image_size):
            for j in range(compressed_image_size):
                for k in range(num_input_channels):
                    output = model(samples)
                    probs = F.softmax(output[:, :, i, j], dim=-1).data
                    samples[:, :, i, j] = torch.multinomial(probs, 1).float()
        return samples


class Config():
    def __init__(self):
        self.epochs = 20
        self.batch_size = 32
        self.kernel_size = 7
        self.channel_dimention = 64
        self.hidden_layers = 6
        self.learning_rate = 1e-4
        self.reg = 1e-4




vqvae = train_vqvae(num_embeddings=num_embeddings, embedding_size=embedding_size, res_hidden_channels=32, decoder_bn=True,
            commitment_cost=0.2, num_epochs=20, batch_size=32, lr=1e-3, plot_final_only=False)

torch.save(vqvae.state_dict(),'/home/syslink/Documents/gm/ex4/vqvae_big.pt')

vqvae = VQVAE(1, num_embeddings, embedding_size, 32, True, 0.2, res_stride=1).to(device)
vqvae.load_state_dict(torch.load('/home/syslink/Documents/gm/ex4/vqvae_big.pt'))
vqvae.eval()

n_classes = num_embeddings
cfg = Config()

#pixelCNN = PixelCNN(cfg=cfg).to(device)
#pixelCNN.load_state_dict(torch.load('/home/syslink/Documents/gm/ex4/pixelcnn_big.pt'))
#pixelCNN.eval()


pixelCNN = PixelCNN(cfg=cfg).to(device)

optimizer = optim.Adam(pixelCNN.parameters(), cfg.learning_rate, weight_decay=cfg.reg)
criterion = nn.CrossEntropyLoss()

train_nll_history = []
val_nll_history = []

num_epochs = cfg.epochs

train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

for epoch in range(num_epochs):
    print(f'PixelCNN epoch {epoch + 1}/{num_epochs} ', end='')
    
    pixelCNN.train()
    train_loss = 0
    n_batches = 0

    for i, minibatch in enumerate(train_loader):
        optimizer.zero_grad()

        x = minibatch[0].to(device)

        x_recon, quantized, indices, _ = vqvae(x)

        inputs = indices
        targets = Variable(inputs[:, 0, :, :])
        #targets *= n_classes - 1
        targets = targets.long()

        inputs = inputs.to(device).float()
        targets = targets.to(device)

        outputs = pixelCNN(inputs)

        nll = criterion(outputs, targets)
        nll.backward()

        optimizer.step()
        
        train_loss += nll.item() / np.log(2.) / 2.
        
        if i % 50 == 0:
            print('.', end='')

        n_batches += 1
        #train_nll_history.append(train_loss)

    train_loss /= n_batches
    train_nll_history.append(train_loss)

    print("Train loss: ", train_loss)
    
    pixelCNN.eval()


plt.plot(train_nll_history)
plt.savefig('/home/syslink/Documents/gm/ex4/train_pixel_cnn.png')
torch.save(pixelCNN.state_dict(),'/home/syslink/Documents/gm/ex4/pixelcnn_big.pt')



class PixelCNNVQVAE(nn.Module):
    
    def __init__(self, pixelcnn, vqvae, image_size):
        super().__init__()
        self.pixelcnn = pixelcnn
        self.vqvae = vqvae
        self.image_size = image_size
    
    def sample_prior(self, batch_size):
        indices = sample_pixel_cnn(self.pixelcnn, batch_size, 1).long()
        indices = indices.permute(0, 2, 3, 1).contiguous()  # from BCHW to BHWC
        indices_flat = indices.view(-1, 1)
        indices = indices_flat.view(*[batch_size, self.image_size, self.image_size]).unsqueeze(dim=1)
        with torch.no_grad():
            quantized = self.vqvae.embed(indices_flat)
        
        return quantized, indices
    
    def sample(self, batch_size):
        quantized_flat, indices = self.sample_prior(batch_size)
        quantized = quantized_flat.permute(0, 2, 1).view(-1, quantized_flat.size()[-1], self.image_size, self.image_size)
        with torch.no_grad():
            x_recon = self.vqvae.decode(quantized)
        return x_recon, quantized, indices

model = PixelCNNVQVAE(pixelCNN, vqvae, compressed_image_size)

#x_recon, quantized, indices = model.sample(1)


def plot_latents_with_pixelcnn(model):
    model.eval()
    with torch.no_grad():
        samples = []
        indices = []
        for _ in range(64):
            sample, quantized, index = model.sample(1)
            samples.append(sample)
            indices.append(index)

        x_recon = torch.stack(samples)[:, 0, :, :, :]
        indices = torch.stack(indices)[:, 0, :, :, :]
    
    #fig, axes = plt.subplots(1, 1, figsize=(16,16))
    
    
    #axes[0].set_title('Reconstructed images')
    plt.imshow(torchvision.utils.make_grid(
        torch.clamp(1 - x_recon, 0., 1.), nrow=8, pad_value=1).permute(1, 2, 0).cpu().numpy())
    plt.axis('off')


plot_latents_with_pixelcnn(model)
plt.savefig('/home/syslink/Documents/gm/ex4/samples.png')

a = 5

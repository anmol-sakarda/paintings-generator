# import the necessary libraries
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import datasets
from torchvision import transforms
import torch.optim as optim
import os

batch_size = 32
img_size = 32
data_dir = 'image_folder/'

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Apply the transformations
transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
# Load the dataset
imagenet_data = datasets.ImageFolder(data_dir, transform=transform)

# Load the image data into dataloader
celeba_train_loader = torch.utils.data.DataLoader(imagenet_data,
                                                  batch_size,
                                                  shuffle=True)


def scale(img, feature_range=(-1, 1)):
    min_, max_ = feature_range
    img = img * (max_ - min_) + min_
    return img


# helper conv function
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    layers = []
    conv_layer = nn.Conv2d(in_channels, out_channels,
                           kernel_size, stride, padding, bias=False)

    # Appending the layer
    layers.append(conv_layer)
    # Applying the batch normalization if it's given true
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    # returning the sequential container
    return nn.Sequential(*layers)


# helper deconv function
def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    layers = []
    convt_layer = nn.ConvTranspose2d(in_channels, out_channels,
                                     kernel_size, stride, padding, bias=False)

    # Appending the above conv layer
    layers.append(convt_layer)

    if batch_norm:
        # Applying the batch normalization if True
        layers.append(nn.BatchNorm2d(out_channels))

    # Returning the sequential container
    return nn.Sequential(*layers)


class Discriminator(nn.Module):

    def __init__(self, conv_dim):
        super(Discriminator, self).__init__()

        self.conv_dim = conv_dim

        # 32 x 32
        self.cv1 = conv(3, self.conv_dim, 4, batch_norm=False)
        # 16 x 16
        self.cv2 = conv(self.conv_dim, self.conv_dim * 2, 4, batch_norm=True)
        # 4 x 4
        self.cv3 = conv(self.conv_dim * 2, self.conv_dim * 4, 4, batch_norm=True)
        # 2 x 2
        self.cv4 = conv(self.conv_dim * 4, self.conv_dim * 8, 4, batch_norm=True)
        # Fully connected Layer
        self.fc1 = nn.Linear(self.conv_dim * 8 * 2 * 2, 1)

    def forward(self, x):
        # After passing through each layer
        # Applying leaky relu activation function
        x = F.leaky_relu(self.cv1(x), 0.2)
        x = F.leaky_relu(self.cv2(x), 0.2)
        x = F.leaky_relu(self.cv3(x), 0.2)
        x = F.leaky_relu(self.cv4(x), 0.2)
        # To pass throught he fully connected layer
        # We need to flatten the image first
        x = x.view(-1, self.conv_dim * 8 * 2 * 2)
        # Now passing through fully-connected layer
        x = self.fc1(x)
        return x


class Generator(nn.Module):

    def __init__(self, z_size, conv_dim):
        super(Generator, self).__init__()

        self.z_size = z_size

        self.conv_dim = conv_dim

        # fully-connected-layer
        self.fc = nn.Linear(z_size, self.conv_dim * 8 * 2 * 2)
        # 2x2
        self.dcv1 = deconv(self.conv_dim * 8, self.conv_dim * 4, 4, batch_norm=True)
        # 4x4
        self.dcv2 = deconv(self.conv_dim * 4, self.conv_dim * 2, 4, batch_norm=True)
        # 8x8
        self.dcv3 = deconv(self.conv_dim * 2, self.conv_dim, 4, batch_norm=True)
        # 16x16
        self.dcv4 = deconv(self.conv_dim, 3, 4, batch_norm=False)
        # 32 x 32

    def forward(self, x):
        # Passing through fully connected layer
        x = self.fc(x)
        # Changing the dimension
        x = x.view(-1, self.conv_dim * 8, 2, 2)
        # Passing through deconv layers
        # Applying the ReLu activation function
        x = F.relu(self.dcv1(x))
        x = F.relu(self.dcv2(x))
        x = F.relu(self.dcv3(x))
        x = torch.tanh(self.dcv4(x))
        # returning the modified image
        return x


def weights_init_normal(m):
    classname = m.__class__.__name__
    # For the linear layers
    if 'Linear' in classname:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        m.bias.data.fill_(0.01)
    # For the convolutional layers
    if 'Conv' in classname or 'BatchNorm2d' in classname:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)


# Defining the model hyperparamameters
d_conv_dim = 32
g_conv_dim = 32
z_size = 100  # Size of noise vector

D = Discriminator(d_conv_dim)
G = Generator(z_size=z_size, conv_dim=g_conv_dim)
# Applying the weight initialization
D.apply(weights_init_normal)
G.apply(weights_init_normal)


# print(D)
# print()
# print(G)

device = "cuda:0" if torch. cuda.is_available() else "cpu"


def real_loss(D_out, smooth=False):
    batch_size = D_out.size(0)
    print("batch size")
    if smooth:
        labels = torch.ones(batch_size) * 0.9
    else:
        labels = torch.ones(batch_size)

    labels = labels.to(device)
    print(labels.size())
    print("change labels")
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    print("set loss")
    return loss


def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size)
    labels = labels.to(device)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss


lr = 0.0005
beta1 = 0.3
beta2 = 0.999  # default value

# Optimizers
d_optimizer = optim.Adam(D.parameters(), lr, betas=(beta1, beta2))
g_optimizer = optim.Adam(G.parameters(), lr, betas=(beta1, beta2))


def train(D, G, n_epochs, print_every=100):
    # Trains adversarial networks for some number of epochs
    #  param, D: the discriminator network
    # param, G: the generator network
    #  param, n_epochs: number of epochs to train for
    # param, print_every: when to print and record the models' losses
    # return: D and G losses

    # move models to GPU
    #  if train_on_gpu:
    #     D.cuda()
    #     G.cuda()

    # keep track of loss and generated, "fake" samples
    samples = []
    losses = []

    # Get some fixed data for sampling. These are images that are held
    # constant throughout training, and allow us to inspect the model's performance
    sample_size = 16
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()
    # move z to GPU if available
    #   if train_on_gpu:
    #     fixed_z = fixed_z.cuda()

    # epoch training loop
    for epoch in range(n_epochs):

        # batch training loop
        for batch_i, (real_images, _) in enumerate(celeba_train_loader):
            #if batch_i == 31:
                #continue
            batch_size = real_images.size(0)
            real_images = scale(real_images)
            print("batch sz & real images")

            # ===============================================
            #         YOUR CODE HERE: TRAIN THE NETWORKS
            # ===============================================

            # 1. Train the discriminator on real and fake images
            d_optimizer.zero_grad()

            # real images
            real_images = real_images.to(device)

            dreal = D(real_images)
            print(batch_i)
            dreal_loss = real_loss(dreal)
            print("real images")

            # fake images

            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            print("made fake images")
            # move x to GPU, if available
            z = z.to(device)
            print("to device")
            fake_images = G(z)
            print("Generator thing")

            # loss of fake images
            dfake = D(fake_images)
            dfake_loss = fake_loss(dfake)
            print("fake image loss")

            # Adding both lossess
            d_loss = dreal_loss + dfake_loss
            # Backpropogation step
            d_loss.backward()
            d_optimizer.step()

            # 2. Train the generator with an adversarial loss
            g_optimizer.zero_grad()

            # Generate fake images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            z = z.to(device)
            fake_images = G(z)
            print("generated fake images")

            # Compute the discriminator losses on fake images
            # using flipped labels!
            D_fake = D(fake_images)
            g_loss = real_loss(D_fake, True)  # use real loss to flip labels
            print("compute discriminator losses on fake images")

            # perform backprop
            g_loss.backward()
            g_optimizer.step()

            # ===============================================
            #              END OF YOUR CODE
            # ===============================================

            # Print some loss stats
            if batch_i % print_every == 0:
                # append discriminator loss and generator loss
                losses.append((d_loss.item(), g_loss.item()))
                # print discriminator and generator loss
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch + 1, n_epochs, d_loss.item(), g_loss.item()))

        ## AFTER EACH EPOCH##
        # this code assumes your generator is named G, feel free to change the name
        # generate and save sample, fake images
        G.eval()  # for generating samples
        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train()  # back to training mode

    # Save training generator samples
    with open('train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)

    # finally return losses
    return losses


# set number of epochs
n_epochs = 400

# call training function
losses = train(D, G, n_epochs=n_epochs)
from __future__ import print_function

# diff 
import torch
import torch.nn.functional as F

# train
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdimport os
import glob
from PIL import Image
from skimage.measure import label
import sys
import math
from numbers import Number
from multiprocessing import cpu_count
from torch.nn.functional import adaptive_avg_pool2d
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.autograd import Variable
import torchvision
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import warnings
import torchvision.models as models
#from tqdm import tqdm_notebook as tqdm
import time
from torch.utils.tensorboard import SummaryWriter

from models.models import Generator, Discriminator
from fid.fid import FID, fid_obj
from diffAug.diffAug import DiffAugment, rand_brightness, rand_saturation, rand_contrast, rand_translation, rand_cutout, AUGMENT_FNS
from bay.bay import NoiseLoss, PriorLoss
from utils.utils import weights_init

parser = argparse.ArgumentParser()
dataroot="/content/animeface-character-dataset/animeface-character-dataset"
name="anime"
parser.add_argument("--workers", type=int, default=2, help="Number of workers for dataloader")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size during training")
parser.add_argument("--image_size", type=int, default=64, help="spatial size of training images. All images will be resized to this size")
parser.add_argument("--nc", type=int, default=3, help="Number of channels in the training images. For color images this is 3")
parser.add_argument("--nz", type=int, default=100, help="Size of z latent vector (i.e. size of generator input)")
parser.add_argument("--ngf", type=int, default=64, help="Size of feature maps in generator")
parser.add_argument("--ndf", type=int, default=64, help="Size of feature maps in discriminator")
parser.add_argument("--num_epochs", type=int, default=100, help="Number of training steps")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--aug_prob", type=float, default=1.0, help="probability of using Diff Augmentation during training")
parser.add_argument('--bayes', type=int, default=1, help='Do Bayesian GAN or normal GAN')
parser.add_argument('--gnoise_alpha', type=float, default=0.0001, help='')
parser.add_argument('--dnoise_alpha', type=float, default=0.0001, help='')

print("==============")
opt, args = parser.parse_known_args(sys.argv[1:])
print("==============")

# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(opt.image_size),
                                transforms.CenterCrop(opt.image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

ngpu = torch.cuda.device_count()


# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                     shuffle=True, num_workers=opt.workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()

netG = Generator(opt).to(device)
netG_no_diff = Generator(opt).to(device)
netG_bay = Generator(opt).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    print("This activates on gpu")
    netG = nn.DataParallel(netG, list(range(ngpu)))
    netG_no_diff = nn.DataParallel(netG_no_diff, list(range(ngpu)))
    netG_bay = nn.DataParallel(netG_bay, list(range(ngpu)))
# Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
netG.apply(weights_init)
netG_no_diff.apply(weights_init)
netG_bay.apply(weights_init)

# Print the G


netD = Discriminator(opt).to(device)
netD_no_diff = Discriminator(opt).to(device)
netD_bay = Discriminator(opt).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
    netD_no_diff = nn.DataParallel(netD_no_diff, list(range(ngpu)))
    netD_bay = nn.DataParallel(netD_bay, list(range(ngpu)))
    
# Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
netD.apply(weights_init)
netD_no_diff.apply(weights_init)
netD_bay.apply(weights_init)

# Print the model


# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize the progression of the generator
fixed_noise = torch.randn(64, opt.nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizerD_no_diff = optim.Adam(netD_no_diff.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2))
optimizerG_no_diff = optim.Adam(netG_no_diff.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2))
optimizerD_bay = optim.Adam(netD_bay.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2))
optimizerG_bay = optim.Adam(netG_bay.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2))

# Training Loop

# Lists to keep track of progress
img_list = []
img_list_no_diff = []
img_list_bay = []

G_losses = []
D_losses = []
G_losses_no_diff = []
D_losses_no_diff = []
G_losses_bay = []
D_losses_bay = []

FID_diff = []
FID_no_diff = []
FID_bay = []

# iters = 0
# since the log posterior is the average per sample, we also scale down the prior and the noise
# prior would be simulated enough times

gprior_criterion = PriorLoss(prior_std=1., observed=1000.)
gnoise_criterion = NoiseLoss(params=netG.parameters(), scale=math.sqrt(2*opt.gnoise_alpha/opt.lr), observed=1000.)
dprior_criterion = PriorLoss(prior_std=1., observed=50000.)
dnoise_criterion = NoiseLoss(params=netD.parameters(), scale=math.sqrt(2*opt.dnoise_alpha*opt.lr), observed=50000.)

print("Starting Training Loop...")
# For each epoch
for epoch in range(opt.num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        rand_prob = np.random.uniform(0, 1)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        netD_no_diff.zero_grad()
        netD_bay.zero_grad()

        # Format batch
        real_cpu = data[0].to(device)
        real_cpu_no_diff = data[0].to(device)
        real_cpu_bay = data[0].to(device)

        if rand_prob < opt.aug_prob:
            real_cpu = DiffAugment(real_cpu, policy='color,translation,cutout')
            real_cpu_bay = DiffAugment(real_cpu_bay, policy='color,translation,cutout')
           # plt.subplot(1, 2, 1)
           # plt.imshow(np.transpose(data[0][1].cpu().numpy(), axes=[1, 2, 0]))
           # plt.subplot(1, 2, 2)
           # plt.imshow(np.transpose(real_cpu[1].cpu().numpy(), axes=[1, 2, 0]))
           # plt.show()
      
        b_size = real_cpu.size(0)
        b_size_no_diff = real_cpu_no_diff.size(0)
        b_size_bay = real_cpu_bay.size(0)

        label = torch.full((b_size,), real_label, device=device)
        label_no_diff = torch.full((b_size_no_diff,), real_label, device=device)
        label_bay = torch.full((b_size_bay,), real_label, device=device)

        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        output_no_diff = netD_no_diff(real_cpu_no_diff).view(-1)
        output_bay = netD_bay(real_cpu_bay).view(-1)

        # Calculate loss on all-real batch
        label = label.to(dtype=torch.float)
        label_no_diff = label_no_diff.to(dtype=torch.float)
        label_bay = label_bay.to(dtype=torch.float)
        
        #label = torch.full((b_size,), label, device=device, dtype=torch.float)
        errD_real = criterion(output, label)
        errD_real_no_diff = criterion(output_no_diff, label_no_diff)
        errD_real_bay = criterion(output_bay, label_bay)

        # Calculate gradients for D in backward pass
        errD_real.backward()
        errD_real_no_diff.backward()
        errD_real_bay.backward()

        D_x = output.mean().item()
        D_x_no_diff = output_no_diff.mean().item()
        D_x_bay = output_bay.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, opt.nz, 1, 1, device=device)
        noise_no_diff = torch.randn(b_size_no_diff, opt.nz, 1, 1, device=device)
        noise_bay = torch.randn(b_size_bay, opt.nz, 1, 1, device=device)
        
        # Generate fake image batch with G
        fake = netG(noise)
        fake_no_diff = netG_no_diff(noise_no_diff)
        fake_bay = netG_bay(noise_bay)

        if rand_prob < opt.aug_prob:
            fake = DiffAugment(fake, policy='color,translation,cutout')
            fake_bay = DiffAugment(fake_bay, policy='color,translation,cutout')

            #plt.subplot(1, 2, 1)
            #plt.imshow(np.transpose(fake[0].cpu().detach().numpy(), axes=[1, 2, 0]))
            #plt.subplot(1, 2, 2)
            #plt.imshow(np.transpose(fake_aug[0].cpu().detach().numpy(), axes=[1, 2, 0]))
            #plt.show()
        
        label.fill_(fake_label)
        label_no_diff.fill_(fake_label)
        label_bay.fill_(fake_label)

        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        output_no_diff = netD_no_diff(fake_no_diff.detach()).view(-1)
        output_bay = netD_bay(fake_bay.detach()).view(-1)

        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        errD_fake_no_diff = criterion(output_no_diff, label_no_diff)
        errD_fake_bay = criterion(output_bay, label_bay)

        # Calculate the gradients for this batch
        errD_fake.backward()
        errD_fake_no_diff.backward()
        errD_fake_bay.backward()

        D_G_z1 = output.mean().item()
        D_G_z1_no_diff = output_no_diff.mean().item()
        D_G_z1_bay = output_bay.mean().item()

        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        errD_no_diff = errD_real_no_diff + errD_fake_no_diff
        errD_bay = errD_real_bay + errD_fake_bay

        if opt.bayes:
            errD_prior = dprior_criterion(netD_bay.parameters())
            errD_prior.backward()
            errD_noise = dnoise_criterion(netD_bay.parameters())
            errD_noise.backward()
            errD_bay += errD_prior 
            errD_bay += errD_noise
            
        # Update D
        optimizerD.step()
        optimizerD_no_diff.step()
        optimizerD_bay.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        netG_no_diff.zero_grad()
        netG_bay.zero_grad()

        label.fill_(real_label)  # fake labels are real for generator cost
        label_no_diff.fill_(real_label)
        label_bay.fill_(real_label)

        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        output_no_diff = netD_no_diff(fake_no_diff).view(-1)
        output_bay = netD_bay(fake_bay).view(-1)

        # Calculate G's loss based on this output
        errG = criterion(output, label)
        errG_no_diff = criterion(output_no_diff, label_no_diff)
        errG_bay = criterion(output_bay, label_bay)

        if opt.bayes:
           errG_bay += gprior_criterion(netG_bay.parameters())
           errG_bay += gnoise_criterion(netG_bay.parameters())

        # Calculate gradients for G
        errG.backward()
        errG_no_diff.backward()
        errG_bay.backward()

        D_G_z2 = output.mean().item()
        D_G_z2_no_diff = output_no_diff.mean().item()
        D_G_z2_bay = output_bay.mean().item()

        # Update G
        optimizerG.step()
        optimizerG_no_diff.step()
        optimizerG_bay.step()

        # Output training stats
        if i % 20 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch+1, opt.num_epochs, i+1, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            print('[%d/%d][%d/%d]\tLoss_D_no_diff: %.4f\tLoss_G_no_diff: %.4f\tD(x)_no_diff: %.4f\tD(G(z))_no_diff: %.4f / %.4f'
                      % (epoch+1, opt.num_epochs, i+1, len(dataloader),
                         errD_no_diff.item(), errG_no_diff.item(), D_x_no_diff, D_G_z1_no_diff, D_G_z2_no_diff))
            print('[%d/%d][%d/%d]\tLoss_D_bay: %.4f\tLoss_G_bay: %.4f\tD(x)_bay: %.4f\tD(G(z))_bay: %.4f / %.4f'
                      % (epoch+1, opt.num_epochs, i+1, len(dataloader),
                         errD_bay.item(), errG_bay.item(),D_x_bay, D_G_z1_bay, D_G_z2_bay))   

      
        if i == len(dataloader)-1:
            fid_val = fid_obj.compute_fid(real_cpu, fake)
            print("Diff {}".format(fid_val))
            FID_diff.append(fid_val)
            
            fid_val_no_diff = fid_obj.compute_fid(real_cpu_no_diff, fake_no_diff)
            print("No diff {}".format(fid_val_no_diff))
            FID_no_diff.append(fid_val_no_diff)
            
            fid_val_bay = fid_obj.compute_fid(real_cpu_bay, fake_bay)
            print("Bay + Diff {}".format(fid_val_bay))
            FID_bay.append(fid_val_bay)

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        G_losses_no_diff.append(errG_no_diff.item())
        D_losses_no_diff.append(errD_no_diff.item())
        G_losses_bay.append(errG_bay.item())
        D_losses_bay.append(errD_bay.item())

    if epoch % 10 == 0 or epoch == opt.num_epochs - 1:
        state = {'epoch_done': epoch, 'opt': opt, 'weight': netG.state_dict()}
        state_no_diff = {'epoch_done' : epoch, 'opt': opt, 'weight': netG_no_diff.state_dict()}
        state_bay = {'epoch_done' : epoch, 'opt': opt, 'weight': netG_bay.state_dict()}

        torch.save(state, "%s/%s/saved_models/generator_%d.pth" % ("checkpoint", name, epoch))
        torch.save(netD.state_dict(),
                       "%s/%s/saved_models/discriminator_%d.pth" % ("checkpoint", name, epoch))
        
        torch.save(state_no_diff, "%s/%s/saved_models_no_diff/generator_%d.pth" % ("checkpoint", name, epoch))
        torch.save(netD_no_diff.state_dict(),
                       "%s/%s/saved_models_no_diff/discriminator_%d.pth" % ("checkpoint", name, epoch))
        
        torch.save(state_bay, "%s/%s/saved_models_bay/generator_%d.pth" % ("checkpoint", name, epoch))
        torch.save(netD_bay.state_dict(),
                       "%s/%s/saved_models_bay/discriminator_%d.pth" % ("checkpoint", name, epoch))

        # Check how the generator is doing by saving G's output on fixed_noise
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
            fake_no_diff = netG_no_diff(fixed_noise).detach().cpu()
            fake_bay = netG_bay(fixed_noise).detach().cpu()

        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        vutils.save_image(fake, "%s/%s/images/%d.jpg" % ("checkpoint", name, epoch))
        img_list_no_diff.append(vutils.make_grid(fake_no_diff, padding=2, normalize=True))
        vutils.save_image(fake_no_diff, "%s/%s/images_no_diff/%d.jpg" % ("checkpoint", name, epoch))
        img_list_bay.append(vutils.make_grid(fake_bay, padding=2, normalize=True))
        vutils.save_image(fake_bay, "%s/%s/images_bay/%d.jpg" % ("checkpoint", name, epoch))  

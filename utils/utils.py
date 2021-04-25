
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

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

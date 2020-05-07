import os
import time
import sys
import random
import psutil


import copy
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box
from sklearn.metrics import confusion_matrix

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

class ConvLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 bias = True,
                 pool=False,
                 mp_kernel_size=2,
                 mp_stride=2):
        super(ConvLayer, self).__init__()
        if pool:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(0.5),
                nn.LeakyReLU(negative_slope=0.1), ## nn.ReLU(),
                nn.MaxPool2d(kernel_size=mp_kernel_size, stride=mp_stride))
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(0.5),
                nn.LeakyReLU(negative_slope=0.1), ## nn.ReLU(),
                )

    def forward(self, x):
        return self.layer(x)

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearLayer, self).__init__()
        self.layer = nn.Sequential(
            torch.nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.Dropout(0.5),
            nn.LeakyReLU(negative_slope=0.1) ## nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)

class ConvTLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 unpool=False,
                 mp_kernel_size=2,
                 mp_stride=2):
        super(ConvTLayer, self).__init__()
        if unpool:
            self.layer = nn.Sequential(
                nn.ConvTranspose2d(in_channels,
                                   out_channels,
                                   kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   output_padding=output_padding,
                                   bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(0.5),
                nn.LeakyReLU(negative_slope=0.1), ## nn.ReLU()
                nn.MaxUnpool2d(kernel_size=mp_kernel_size, stride=mp_stride)
            )
        else:
            self.layer = nn.Sequential(
                nn.ConvTranspose2d(in_channels,
                                   out_channels,
                                   kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   output_padding=output_padding,
                                   bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(0.5),
                nn.LeakyReLU(negative_slope=0.1), ## nn.ReLU()
            )
    def forward(self, x):
        return self.layer(x)

class Encoder1(nn.Module):
    def __init__(self, d):
        super(Encoder1, self).__init__()
        self.conv1 = ConvLayer(3,96, stride=2)
        self.conv2 = ConvLayer(96,128, stride=2)
        self.conv3 = ConvLayer(128,256, stride=2)
        self.conv4 = ConvLayer(256,512, stride=2)
        self.conv5 = ConvLayer(512,1024, stride=2)
        self.conv6 = ConvLayer(1024,2048, stride=2)
        self.lin1 = nn.Linear(2048*3*3, d)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.lin1(x.reshape(-1,2048*3*3))
        return x

class CNN(nn.Module):
    def __init__(self, d=650, output_size=4):
        super(CNN, self).__init__()
        self.encoder = Encoder1(d=d)
        self.linear = nn.Linear(d,4)

    def forward(self, x):
        x = self.encoder(x)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)

class Encoder(nn.Module):
    def __init__(self, d=650):
        super(Encoder, self).__init__()
        self.encoder = Encoder1(d=d)

    def forward(self, x):
        x = self.encoder(x)
        return x


class EncoderY(nn.Module):
    def __init__(self,  d):
        super(EncoderY, self).__init__()
        self.conv1 = ConvLayer(3,96, stride=2)
        self.conv2 = ConvLayer(96,128, stride=2)
        self.conv3 = ConvLayer(128,256, stride=2)
        self.conv4 = ConvLayer(256,512, stride=2)
        self.conv5 = ConvLayer(512,1024, stride=2)
        self.conv6 = ConvLayer(1024,2048, stride=2)
        self.lin1 = nn.Linear(2048*3*3, d)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        #print(x.shape)
        x = self.lin1(x.reshape(-1,2048*3*3))
        return x

class EncoderX(nn.Module):
    def __init__(self, d):
        super(EncoderX, self).__init__()
        self.conv1 = ConvLayer(1,16, stride=2)
        self.conv2 = ConvLayer(16,32, stride=2)
        self.conv3 = ConvLayer(32,48, stride=2)
        self.conv4 = ConvLayer(48,64, stride=2)
        self.conv5 = ConvLayer(64,96, stride=2)
        self.conv6 = ConvLayer(96,128, stride=2)
        self.conv7 = ConvLayer(128,256, stride=2)
        self.conv8 = ConvLayer(256,512, stride=2)
        self.lin1 = nn.Linear(512*2*2, d)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        #print(x.shape)
        x = self.lin1(x.reshape(-1,512*2*2))
        return x

class DecoderX(nn.Module):
    def __init__(self):
        super(DecoderX, self).__init__()
        self.convt1 = ConvTLayer(4096, 2048, kernel_size=3, stride=2)
        self.convt2 = ConvTLayer(2048, 1024, kernel_size=3, stride=3, output_padding=(0,0))
        self.convt3 = ConvTLayer(1024, 512, kernel_size=3, stride=2, padding=(1,1), output_padding=(0,0))
        self.convt4 = ConvTLayer(512, 256, kernel_size=3, stride=3, padding=(1,1), output_padding=(0,0))
        self.convt5 = ConvTLayer(256, 128, kernel_size=3, stride=2, output_padding=(0,0))
        self.convt6 = ConvTLayer(128, 96, kernel_size=3, stride=2, output_padding=(0,0))
        self.convt7 = ConvTLayer(96, 64, kernel_size=3, stride=2, output_padding=(0,0))
        self.convt8 = ConvTLayer(64, 1, kernel_size=3, stride=2, output_padding=(1,1))

    def forward(self,z):
        z = self.convt1(z)
        z = self.convt2(z)
        z = self.convt3(z)
        z = self.convt4(z)
        z = self.convt5(z)
        z = self.convt6(z)
        z = self.convt7(z)
        z = self.convt8(z)
        return torch.sigmoid(z)

# Defining the model

class CNN_VAE(nn.Module):
    def __init__(self, hidden_d=196, image_d=650): #hidden_d=196, image_d=650 or hidden_d=286, image_d=625
        super().__init__()

        self.d = hidden_d
        self.id = image_d

        self.y_encoder = EncoderY(d=self.id)

        self.x_encoder = EncoderX(d=2*self.d)

        self.x_decoder = DecoderX()

    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, y):
        mu_logvar = self.x_encoder(x).view(-1, 2, self.d)
        #print(mu_logvar.shape)
        img_enc = [self.y_encoder(img.squeeze()) for img in y]
        mu = mu_logvar[:, 0, :]
        #print(mu.shape)
        logvar = mu_logvar[:, 1, :]
        #print(logvar.shape)
        z = self.reparameterise(mu, logvar)
        img_enc.append(z)
        out = torch.cat(img_enc,axis=1).reshape(-1,4096,1,1)
        return self.x_decoder(out), mu, logvar

    def inference(self, y, device, mu=None, logvar=None):
        N = y.size(1)
        z = torch.randn((N, self.d)).to(device)
        #print('Prior:',z.shape)
        if mu is not None and logvar is not None:
            #print(mu.shape)
            #print(logvar.shape)
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            z = eps.mul(std).add_(mu)
            #print('Post:',z.shape)
        z = z.reshape(-1,self.d)
        img_enc = [self.y_encoder(img) for img in y]
        img_enc.append(z)
        out = torch.cat(img_enc,axis=1).reshape(-1,4096,1,1)
        return self.x_decoder(out)

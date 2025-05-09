# CS 551 - Deep Learning
# The code base is based on the great work from CSC 321, U Toronto
# https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip
# CSC 321, Assignment 4
#
# This file contains the models used for both parts of the assignment:
#
#   - DCGenerator       --> Used in the vanilla GAN in Part 1
#   - CycleGenerator    --> Used in the CycleGAN in Part 2
#   - DCDiscriminator   --> Used in both the vanilla GAN and CycleGAN (Parts 1 and 2)
# For the assignment, you are asked to create the architectures of these three networks by
# filling in the __init__ and forward methods in the DCGenerator, CycleGenerator, and DCDiscriminator classes.
# Feel free to add and try your own models

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, norm='batch'):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))

    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, norm='batch', init_zero_weights=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    layers.append(conv_layer)

    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))
    return nn.Sequential(*layers)


class DCGenerator(nn.Module):
    def __init__(self, noise_size, conv_dim):
        super(DCGenerator, self).__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        self.deconv1 = deconv(noise_size, conv_dim * 8, kernel_size=4, stride=1, padding=0, norm='batch')
        self.deconv2 = deconv(conv_dim * 8, conv_dim * 4, kernel_size=4, stride=2, padding=1, norm='batch')
        self.deconv3 = deconv(conv_dim * 4, conv_dim * 2, kernel_size=4, stride=2, padding=1, norm='batch')
        self.deconv4 = deconv(conv_dim * 2, conv_dim, kernel_size=4, stride=2, padding=1, norm='batch')
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(conv_dim, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )  

    def forward(self, z):
        """Generates an image given a sample of random noise.

            Input
            -----
                z: BS x noise_size x 1 x 1   -->  16x100x1x1

            Output
            ------
                out: BS x channels x image_width x image_height  -->  16x3x32x32
        """


        ###########################################
        ##   FILL THIS IN: FORWARD PASS   ##
        ###########################################

        out = F.relu(self.deconv1(z))
        out = F.relu(self.deconv2(out))
        out = F.relu(self.deconv3(out))
        out = F.relu(self.deconv4(out))
        out = F.tanh(self.deconv5(out))
        return out


class ResnetBlock(nn.Module):
    def __init__(self, conv_dim, norm):
        super(ResnetBlock, self).__init__()
        self.conv_layer = conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1, norm=norm)

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out



class CycleGenerator(nn.Module):
    """Defines the architecture of the generator network.
       Note: Both generators G_XtoY and G_YtoX have the same architecture in this assignment.
    """
    def __init__(self, conv_dim=64, init_zero_weights=False, norm='batch'):
        super(CycleGenerator, self).__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        # 1. Define the encoder part of the generator (that extracts features from the input image)
        self.pad = nn.ReflectionPad2d(3)
        self.conv1 = nn.Sequential(self.pad, conv(3, conv_dim, kernel_size=7, stride=1, padding=0, norm=norm))
        self.conv2 = conv(conv_dim, conv_dim * 2, kernel_size=3, stride=2, padding=1, norm=norm)

        # 2. Define the transformation part of the generator
        self.resnet_block = nn.Sequential(
            *[ResnetBlock(conv_dim * 2, norm) for _ in range(6)]
        )

        # 3. Define the decoder part of the generator (that builds up the output image from features)
        self.deconv1 = deconv(conv_dim * 2, conv_dim, kernel_size=4, stride=2, padding=1, norm=norm)
        self.deconv2 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(conv_dim, 3, kernel_size=7, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        """Generates an image conditioned on an input image.

            Input
            -----
                x: BS x 3 x 32 x 32

            Output
            ------
                out: BS x 3 x 32 x 32
        """

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))

        out = F.relu(self.resnet_block(out))

        out = F.relu(self.deconv1(out))
        out = F.tanh(self.deconv2(out))

        return out


class DCDiscriminator(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """
    def __init__(self, conv_dim=64, norm='batch'):
        super(DCDiscriminator, self).__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        self.conv1 = conv(3, conv_dim, kernel_size=4, stride=2, padding=1, norm=None) 
        self.conv2 = conv(conv_dim, conv_dim * 2, kernel_size=4, stride=2, padding=1, norm=norm) 
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, kernel_size=4, stride=2, padding=1, norm=norm) 
        self.conv4 = conv(conv_dim * 4, conv_dim * 8, kernel_size=4, stride=2, padding=1, norm=norm) 
        self.conv5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(conv_dim * 8, 1, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        ###########################################
        ##   FILL THIS IN: FORWARD PASS   ##
        ###########################################

        out = F.leaky_relu(self.conv1(x), 0.2)
        out = F.leaky_relu(self.conv2(out), 0.2)
        out = F.leaky_relu(self.conv3(out), 0.2)
        out = F.leaky_relu(self.conv4(out), 0.2)
        out = self.conv5(out)
        out = torch.sigmoid(out)
        return out.view(out.size(0), -1).mean(1)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:37:59 2020

@author: Brenda
"""

"""CNN Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, img_shape):
        super(MyModel, self).__init__()

        ## define layers

        # convolution layer
        c_out = 32
        c_kernel = 5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c_out, kernel_size=c_kernel)
        self.conv2 = nn.Conv2d(32, 64 ,kernel_size=c_kernel)
        self.conv3 = nn.Conv2d(64,128,kernel_size=c_kernel)
        #x = torch.randn(50,50).view(-1,1,50,50)
        #self._to_linear = None
        #self.convs(x)
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

        # linear layers (note this is a simplified formula assuming default padding, dilation, etc.
        #d1_input = c_out * (img_shape[0]-c_kernel+1) * (img_shape[1]-c_kernel+1)
        #d1_output = 512
        #self.d1 = nn.Linear(d1_input, d1_output)
        #self.d2 = nn.Linear(d1_output, 2)
        
    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        
        #print(x[0].shape)
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        # 32x1x28x28 => 32x32x26x26
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # flatten => 32 x (32*26*26)
        #x = x.flatten(start_dim = 1)

        # 32 x (32*26*26) => 32x128
        #x = self.d1(x)
       # x = F.relu(x)

        # logits => 32x10
       ## out = (logits, dim=1)
        return F.softmax(x, dim=1)

## compute accuracy
def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()

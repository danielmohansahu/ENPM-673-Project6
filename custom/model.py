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
        c_out = 5
        c_kernel = 10
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c_out, kernel_size=c_kernel)

        # linear layers (note this is a simplified formula assuming default padding, dilation, etc.
        d1_input = c_out * (img_shape[0]-c_kernel+1) * (img_shape[1]-c_kernel+1)
        d1_output = 64
        self.d1 = nn.Linear(d1_input, d1_output)
        self.d2 = nn.Linear(d1_output, 2)

    def forward(self, x):
        # 32x1x28x28 => 32x32x26x26
        x = self.conv1(x)
        x = F.relu(x)

        # flatten => 32 x (32*26*26)
        x = x.flatten(start_dim = 1)

        # 32 x (32*26*26) => 32x128
        x = self.d1(x)
        x = F.relu(x)

        # logits => 32x10
        logits = self.d2(x)
        out = F.softmax(logits, dim=1)
        return out

## compute accuracy
def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()

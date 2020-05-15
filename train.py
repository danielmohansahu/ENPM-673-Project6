#!/usr/bin/env python3
"""
Code adapted from:
https://colab.research.google.com/github/omarsar/pytorch_notebooks/blob/master/pytorch_quick_start.ipynb#scrollTo=BZz7LAewgGAK
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from custom.data import MyDataset
from custom.model import MyModel, get_accuracy

BATCH_SIZE = 500
MAX_IMGS = 5000
IMG_SHAPE = (250,250)
LEARNING_RATE = 0.001
NUM_EPOCHS = 25

if __name__ == "__main__":
    ## prepare dataset
    
    transform = transforms.Compose(
        [transforms.Resize(IMG_SHAPE),
         #transforms.RandomResizedCrop(50),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()
        ])
    
    trainset = MyDataset("./data/dogs-vs-cats/train/",  transform, MAX_IMGS)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=4)
    
    # instantiate model and prepare GPU 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MyModel(IMG_SHAPE)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # train
    result_accuracy = []
    result_loss = []
    for epoch in range(NUM_EPOCHS):
        train_running_loss = 0.0
        train_acc = 0.0
    
        model = model.train()
    
        ## training step
        for i, (images, labels) in enumerate(trainloader):
            
            images = images.to(device)
            labels = labels.to(device)
    
            ## forward + backprop + loss
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
    
            ## update model params
            optimizer.step()
    
            train_running_loss += loss.detach().item()
            train_acc += get_accuracy(logits, labels, BATCH_SIZE)
        
        model.eval()
        result_accuracy.append(train_acc/i)
        result_loss.append(train_running_loss/i)
        print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' \
              %(epoch, train_running_loss / i, train_acc/i))  

    # plot the results
    fig,axs = plt.subplots(2)
    fig.suptitle("Training Results")
    axs[0].plot(range(NUM_EPOCHS), result_accuracy)
    axs[0].set(ylabel="Accuracy")
    axs[1].plot(range(1,NUM_EPOCHS+1), result_loss)
    axs[1].set(xlabel="Epoch",ylabel="Loss")
    plt.show()

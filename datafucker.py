import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np

def data_generator():
    torch.manual_seed(73)
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader,test_loader

if __name__=="__main__":
    data_generator()
    

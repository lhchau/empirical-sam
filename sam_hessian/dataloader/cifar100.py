import os
import torch
import torchvision
import torchvision.transforms as transforms
from .cutout import Cutout
from torch.utils.data import random_split


def get_cifar100(
    batch_size=128,
    num_workers=4,
    split=(0.8, 0.2) 
):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        Cutout()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    data_train, data_val = random_split(
        dataset=torchvision.datasets.CIFAR100(root=os.path.join('.', 'data'), train=True, download=True, transform=transform_train),
        lengths=split,
        generator=torch.Generator().manual_seed(42)
    )
    data_test = torchvision.datasets.CIFAR100(root=os.path.join('.', 'data'), train=False, download=True, transform=transform_test)
    
    train_dataloader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(
        data_val, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(
        data_test, batch_size=100, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_dataloader, val_dataloader, test_dataloader, len(data_test.classes)
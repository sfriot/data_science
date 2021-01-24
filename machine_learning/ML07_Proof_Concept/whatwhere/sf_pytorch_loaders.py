# -*- coding: utf-8 -*-
"""
Created on Thu Nov 5

@author: Sylvain Friot
Based on https://github.com/alinlab/L2T-ww/blob/f4dde04e8d5d5725dc3bff2f59cb0d0c26d0bcbe/check_dataset.py
"""

import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.utils.data as data
from sklearn.model_selection import train_test_split


class DatasetFromSubset(data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def get_preprocess(model, with_data_augmentation=True):
    if model.model_family in ["resnet", "vgg", "densenet", "efficientnet"]:
        img_tocrop = 256
        img_size = 224
        img_normalization = transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
    elif model.model_family in ["xception", "inception"]:
        img_tocrop = 333
        img_size = 299
        if model.name == "xception":
            img_normalization = transforms.Normalize([0.5, 0.5, 0.5],
                                                     [0.5, 0.5, 0.5])
        else:
            img_normalization = transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])
    elif model.model_family == "mycnn":
        img_tocrop = 290
        img_size = 256
        img_normalization = transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
    else:
        raise ValueError("This model is not implemented")

    if with_data_augmentation:
        train_transform = transforms.Compose(
                [transforms.RandomResizedCrop((img_size, img_size)),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 img_normalization])
        validation_transform = transforms.Compose(
                [transforms.Resize((img_tocrop, img_tocrop)),
                 transforms.CenterCrop(img_size),
                 transforms.ToTensor(),
                 img_normalization])
    else:
        train_transform = transforms.Compose(
                [transforms.Resize((img_size, img_size)),
                 transforms.ToTensor(),
                 img_normalization])
        validation_transform = transforms.Compose(
                [transforms.Resize((img_size, img_size)),
                 transforms.ToTensor(),
                 img_normalization])
    return train_transform, validation_transform


def get_dataset_fromsubfolders(dataset_path, stratify, train_val_split,
                               subfolders_name, mini_data=1.0):
    train, test = subfolders_name[0], subfolders_name[1]
    ## train and valid set
    train_data = dset.ImageFolder(root=os.path.join(dataset_path, train))
    valid_size = int(train_val_split[1] * len(train_data))
    train_size = len(train_data) - valid_size
    if mini_data < 1.0:
        mini_size = int(train_size * mini_data) + 1
        unused_size = train_size - mini_size
    if stratify:
        targets = np.array(train_data.targets)
        train_idx, valid_idx = train_test_split(np.arange(len(targets)),
                                                train_size=train_size, test_size=valid_size,
                                                shuffle=True, random_state=42,
                                                stratify=targets)
        if mini_data < 1.0:
            mini_idx, _ = train_test_split(train_idx,
                                           train_size=mini_size, test_size=unused_size,
                                           shuffle=True, random_state=42,
                                           stratify=targets[train_idx])
            train_set = data.Subset(train_data, mini_idx)
        else:
            train_set = data.Subset(train_data, train_idx)
        valid_set = data.Subset(train_data, valid_idx)
    else:
        if mini_data < 1.0:
            train_set, valid_set, _ = \
                data.random_split(train_data, [mini_size, valid_size, unused_size],
                                  generator=torch.Generator().manual_seed(42))
        else:
            train_set, valid_set = \
                data.random_split(train_data, [train_size, valid_size],
                                  generator=torch.Generator().manual_seed(42))
    # test set
    test_set = dset.ImageFolder(root=os.path.join(dataset_path, test))
    return train_set, valid_set, test_set


def get_dataset_fromsamefolder(dataset_path, stratify, train_val_split,
                               mini_data=1.0):
    full_data = dset.ImageFolder(root=dataset_path)
    fulltrain_size = int(train_val_split[0] * len(full_data))
    valid_size = int(train_val_split[1] * fulltrain_size)
    train_size = fulltrain_size - valid_size
    test_size = len(full_data) - fulltrain_size
    if mini_data < 1.0:
        mini_size = int(train_size * mini_data) + 1
        unused_size = train_size - mini_size
    if stratify:
        targets = np.array(full_data.targets)
        fulltrain_idx, test_idx = train_test_split(np.arange(len(targets)),
                                                   train_size=fulltrain_size, test_size=test_size,
                                                   shuffle=True, random_state=42,
                                                   stratify=targets)
        train_idx, valid_idx = train_test_split(fulltrain_idx,
                                                train_size=train_size, test_size=valid_size,
                                                shuffle=True, random_state=42,
                                                stratify=targets[fulltrain_idx])
        if mini_data < 1.0:
            mini_idx, _ = train_test_split(train_idx,
                                           train_size=mini_size, test_size=unused_size,
                                           shuffle=True, random_state=42,
                                           stratify=targets[train_idx])
            train_set = data.Subset(full_data, mini_idx)
        else:
            train_set = data.Subset(full_data, train_idx)
        valid_set = data.Subset(full_data, valid_idx)
        test_set = data.Subset(full_data, test_idx)
    else:
        if mini_data < 1.0:
            train_set, valid_set, test_set, _ = \
                data.random_split(full_data, [mini_size, valid_size, test_size, unused_size],
                                  generator=torch.Generator().manual_seed(42))
        else:
            train_set, valid_set, test_set = \
                data.random_split(full_data, [train_size, valid_size, test_size],
                                  generator=torch.Generator().manual_seed(42))
    return train_set, valid_set, test_set


def get_dataset(dataset_path, model, batch_size=32, mini_data=1.0, stratify=True,
                with_data_augmentation=True, train_val_split=[0.8, 0.2],
                traintest_subfolders=False, subfolders_name=["train", "test"]):
    # setting of subsets
    if traintest_subfolders:
        train_set, valid_set, test_set = get_dataset_fromsubfolders(
                dataset_path, stratify, train_val_split,
                subfolders_name, mini_data)
    else:
        train_set, valid_set, test_set = get_dataset_fromsamefolder(
                dataset_path, stratify, train_val_split, mini_data)
    # transformation of subsets
    train_transform, validation_transform = \
        get_preprocess(model, with_data_augmentation=with_data_augmentation)
    # sets definition and loading
    sets = [DatasetFromSubset(train_set, transform=train_transform),
            DatasetFromSubset(train_set, transform=validation_transform),
            DatasetFromSubset(valid_set, transform=validation_transform),
            DatasetFromSubset(test_set, transform=validation_transform)]
    loaders = [data.DataLoader(subset, batch_size=batch_size,
                               shuffle=False, num_workers=0)  # shuffle = False to keep same batches between source and target data
               for subset in sets]
    return loaders

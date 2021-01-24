# -*- coding: utf-8 -*-
"""
Created on Sun Nov 8 2020

@author: Sylvain Friot
FeatureExtraction is a class to handle the fonctions related to features
This class is inherited by each model
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os


class FeatureExtraction(nn.Module):

    def __init__(self, max_num_features=None, folder_saves=None):
        super().__init__()
        self.extraction_num_features = None
        self.extraction_dimension = None
        self.extraction_best_features = None
        self.max_num_features = max_num_features
        self.folder_saves = folder_saves
        self.path_to_best_features = None
        self.set_auto_path()

    def set_auto_path(self):
        if (self.folder_saves is not None) & (self.max_num_features is not None):
            self.path_to_best_features = "{}/{}_{}.npy"\
                .format(self.folder_saves, self.model_name, self.max_num_features)
        else:
            self.path_to_best_features = None

    def forward_with_features(self, x, with_filter=True):
        out, feat = self.forward(x, include_feat=True)
        if with_filter:
            return out, self.filter_features(feat)
        return out, feat

    def forward_only_features(self, x, with_filter=True):
        out, feat = self.forward(x, include_feat=True)
        if with_filter:
            return self.filter_features(feat)
        return feat

    def filter_features(self, feat):
        filtered_feat = feat.copy()
        for idx in range(len(self.extraction_num_features)):
            if self.extraction_best_features[idx] is not None:
                filtered_feat[idx] = filtered_feat[idx]\
                    [:, self.extraction_best_features[idx], :, :]
        return filtered_feat

    def get_num_features(self):
        """
        This method calculates the number of features, and their dimension, for each extraction level
        """
        with torch.no_grad():
            input_test = Image.fromarray(np.random.randint(0, 256, size=(224, 224, 3)).astype("uint8"))
            x = transforms.ToTensor()(input_test)
            x = x.unsqueeze(0)        
            feat = self.forward_only_features(x, with_filter=False)
        self.extraction_num_features = [f.size(1) for f in feat]
        self.extraction_dimension = [f.size(2) for f in feat]
        self.extraction_best_features = [None for f in feat]
        if self.path_to_best_features:
            if os.path.exists(self.path_to_best_features):
                self.extraction_best_features = np.load(self.path_to_best_features, allow_pickle=True)
        if self.max_num_features is None:
            self.max_num_features = max(self.extraction_num_features)

    def freeze_layers(self, first_layer=0, last_layer=None):
        if first_layer < 0:
            first_layer += len(list(self.children()))
        if last_layer is None:
            last_layer = len(list(self.children()))
        if last_layer < 0:
            last_layer += len(list(self.children()))
        idx_layer = 0
        for name, child in self.named_children():
            print("{} : {}".format(idx_layer, name))
            if (idx_layer >= first_layer) & (idx_layer < last_layer):
                print("freezed")
                for param in child.parameters():
                    param.requires_grad = False
            else:
                print("unfreezed")
                for param in child.parameters():
                    param.requires_grad = True
            idx_layer += 1

    def unfreeze_layers(self, first_layer=0, last_layer=None):
        if first_layer < 0:
            first_layer += len(list(self.children()))
        if last_layer is None:
            last_layer = len(list(self.children()))
        if last_layer < 0:
            last_layer += len(list(self.children()))
        idx_layer = 0
        for name, child in self.named_children():
            print("{} : {}".format(idx_layer, name))
            if (idx_layer >= first_layer) & (idx_layer < last_layer):
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False
            idx_layer += 1

    def save_model(self, url):
        torch.save(self.state_dict(), url)

    def load_model(self, url):
        self.load_state_dict(torch.load(url))

    def _define_best_features(self, dataloader, max_num_features=None, to_device=None, path_save=None, debug=False):
        """
        This method determines the best features for an extraction level if this level has over 'max_features' features
        I extract the features of this level. I calculate a mean value by class for each pixel and each feature.
        For each feature, I calculate the standard deviation of those mean values for each pixel across the different classes.
        To finish, I calculate the mean of all those standard deviation for each feature.
        More this mean of standard deviation is high, more the pixels of the feature seem able to segregate the different classes.
        So I keep the top "max_features" with the largest mean of standard deviation.
        """
        if max_num_features is not None:
            self.max_num_features = max_num_features
            self.get_auto_path()
        self.extraction_best_features = [None for i in self.extraction_num_features]
        for idx, nb in enumerate(self.extraction_num_features):
            if nb > self.max_num_features:
                if debug:
                    self.extraction_best_features[idx] = [i for i in range(self.max_num_features)]
                else:
                    self.extraction_best_features[idx] = self.__get_best_features(idx, dataloader, self.max_num_features, to_device)
        self.__save_best_features(path_save)

    def __get_best_features(self, idx, dataloader, max_features, to_device=None):
        mypixels = AveragePixels(self.extraction_num_features[idx],
                                 self.num_classes,
                                 self.extraction_dimension[idx])
        i = 0
        with torch.no_grad():
            for x, y in dataloader:
                i += 1
                if to_device is not None:
                    x, y = x.to(to_device), y.to(to_device)
                extraction = self.forward_only_features(x, with_filter=False)[idx]
                for extract_idx in range(extraction.size(0)):
                    for feat_idx in range(extraction.size(1)):
                        for h_idx in range(extraction.size(2)):
                            for w_idx in range(extraction.size(3)):
                                mypixels.update(y[extract_idx], feat_idx,
                                                h_idx, w_idx,
                                                extraction[extract_idx, feat_idx, h_idx, w_idx])
            features_scores = mypixels.get_stats()
        return np.argpartition(-features_scores, max_features)[:max_features]

    def __save_best_features(self, path_save=None):
        if path_save is None:
            if self.path_to_best_features is None:
                raise ValueError("path_save is not defined")
            path_save = self.path_to_best_features
        np.save(path_save, self.extraction_best_features)


class AveragePixels(object):
    """Computes the average and standard deviation values needed to extract the best features"""
    def __init__(self, num_features, num_classes, num_pixels):
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_pixels = num_pixels
        self.reset()

    def reset(self):
        self.sum = np.zeros((self.num_features, self.num_classes, self.num_pixels, self.num_pixels))
        self.count = np.zeros(self.num_classes)
        self.avg = np.zeros((self.num_features, self.num_classes, self.num_pixels, self.num_pixels))
        self.std = np.zeros((self.num_features, self.num_pixels, self.num_pixels))
        self.mean_std = np.zeros(self.num_features)

    def update(self, label_idx, feat_idx, h_idx, w_idx, value):
        self.sum[feat_idx, label_idx, h_idx, w_idx] += value
        self.count[label_idx] += 1
        
    def get_stats(self):
        # average by class of the pixels values for each feature map
        for feat_idx in range(self.num_features):
            for h_idx in range(self.num_pixels):
                for w_idx in range(self.num_pixels):
                    for label_idx in range(self.num_classes):
                        self.avg[feat_idx, label_idx, h_idx, w_idx] = \
                            self.sum[feat_idx, label_idx, h_idx, w_idx] \
                            / self.count[label_idx]
        # standard deviation of averages previously calculated, across classes
        for feat_idx in range(self.num_features):
            for h_idx in range(self.num_pixels):
                for w_idx in range(self.num_pixels):
                    values = []
                    for label_idx in range(self.num_classes):
                        values.append(self.avg[feat_idx, label_idx, h_idx, w_idx])
                    self.std[feat_idx, h_idx, w_idx] = np.std(values)
        # average of stantdard deviation = power of classification of the feature map
        for feat_idx in range(self.num_features):
            values = []
            for h_idx in range(self.num_pixels):
                for w_idx in range(self.num_pixels):
                    values.append(self.std[feat_idx, h_idx, w_idx])
            self.mean_std[feat_idx] = np.mean(values)
        return self.mean_std

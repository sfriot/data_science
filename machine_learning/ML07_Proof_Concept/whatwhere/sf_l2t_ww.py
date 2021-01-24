# -*- coding: utf-8 -*-
"""
Created on Wed Nov 4 2020

@author: Sylvain Friot
Based on https://github.com/alinlab/L2T-ww/blob/master/train_l2t_ww.py
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sf_pytorch_loaders as sf_load
import sf_meta_optimizers as sf_optim
import csv
from PIL import Image

torch.backends.cudnn.benchmark = True


def _get_num_features(model, device):
    """
    This method automatically gets the number of feature maps of extracted layers
    We select only layers before a dimension reduction
    """
    input_test = Image.fromarray(np.random.randint(0, 256, size=(256, 256, 3)).astype("uint8"))
    _, valid_transform = sf_load.get_preprocess(model, with_data_augmentation=False)
    x = valid_transform(input_test)
    x = x.unsqueeze(0)
    x = x.to(device)
    feat = model.forward_only_features(x)
    return [f.size(1) for f in feat]


def _get_pairs(source_num_features, target_num_features):
    pairs = []
    for src_idx in range(len(source_num_features)):
        for tgt_idx in range(len(target_num_features)):
            pairs.append((src_idx, tgt_idx))
    return pairs


class FeatureMatching(nn.ModuleList):
    def __init__(self, source_num_features, target_num_features, pairs):
        super(FeatureMatching, self).__init__()
        self.src_list = source_num_features
        self.tgt_list = target_num_features
        self.pairs = pairs
        for src_idx, tgt_idx in pairs:
            self.append(nn.Conv2d(self.tgt_list[tgt_idx], self.src_list[src_idx], 1))

    def forward(self, source_features, target_features,
                weight, beta, loss_weight):
        matching_loss = 0.0
        for i, (src_idx, tgt_idx) in enumerate(self.pairs):
            sw = source_features[src_idx].size(3)
            tw = target_features[tgt_idx].size(3)
            if sw == tw:
                diff = source_features[src_idx] - self[i](target_features[tgt_idx])
            else:
                diff = F.interpolate(
                    source_features[src_idx],
                    scale_factor=tw / sw,
                    mode='bilinear'
                ) - self[i](target_features[tgt_idx])
            diff = diff.pow(2).mean(3).mean(2)
            if loss_weight is None and weight is None:
                diff = diff.mean(1).mean(0).mul(beta[i])
            elif loss_weight is None:
                diff = diff.mul(weight[i]).sum(1).mean(0).mul(beta[i])
            elif weight is None:
                diff = (diff.sum(1)*(loss_weight[i].squeeze())).mean(0).mul(beta[i])
            else:
                diff = (diff.mul(weight[i]).sum(1)*(loss_weight[i].squeeze())).mean(0).mul(beta[i])
            matching_loss = matching_loss + diff
        return matching_loss


class WeightNetwork(nn.ModuleList):
    def __init__(self, source_num_features, pairs):
        super(WeightNetwork, self).__init__()
        n = source_num_features
        for i, _ in pairs:
            self.append(nn.Linear(n[i], n[i]))
            self[-1].weight.data.zero_()
            self[-1].bias.data.zero_()
        self.pairs = pairs

    def forward(self, source_features):
        outputs = []
        for i, (idx, _) in enumerate(self.pairs):
            f = source_features[idx]
            f = F.avg_pool2d(f, f.size(2)).view(-1, f.size(1))
            outputs.append(F.softmax(self[i](f), 1))
        return outputs


class LossWeightNetwork(nn.ModuleList):
    def __init__(self, source_num_features, pairs, weight_type='relu', init=None):
        super(LossWeightNetwork, self).__init__()
        n = source_num_features
        if weight_type == 'const':
            self.weights = nn.Parameter(torch.zeros(len(pairs)))
        else:
            for i, _ in pairs:
                metanetw = nn.Linear(n[i], 1)
                if init is not None:
                    nn.init.constant_(metanetw.bias, init)
                self.append(metanetw)
        self.pairs = pairs
        self.weight_type = weight_type

    def forward(self, source_features):
        outputs = []
        if self.weight_type == 'const':
            for w in F.softplus(self.weights.mul(10)):
                outputs.append(w.view(1, 1))
        else:
            for i, (idx, _) in enumerate(self.pairs):
                f = source_features[idx]
                f = F.avg_pool2d(f, f.size(2)).view(-1, f.size(1))
                if self.weight_type == 'relu':
                    outputs.append(F.relu(self[i](f)))
                elif self.weight_type == 'relu-avg':
                    outputs.append(F.relu(self[i](f.div(f.size(1)))))
                elif self.weight_type == 'relu6':
                    outputs.append(F.relu6(self[i](f)))
        return outputs


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy_topk(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy_top1(outputs_data, labels):
    _, pred = torch.max(outputs_data, 1)
    correct = (pred == labels).sum().item()
    return correct / labels.size(0)


class WhatWhere():

    def __init__(self, source_model, target_model, device=None, beta=0.5,
                 loss_weight_type="relu6", loss_weight_init=1.0,
                 opt_lr=0.1, opt_momentum=0.9, opt_wd=1e-4, opt_nesterov=False,
                 optimizer="adam", meta_lr=1e-4, meta_wd=1e-4, opt_T=2,
                 folder_logs="logs", folder_saves="models"):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.folder_logs = folder_logs
        self.folder_saves = folder_saves
        os.makedirs(folder_logs, exist_ok=True)
        os.makedirs(folder_saves, exist_ok=True)

        self.source_model = source_model.to(self.device)
        self.target_model = target_model.to(self.device)
        self.source_loaders = None
        self.target_loaders = None
        self.source_num_features = self.source_model.extraction_num_features
        self.target_num_features = self.target_model.extraction_num_features
        if max(self.source_num_features) > self.source_model.max_num_features:
            self.source_num_features = [self.source_model.max_num_features
                                        if nb > self.source_model.max_num_features
                                        else nb for nb in self.source_num_features]
        if max(self.target_num_features) > self.target_model.max_num_features:
            raise ValueError("Selecting the best features in a layer is not allowed for the target model")
        self.pairs = _get_pairs(self.source_num_features, self.target_num_features)

        self.wnet = WeightNetwork(self.source_num_features, self.pairs).to(self.device)
        weight_params = list(self.wnet.parameters())
        self.lwnet = LossWeightNetwork(self.source_num_features, self.pairs,
                                       loss_weight_type, loss_weight_init).to(self.device)
        weight_params = weight_params + list(self.lwnet.parameters())
        self.target_branch = FeatureMatching(self.source_num_features,
                                             self.target_num_features,
                                             self.pairs).to(self.device)
        target_params = list(self.target_model.parameters()) + list(self.target_branch.parameters())
        self.beta = beta
        self.opt_T = opt_T

        if optimizer == "sgd":
            self.source_optimizer = optim.SGD(weight_params, lr=meta_lr,
                                              weight_decay=meta_wd,
                                              momentum=opt_momentum,
                                              nesterov=opt_nesterov)
        else:
            self.source_optimizer = optim.Adam(weight_params, lr=meta_lr,
                                               weight_decay=meta_wd)
        if meta_lr == 0:
            self.target_optimizer = optim.SGD(target_params, lr=opt_lr,
                                              momentum=opt_momentum,
                                              weight_decay=opt_wd)
        else:
            self.target_optimizer = sf_optim.MetaSGD(
                    target_params, [self.target_model, self.target_branch],
                    lr=opt_lr, momentum=opt_momentum, weight_decay=opt_wd,
                    rollback=True, cpu=opt_T>2)
        self.opt_epochs = None
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.target_optimizer, 200)
        self.last_save_epoch = 0
        self.best_acc = 0.0

    def save_parameters(self, name, is_best=False):
        state = {"target_model": self.target_model.state_dict(),
                 "target_branch": self.target_branch.state_dict(),
                 "target_optimizer": self.target_optimizer.state_dict(),
                 "w": self.wnet.state_dict(),
                 "lw": self.lwnet.state_dict(),
                 "source_optimizer": self.source_optimizer.state_dict(),
                 "scheduler": self.scheduler.state_dict(),
                 "nb_epochs": self.opt_epochs,
                 "current_epoch": self.last_save_epoch,
                 "best_accuracy": self.best_acc}
        torch.save(state, os.path.join(self.folder_saves, "ckpt_{}.pth".format(name)))
        if is_best:
            self.logger.info("Best model is saved")
        else:
            self.logger.info("Last model is saved")

    def load_parameters(self, name):
        ckpt = torch.load(os.path.join(self.folder_saves, name))
        self.target_model.load_state_dict(ckpt["target_model"])
        self.target_branch.load_state_dict(ckpt["target_branch"])
        self.target_optimizer.load_state_dict(ckpt["target_optimizer"])
        self.wnet.load_state_dict(ckpt["w"])
        self.lwnet.load_state_dict(ckpt["lw"])
        self.source_optimizer.load_state_dict(ckpt["source_optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.opt_epochs = ckpt["nb_epochs"]
        self.last_save_epoch = ckpt["current_epoch"]
        self.best_acc = ckpt["best_accuracy"]
        print("Model loaded : best accuracy = {:.2%} after {} epochs"\
                         .format(self.best_acc, self.last_save_epoch))

    def validate(self, loader_target, include_loss=False):
        acc = AverageMeter()
        loss = AverageMeter()
        self.target_model.eval()
        with torch.no_grad():
            for x, y in loader_target:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.target_model.forward(x)
                acc.update(accuracy_top1(y_pred.data, y))
                if include_loss:
                    loss.update(F.cross_entropy(y_pred, y))
        if include_loss:
            return acc.avg, loss.avg
        return acc.avg

    def inner_objective(self, data_source, data_target, matching_only=False):
        x, y = data_target[0].to(self.device), data_target[1].to(self.device)
        y_pred, target_features = self.target_model.forward_with_features(x)
        with torch.no_grad():
            source_features = self.source_model.forward_only_features(data_source[0].to(self.device))
        weights = self.wnet.forward(source_features)
        loss_weights = self.lwnet.forward(source_features)
        beta = [self.beta] * len(self.wnet)
        matching_loss = self.target_branch.forward(source_features,
                                                   target_features,
                                                   weights, beta, loss_weights)
        if matching_only:
            return matching_loss
        loss = F.cross_entropy(y_pred, y)
        return loss + matching_loss

    def outer_objective(self, data_target):
        x, y = data_target[0].to(self.device), data_target[1].to(self.device)
        y_pred = self.target_model.forward(x)
        loss = F.cross_entropy(y_pred, y)
        return loss

    def train(self, data_path, epochs=200, batch_size=64, mini_data=1.0,
              with_data_augmentation=True, early_stop=False, patience=20,
              save_last=True, save_best=True, save_name="best"):
        logging.basicConfig(format="[%(asctime)s] [%(name)s] %(message)s",
                            level=logging.INFO,
                            handlers=[logging.FileHandler(
                                    os.path.join(self.folder_logs, "{}.txt".format(save_name))),
                                      logging.StreamHandler(os.sys.stdout)])
        self.logger = logging.getLogger('main')
        self.logger.info(' '.join(os.sys.argv))
        csv_filename = os.path.join(self.folder_logs, "{}.csv".format(save_name))

        self.source_loaders = sf_load.get_dataset(
                data_path, self.source_model, mini_data=mini_data, stratify=True,
                batch_size=batch_size, with_data_augmentation=with_data_augmentation)
        self.target_loaders = sf_load.get_dataset(
                data_path, self.target_model, mini_data=mini_data, stratify=True,
                batch_size=batch_size, with_data_augmentation=with_data_augmentation)
        if self.opt_epochs is None:
            self.opt_epochs = epochs
            csv_memory = open(csv_filename, "w")  # Write the headers to the file
            writer = csv.writer(csv_memory)
            writer.writerow(["epoch", "validation_loss", "training_loss",
                             "validation_accuracy", "training_accuracy"])
            csv_memory.close()
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.target_optimizer, epochs)
        
        deb_epoch = self.last_save_epoch
        best_acc = self.best_acc
        lag_best = 0
        for epoch in range(deb_epoch, self.opt_epochs):
            self.target_model.train()
            self.source_model.eval()
            for i, (data_target, data_source) in enumerate(zip(self.target_loaders[0], self.source_loaders[0])):
                self.target_optimizer.zero_grad()
                self.inner_objective(data_source, data_target).backward()
                self.target_optimizer.step(None)
                for _ in range(self.opt_T):
                    self.target_optimizer.zero_grad()
                    self.target_optimizer.step(self.inner_objective,
                                               data_source, data_target, True)
                self.target_optimizer.zero_grad()
                self.target_optimizer.step(self.outer_objective, data_target)
                self.target_optimizer.zero_grad()
                self.source_optimizer.zero_grad()
                self.outer_objective(data_target).backward()
                self.target_optimizer.meta_backward()
                self.source_optimizer.step()
            self.scheduler.step()

            training_accuracy, training_loss = self.validate(self.target_loaders[1], include_loss=True)
            validation_accuracy, validation_loss = self.validate(self.target_loaders[2], include_loss=True)
            self.last_save_epoch = epoch + 1
            if validation_accuracy > best_acc:
                best_acc = validation_accuracy
                lag_best = 0
                self.best_acc = best_acc
                if save_best:
                    self.save_parameters(save_name, is_best=True)
            else:
                lag_best += 1

            if save_last:
                self.save_parameters("last_{}".format(save_name))
            self.logger.info("[Epoch : {}]  [Training accuracy = {:.2%}] [Validation accuracy = {:.2%}] [Best = {:.2%}]  [Training loss = {:.3f}] [Validation loss = {:.3f}]"\
                             .format(epoch+1, training_accuracy, validation_accuracy, best_acc,
                                     training_loss, validation_loss))
            csv_memory = open(csv_filename, "a")  # Write the results to the file
            writer = csv.writer(csv_memory)
            writer.writerow([epoch, validation_loss, training_loss,
                             validation_accuracy, training_accuracy])
            csv_memory.close()

            if early_stop & (lag_best >= patience):
                self.logger.info("Training stopped by early_stop")
                break
            # next epoch
        self.logger.info("Training is done")

    def evaluate(self, test_data=True, include_loss=False):
        if test_data:
            return self.validate(self.target_loaders[3], include_loss=include_loss)
        return self.validate(self.target_loaders[2], include_loss=include_loss)

    def predict_one(self, pil_img, with_data_augmentation=False):
        _, validation_transform = sf_load.get_preprocess(
                self.target_model, with_data_augmentation=with_data_augmentation)
        x = validation_transform(pil_img)
        x = x.unsqueeze(0)
        x = x.to(self.device)
        out = self.target_model.forward(x)
        y_pred = nn.Softmax(out)
        return y_pred

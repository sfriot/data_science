# -*- coding: utf-8 -*-
"""
Created on Wed Nov 4 2020

@author: Sylvain Friot
Based on https://github.com/alinlab/L2T-ww/blob/master/train_l2t_ww.py
"""

import os
import logging
import torch
import sf_pytorch_loaders as sf_load

torch.backends.cudnn.benchmark = True


class BestFeatures():

    def __init__(self, model, data_path, batch_size=64,
                 with_data_augmentation=False, mini_data=1.0,
                 folder_logs="logs", folder_saves="models", log_name="bestfeat"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.folder_logs = folder_logs
        self.folder_saves = folder_saves
        os.makedirs(folder_logs, exist_ok=True)
        os.makedirs(folder_saves, exist_ok=True)
        logging.basicConfig(format="[%(asctime)s] [%(name)s] %(message)s",
                            level=logging.INFO,
                            handlers=[logging.FileHandler(
                                    os.path.join(folder_logs, "{}.txt".format(log_name))),
                                      logging.StreamHandler(os.sys.stdout)])
        self.logger = logging.getLogger('main')
        self.logger.info(' '.join(os.sys.argv))

        self.model = model.to(self.device)
        self.loaders = sf_load.get_dataset(data_path, model,
                                           mini_data=mini_data, stratify=True,
                                           batch_size=batch_size,
                                           with_data_augmentation=with_data_augmentation)

    def calcul_best_features(self, max_num_features=None, debug=False):
        self.model._define_best_features(self.loaders[1],
                                         max_num_features=max_num_features,
                                         to_device=self.device,
                                         debug=debug)
        self.logger.info("Calculation is done")

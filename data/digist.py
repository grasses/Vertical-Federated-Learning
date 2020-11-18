#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/10/10, homeway'

import copy
import torch
import torchvision
import numpy as np


class Data(object):
    def __init__(self, split, train=True, transform=None, batch_size=128, shuffle=True, nThreads=10):
        self.data_loader = {}
        self.num_steps = 0
        self.nThreads = nThreads
        self.transform = transform
        self.batch_size = batch_size
        self.dataset = torchvision.datasets.MNIST("data/mnist", train=train,
                                                  transform=self.transform,
                                                  target_transform=None,
                                                  download=True)

        # shuffle dataset manually
        if shuffle:
            size = len(self.dataset.data)
            idx = np.random.choice(size, size, replace=False)
            self.dataset.data = self.dataset.data[idx]
            self.dataset.targets = self.dataset.targets[idx]

        # split data into two parts
        self.split_vertical_data(split)
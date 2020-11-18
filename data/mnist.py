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

    def split_vertical(self, dataset, split=10):
        # copy dataset
        A_dataset = copy.deepcopy(dataset)
        B_dataset = copy.deepcopy(dataset)

        # split dataset
        A_dataset.data = A_dataset.data.reshape([-1, 784])[:, :split]
        B_dataset.data = B_dataset.data.reshape([-1, 784])[:, split:]
        B_dataset.target = (B_dataset.targets * 0).long()

        print(A_dataset.data.shape, B_dataset.data.shape)
        return A_dataset, B_dataset

    def split_vertical_data(self, split):
        # split dataset into two A_dataset and B_dataset
        datasets = self.split_vertical(self.dataset, split=split)

        # TODO: PSI operation
        # build data loader
        for uid, dataset in enumerate(datasets):
            print("-> data in party:{}, size:{}".format(uid, len(dataset.targets)))
            self.data_loader[uid] = torch.utils.data.DataLoader(dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=False,
                                                      num_workers=self.nThreads)
            # update num_steps/per round
            self.num_steps = len(self.data_loader[uid])


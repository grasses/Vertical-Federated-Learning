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
        # split data into two parts
        self.shuffle = shuffle
        self.split_vertical_data(split)

    def load_dataset(self, train=True):        
        dataset = torchvision.datasets.MNIST("data/mnist", train=train,
                                                  transform=self.transform,
                                                  target_transform=None,
                                                  download=True)

        # shuffle dataset manually
        if self.shuffle:
            size = len(dataset.data)
            idx = np.random.choice(size, size, replace=False)
            dataset.data = dataset.data[idx]
            dataset.targets = dataset.targets[idx]
        return dataset


    def split_vertical(self, split):
        # copy dataset
        A_dataset = self.load_dataset()
        B_dataset = self.load_dataset()

        # split dataset
        A_dataset.data = A_dataset.data.reshape([-1, 784])[:, :split]
        B_dataset.data = B_dataset.data.reshape([-1, 784])[:, split:]
        B_dataset.target = (B_dataset.targets * 0).long()
        return A_dataset, B_dataset

    def split_vertical_data(self, split):
        # split dataset into two A_dataset and B_dataset
        datasets = self.split_vertical(split=split)

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


#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/10/7, homeway'

import os
import random
import torch
import numpy as np
import datetime

seed = 100
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
format_time = str(datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))


class Party():
    def __init__(self, uid, num_feature=10, num_output=1, lr=0.01):
        self.uid = uid
        self.num_features = num_feature
        self.num_output = num_output
        self.public_key = None
        self.secret_key = None
        self.lr = lr

class Conf(object):
    # machine learning
    batch_size = 100
    momentum = 0.8
    learning_rate = 1e-3
    debug = True

    num_round = 1000
    num_steps = -1
    num_features = 784
    num_classes = 10

    fed_vertical = {
        "party": [
            Party(uid=0, num_feature=300, num_output=1, lr=0.001),
            Party(uid=1, num_feature=484, num_output=1, lr=0.001),
        ],
        "split": 30,
        "num_steps": -1,  # update in dataloader
        "attack_steps": [1]  # selected attack steps
    }
    fed_clients = {}

    device = "cpu:0"
    dataset = "mnist"
    scope_name = f"{dataset}_vertical_{format_time}"
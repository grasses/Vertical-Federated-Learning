#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/6/1, homeway'

import torch
import numpy as np
from torch.autograd import Variable
#from model import Model


class Party(torch.nn.Module):
    def __init__(self, num_features, num_output, weight=None, debug=False):
        super(Party, self).__init__()
        self.debug = debug
        self.w = self.xavier_init([num_features, num_output])
        if weight is not None:
            print(f"-> w={weight}\n-> w.shape={weight.shape}\n\n")
            self.w = torch.nn.Parameter(Variable(weight), requires_grad=True)

    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / np.sqrt(in_dim / 2.)
        return torch.nn.Parameter(Variable(torch.randn(*size) * xavier_stddev, requires_grad=True))

    def grad_step1(self, x, y):
        """
        Implement of algorithm3 in paper.
        x = [batch_size, 13]
        self.w = [13, out_size]
        y = [batch_size, out_size]
        t = [batch_size, out_size]
        u_prime = [batch_size, out_size]
        """
        t = 0.25 * (x @ self.w)
        u_prime = t - 0.5 * y
        if self.debug:
            print(f"-> u_prime={u_prime}\n\n\n")
        return u_prime

    def grad_step2(self, x, u_prime):
        """
        Implement of algorithm3 in paper.
        x = [batch_size, 10]
        self.w = [10, out_size]
        w = [batch_size, out_size]
        u_prime = [batch_size, out_size]
        z = [10, out_size]
        """
        v = 0.25 * (x @ self.w)
        w = u_prime + v
        z = w.t() @ x
        if self.debug:
            print(f"-> v={v}")
            print(f"-> w={w}")
            print(f"-> z={z}\n\n")
        return w, z

    def grad_step3(self, x, w):
        """
        Implement of algorithm3 in paper.
        x = [batch_size, 13]
        w = [batch_size, out_size]
        z_prime = [13, out_size]
        """
        z_prime = w.t() @ x
        if self.debug:
            print(f"-> z_prime={z_prime}\n\n")
        return z_prime

    def loss_step1(self, x, y):
        """
        Implement of algorithm5 in paper.
        x = [batch_size, 13]
        y = [batch_size, output_size]
        u = [batch_size, out_size]
        u_prime = [out_size, out_size]
        u_a = [1, 13]
        """
        u = x @ self.w
        u_prime = 0.125 * (u.t() @ u).t()
        return u, u_prime

    def loss_step2(self, x, u, u_prime):
        """
        Implement of algorithm5 in paper.
        x = [batch_size, 10]
        y = [batch_size, out_size]
        u_prime = [out_size, out_size]
        v_prime = [out_size, out_size]
        w = [out_size, out_size]
        u_b = [1, 10]
        v = [batch_size, out_size]
        """
        v = x @ self.w
        v_prime = 0.125 * (v.t() @ v).t()
        print(f"v_prime={v_prime.shape} u_prime={u_prime.shape}")
        w = u_prime + v_prime + 0.25 * (v.t() @ u)
        return v, w

    def loss_step3(self, y, u, v):
        """
        Implement of algorithm5 in paper.
        y = [batch_size, out_size]
        u = [batch_size, out_size]
        v = [batch_size, out_size]
        uv = [out_size, batch_size]
        """
        uv = -0.5 * (y.float().t() @ (u + v))
        return uv

    def forward(self, x):
        """
        feed forward
        """
        x = x @ self.w
        return x


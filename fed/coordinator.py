#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/10/10, homeway'

import os
import json
import torch
from utils.helper import Helper
from utils.model import Model

class Coordinator(object):
    def __init__(self, conf, data, *kwargs):
        self.conf = conf
        self.data = data
        self.model = {}
        self.optimizer = {}
        self.fed_clients = self.conf.fed_clients
        self.party = self.conf.fed_vertical["party"]
        self.init(*kwargs)

    def init(self, *weight):
        print("-> initial server")
        self.helper = Helper(self.conf)

        # creat homomorphic encryption key pair
        # TODO

        # init weight
        if not len(weight):
            weight = [None, None]

        # setup model
        for uid, party in enumerate(self.party):
            self.model[uid] = Model(party.num_features, party.num_output, weight=weight[uid]).to(self.conf.device)
            self.optimizer[uid] = torch.optim.SGD(self.model[uid].parameters(), lr=self.conf.learning_rate, momentum=self.conf.momentum)

    def get_parameters(self):
        """
        :return: parameters {uid: state_dict}
        """
        parameters = {0: None, 1: None}
        for uid, party in enumerate(self.party):
            parameters[uid] = self.model[uid].state_dict()
        return parameters

    def run(self, preview=10):
        result_dict = {"loss": [], "acc": [], "step": []}
        total_step = self.conf.num_round * self.conf.num_steps

        for step in range(total_step):
            print(f"\n\n\n<-------------------------- Step: [{step}/{total_step}] -------------------------->")

            # start round for all parties
            for uid, party in enumerate(self.party):
                self.fed_clients[uid].start_round(self.get_parameters())

            # three steps for grad
            u_prime = self.fed_clients[0].grad_step1()
            w, z = self.fed_clients[1].grad_step2(u_prime)
            grad = self.fed_clients[0].grad_step3(w, z)


            # update model
            for uid, party in self.party.items():
                self.model[uid].set_grad([grad[uid]])
                self.optimizer[uid].step()

            # stop round for all parties
            for uid, party in self.party.items():
                self.fed_clients[uid].stop_round()

            # calculate loss
            '''
            u, u_prime = self.fed_clients[0].loss_step1()
            v, w = self.fed_clients[1].loss_step2(u, u_prime)
            loss = self.fed_clients[0].loss_step3(u, v, w)
            print(f"-> [{step}/{total_step}] loss={loss.cpu().detach().numpy()}")
            '''

            # run a batch forward
            if step % preview == 0:
                # print(f"-> grad0={grad[0]}\n-> grad1={grad[1]}\n")
                logists = []
                for uid, party in self.party.items():
                    logists.append(self.fed_clients[uid].forward().float())
                logists = logists[0] + logists[1]  # torch.sigmoid(logists[0] + logists[1])
                self.fed_clients[0].batch_evaluation(logists, step, result=result_dict)

            if step > 0 and step % preview == 0:
                with open(os.path.join(self.conf.output_path, "summary.json"), "w") as outfile:
                    json.dump(result_dict, outfile)

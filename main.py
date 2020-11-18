#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/10/10, homeway'


from torchvision import datasets, transforms
from utils.helper import Helper
from fed import Client, Coordinator as Server


def main():
    hepler = Helper()
    conf = hepler.conf()

    # load splitted dataset
    if conf.dataset == "mnist":
        from data.mnist import Data
    elif conf.dataset == "digist":
        from data.digist import Data

    # build dataset & dataloader
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    data = Data(split=300, transform=train_transform, train=True)
    conf.num_steps = data.num_steps

    # load federated learning client & coordinator
    for uid, party in enumerate(conf.fed_vertical["party"]):
        conf.fed_clients[uid] = Client(uid, conf, party, data.data_loader[uid])
    server = Server(conf, data)
    server.run()


if __name__ == '__main__':
    main()
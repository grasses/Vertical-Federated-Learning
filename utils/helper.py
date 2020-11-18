# !/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2020/5/31, homeway'

import os
import copy
import collections
import argparse
from config.conf import Conf
import numpy as np
import shutil

class Helper(object):
    def __init__(self, conf=None):
        self.conf = conf
        self.args = self.get_args()
        if not conf:
            self.conf = self.init()

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default="conf")
        return parser.parse_args()

    def init(self):
        train_flag = True
        try:
            self.conf = __import__("config.{:s}".format(self.args.config), globals(), locals(), ["Conf"]).Conf
            print("-> load config file: config.{}.py".format(self.args.config))
        except:
            train_flag = False
            self.conf = __import__(self.args.config, globals(), locals(), ["Conf"]).Conf
            self.conf.scope_name = self.args.config.split(".")[-2]
            print("-> load config file: {}.py".format(self.args.config))

        # convert list/dict to object
        '''
        for key in dir(self.conf):
            if not key[:2] == "__":
                setattr(self.conf, key, self.to_obj(getattr(self.conf, key)))
        '''

        # set config's attributes
        ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
        setattr(self.conf, "ROOT", ROOT)
        setattr(self.conf, "data_root", os.path.join(ROOT, "data"))
        setattr(self.conf, "data_path", os.path.join(self.conf.data_root, self.conf.dataset))
        setattr(self.conf, "output_root", os.path.join(ROOT, f"output/{self.conf.scope_name}"))
        setattr(self.conf, "output_path", os.path.join(ROOT, f"output/{self.conf.scope_name}/output"))
        setattr(self.conf, "model_path", os.path.join(ROOT, f"output/{self.conf.scope_name}/model"))

        if train_flag:
            # build setup path
            output = os.path.join(self.conf.ROOT, "output")
            init_path = [output, self.conf.output_path, self.conf.model_path]
            for path in init_path:
                if not os.path.exists(path):
                    os.makedirs(path)

            # copy config file
            conf_path = os.path.join(ROOT, "config/{:s}.py".format(self.args.config))
            if os.path.exists(conf_path):
                shutil.copyfile(conf_path, os.path.join(self.conf.output_path, "conf.py"))
        return self.conf
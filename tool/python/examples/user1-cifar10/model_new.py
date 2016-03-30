#!/usr/bin/env python

#/************************************************************
#*
#* Licensed to the Apache Software Foundation (ASF) under one
#* or more contributor license agreements.  See the NOTICE file
#* distributed with this work for additional information
#* regarding copyright ownership.  The ASF licenses this file
#* to you under the Apache License, Version 2.0 (the
#* "License"); you may not use this file except in compliance
#* with the License.  You may obtain a copy of the License at
#*
#*   http://www.apache.org/licenses/LICENSE-2.0
#*
#* Unless required by applicable law or agreed to in writing,
#* software distributed under the License is distributed on an
#* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#* KIND, either express or implied.  See the License for the
#* specific language governing permissions and limitations
#* under the License.
#*
#*************************************************************/

import os, sys
import numpy as np

current_path_ = os.path.dirname(__file__)
singa_root_=os.path.abspath(os.path.join(current_path_,'../../../..'))
sys.path.append(os.path.join(singa_root_,'tool','python'))

from singa.driver import Driver
from singa.layer import *
from singa.model import *
from singa.utils.utility import swap32

neuralnet = [] # neural net (hidden layers)
#shape,inputfile,src
neuralnet.append(data1=Dummy(shape=[10000,3,32,32],path="examples/cifar10/cifar-10-batches-bin",src=[]))
neuralnet.append(data2=Dummy(shape=[10000,1],path="tool/python/examples/datasets/cifar10_mean_image",src=[]))
neuralnet.append(c1=Convolution2D(32, 5, 1, 2, w_std=0.0001, b_lr=2,src=[data1]))
neuralnet.append(p1=MaxPooling2D(pool_size=(3,3), stride=2,src=[c1]))
neuralnet.append(a1=Activation('relu',src=[p1]))
neuralnet.append(l1=LRN2D(3, alpha=0.00005, beta=0.75,src=[a1]))
neuralnet.append(c2=Convolution2D(32, 5, 1, 2, b_lr=2,src=[l1]))
neuralnet.append(a2=Activation('relu',src=[c2]))
neuralnet.append(p2=AvgPooling2D(pool_size=(3,3), stride=2,src=[a2]))
neuralnet.append(l2=LRN2D(3, alpha=0.00005, beta=0.75,src=[p2]))
neuralnet.append(c3=Convolution2D(64, 5, 1, 2,src=[l2]))
neuralnet.append(a3=Activation('relu',src=[c3]))
neuralnet.append(p3=AvgPooling2D(pool_size=(3,3), stride=2,src=[a3]))
neuralnet.append(d=Dense(10, w_wd=250, b_lr=2, b_wd=0,src=[p3]))
neuralnet.append(loss=Loss('softmaxloss',src=[d,data2]))

#algorithm
updater = SGD(decay=0.004, momentum=0.9, lr_type='manual', step=(0,60000,65000), step_lr=(0.001,0.0001,0.00001))

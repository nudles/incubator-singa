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

import os, sys, string
import numpy as np

current_path_ = os.path.dirname(__file__)
singa_root_=os.path.abspath(os.path.join(current_path_,'../../..'))
sys.path.append(os.path.join(singa_root_,'tool','python'))

from singa.driver import Driver
from singa.layer import *
from singa.model import *
from singa.utils.utility import swap32
from google.protobuf.text_format import Merge

'''
Example of MLP with MNIST dataset
'''

def load_dataset():
    '''
       train-images: 4 int32 headers & int8 pixels
       train-labels: 2 int32 headers & int8 labels
    '''
    print '[Loading MNIST dataset]'
    fname_train_data = "examples/mnist/train-images-idx3-ubyte"
    fname_train_label = "examples/mnist/train-labels-idx1-ubyte"
    info = swap32(np.fromfile(fname_train_data, dtype=np.uint32, count=4))
    nb_samples = info[1] 
    shape = (info[2],info[3])
    
    x = np.fromfile(fname_train_data, dtype=np.uint8)
    x = x[4*4:] # skip header 
    x = x.reshape(nb_samples, shape[0]*shape[1]) 
    print '   data x:', x.shape
    y = np.fromfile(fname_train_label, dtype=np.uint8)
    y = y[4*2:] # skip header
    y = y.reshape(nb_samples, 1) 
    print '  label y:', y.shape
    return x, y

#-------------------------------------------------------------------
print '[Layer registration/declaration]'
d = Driver()
d.Init(sys.argv)

input = Dummy()
label = Dummy()

nn = []
nn.append(input)
nn.append(Dense(2500, init='uniform'))
nn.append(Activation('stanh'))
nn.append(Dense(2000, init='uniform'))
nn.append(Activation('stanh'))
nn.append(Dense(1500, init='uniform'))
nn.append(Activation('stanh'))
nn.append(Dense(1000, init='uniform'))
nn.append(Activation('stanh'))
nn.append(Dense(500, init='uniform'))
nn.append(Activation('stanh'))
nn.append(Dense(10, init='uniform'))
loss = Loss('softmaxloss')

# updater
sgd = SGD(lr=0.001, lr_type='step')

#-------------------------------------------------------------------
print '[Start training]'
batchsize = 64 
disp_freq = 10

x, y = load_dataset()

for i in range(x.shape[0] / batchsize):
    xb, yb = x[i*batchsize:(i+1)*batchsize,:], y[i*batchsize:(i+1)*batchsize,:]
    nn[0].Feed(xb)
    label.Feed(yb, is_label=1)
    for h in range(1, len(nn)):
        nn[h].ComputeFeature(nn[h-1])
    loss.ComputeFeature(nn[-1], label)
    if (i+1)%disp_freq == 0:
        print '  Step {:>3}: '.format(i+1),
        loss.display()
    loss.ComputeGradient(i+1, sgd)

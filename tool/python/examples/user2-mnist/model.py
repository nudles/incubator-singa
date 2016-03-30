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

from singa.layer import *
from singa.model import *

input = Dummy()

neuralnet = [] # neural net (hidden layers)
neuralnet.append(input)
neuralnet.append(Dense(2500, init='uniform'))
neuralnet.append(Activation('stanh'))
neuralnet.append(Dense(2000, init='uniform'))
neuralnet.append(Activation('stanh'))
neuralnet.append(Dense(1500, init='uniform'))
neuralnet.append(Activation('stanh'))
neuralnet.append(Dense(1000, init='uniform'))
neuralnet.append(Activation('stanh'))
neuralnet.append(Dense(500, init='uniform'))
neuralnet.append(Activation('stanh'))
neuralnet.append(Dense(10, init='uniform'))
loss = Loss('softmaxloss')

updater = SGD(lr=0.001, lr_type='step')

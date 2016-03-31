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

from model import neuralnet, updater
from singa.driver import Driver
from singa.layer import *
from singa.model import save_model_parameter, load_model_parameter 
from singa.utils.utility import swap32

'''
Example of CNN with cifar10 dataset
'''
def train(batchsize,disp_freq,check_freq,train_step,workspace,checkpoint=None):
    print '[Layer registration/declaration]'
    # TODO change layer registration methods
    d = Driver()
    d.Init(sys.argv)

    print '[Start training]'

    #if need to load checkpoint
    if checkpoint:
        load_model_parameter(workspace+checkpoint, neuralnet, batchsize)
   
    for i in range(0,train_step):       
    
        for h in range(len(neuralnet)):
            #Fetch data for input layer
            if neuralnet[h].layer.type==kDummy:
                neuralnet[h].FetchData(batchsize)
            else:
                neuralnet[h].ComputeFeature()
    
        neuralnet[-1].ComputeGradient(i+1, updater)
    
        if (i+1)%disp_freq == 0:
            print '  Step {:>3}: '.format(i+1),
            neuralnet[h].display()
    
        if (i+1)%check_freq == 0:   
            save_model_parameter(i+1, workspace, neuralnet)


    print '[Finish training]'

if __name__=='__main__':
    if len(sys.argv) < 5:
        print "argv should be more than 5"
    if len(sys.argv) > 5:
        checkpoint = sys.argv[5]
    else:
        checkpoint = None
    print checkpoint
    train(
          batchsize = int(sys.argv[1]), 
          disp_freq = int(sys.argv[2]),
          check_freq = int(sys.argv[3]), 
          train_step = int(sys.argv[4]),
          workspace = '/workspace',
          checkpoint = checkpoint,
          )



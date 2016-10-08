# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
''' This model is created following the structure from
https://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layers-18pct.cfg
Following the same setting for hyper-parameters and data pre-processing, the final
validation accuracy would be about 82%.
'''

# sys.path.append(os.path.join(os.path.dirname(__file__), '../../build/python'))
from singa import layer
from singa import metric
from singa import loss
from singa import net as ffnet


def create_net(use_cpu=False):
    if use_cpu:
        layer.engine = 'singacpp'

    net = ffnet.FeedForwardNet(loss.SoftmaxCrossEntropy(), metric.Accuracy())
    W0_filler = {'init': 'xavier', 'std': 0.1}
    b0_filler = {'init': 'constant', 'value': 0.2}
    
	input_shape = (3, 244, 244)
	
    # Conv 1
    net.add(layer.Conv2D('conv1', 64, 11, 4, pad=2, W_specs=W0_filler.copy(), b_specs=b0_filler.copy(), input_sample_shape=input_shape))
    net.add(layer.Activation('relu1'))
    net.add(layer.MaxPooling2D('pool1', 3, 2))
    
    # Conv 2
    net.add(layer.Conv2D('conv2', 192, 5, 1, pad=2, W_specs=W0_filler.copy(), b_specs=b0_filler.copy()))
    net.add(layer.Activation('relu2'))
    net.add(layer.MaxPooling2D('pool2', 3, 2))
    
    # Conv 3
    net.add(layer.Conv2D('conv3', 384, 3, 1, pad=1, W_specs=W0_filler.copy(), b_specs=b0_filler.copy()))
    net.add(layer.Activation('relu3'))
    
    # Conv 4
    net.add(layer.Conv2D('conv4', 256, 3, 1, pad=1, W_specs=W0_filler.copy(), b_specs=b0_filler.copy()))
    net.add(layer.Activation('relu4'))
    
    # Conv 5
    net.add(layer.Conv2D('conv5', 256, 3, 1, pad=1, W_specs=W0_filler.copy(), b_specs=b0_filler.copy()))
    net.add(layer.Activation('relu5'))
    net.add(layer.MaxPooling2D('pool5', 3, 2))
    
    # L2 Norm -> Inner product
    # Flatten here maybe?
    #net.add(layer.Flatten('flat'))
    net.add(layer.Dense('fc6', 4096))
    net.add(layer.Activation('relu6'))
    
    net.add(layer.Dense('fc7', 4096))
    net.add(layer.Activation('relu7'))
    
    net.add(layer.Dense('fc8', 1000))

    for (p, specs) in zip(net.param_values(), net.param_specs()):
        filler = specs.filler
        if filler.type == 'gaussian':
            p.gaussian(filler.mean, filler.std)
        else:
            p.set_value(0)
        print specs.name, filler.type, p.l1()

    return net

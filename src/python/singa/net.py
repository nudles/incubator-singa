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
"""
Nerual net class for constructing the nets using layers and providing access
functions for net info, e.g., parameters.
"""


class FeedForwardNet(object):
    def __init__(self, loss=None, metric=None):
        self.loss = loss
        self.metric = metric
        self.layers = []
        self.param_names = []
        self.param_values = []
        self.param_specs = []

    def to_device(self, dev):
        for lyr in self.layers:
            lyr.to_device(dev)

    def add(self, lyr):
        """Append a layer into the layer list.

        This function will get the sample shape from the last layer to setup
        the newly added layer. For the first layer, it is setup outside.
        The calling function should ensure the correctness of the layer order.

        Args:
            lyr (Layer): the layer to be added
        """
        if len(self.layers) > 0 and lyr.has_setup() is False:
            shape = self.layers[-1].get_output_sample_shape()
            lyr.setup(shape)
        self.layers.append(lyr)
        self.param_names.extend(lyr.param_names())
        self.param_values.extend(lyr.param_values())
        self.param_specs.extend(lyr.param_specs())

    def init_params(self):
        """Init the parameter values according to param specs"""
        for (spec, p) in zip(self.param_specs, self.param_values):
            init = initializer.create_initializer(spec.filler)
            init.apply(p)

    def train(self, x, y):
        out = self.forward(kTrain, x)
        l = self.loss.forward(out, y)
        if self.metric is not None:
            m = self.metric.evaluate(out, y)
        return self.backward(), (l, m)

    def evaluate(self, x, y):
        """Evaluate the loss and metric of the given data"""
        out = self.forward(kEval, x)
        l = None
        m = None
        assert self.loss is not None or self.metric is not None,\
            'Cannot do evaluation, as neither loss nor metic is set'
        if self.loss is not None:
            l = self.loss.evaluate(out, y)
        if self.metric is not None:
            m = self.metric.evaluate(out, y)
        return l, m

    def predict(self, x):
        return self.forward(kEval, x)

    def forward(self, flag, x):
        for lyr in self.layers:
            x = lyr.Forward(flag, x)
        return x

    def backward(self, flag=kTrain):
        grad = self.loss.Backward()
        pgrads = []
        for lyr in reversed(self.layers):
            grad, _pgrads = lyr.Backward(flag, grad)
            for g in reversed(_pgrads):
                pgrads.append(g)
        return (self.param_values, pgrads.reverse())

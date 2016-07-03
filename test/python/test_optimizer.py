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
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# =============================================================================
import sys
import os
import unittest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../build/python'))

import singa.tensor as tensor
import singa.optimizer as opt
import singa.device as device


class TestOptimizer(unittest.TestCase):

    def SetUp(self):
        self.np_W = np.array([0.1, 0.2, 0.3, 0.4])
        self.W = tensor.from_numpy(self.np_W)
        self.np_g = np.array([0.1, 0.3, 0.1, 0.2])
        self.g = tensor.from_numpy(self.np_g)

    def to_cuda(self):
        cuda = device.CudaGPU()
        self.W.to_device(cuda)
        self.g.to_device(cuda)

    def test_sgd(self):
        lr = 0.1
        sgd = opt.SGD(lr)
        sgd.apply(0, self.g, self.W)
        w = tensor.to_numpy(self.W)
        for i in range(self.W.size()):
            self.assertEqual(w[i], self.np_W[i] - lr * self.np_g[i])

    def test_sgd_cuda(self):
        lr = 0.1
        sgd = opt.SGD(lr)
        self.to_cuda()
        sgd.apply(0, self.g, self.W)
        self.W.to_host()
        w = tensor.to_numpy(self.W)
        for i in range(self.W.size()):
            self.assertEqual(w[i], self.np_W[i] - lr * self.np_g[i])

    def test_constraint(self):
        threshold = 0.02
        cons = opt.L2Constraint(threshold)
        cons.apply(0, self.W, self.g)
        g = tensor.to_numpy(self.g)
        nrm = np.linalg.norm(self.np_g)
        for i in range(g.size):
            self.assertEqual(g[i], self.np_g[i] * threshold / nrm)

    def test_constraint_cuda(self):
        threshold = 0.02
        self.to_cuda()
        cons = opt.L2Constraint(threshold)
        cons.apply(0, self.W, self.g)
        self.g.to_host()
        g = tensor.to_numpy(self.g)
        nrm = np.linalg.norm(self.np_g)
        for i in range(g.size):
            self.assertEqual(g[i], self.np_g[i] * threshold / nrm)

    def test_regularizer(self):
        coefficient = 0.0001
        reg = opt.L2Regularizer(coefficient)
        reg.apply(0, self.W, self.g)
        g = tensor.to_numpy(self.g)
        for i in range(g.size):
            self.assertEqual(g[i], self.np_g[i] + coefficient * self.w[i])

    def test_regularizer_cuda(self):
        coefficient = 0.0001
        reg = opt.L2Regularizer(coefficient)
        self.to_cuda()
        reg.apply(0, self.W, self.g)
        self.g.to_host()
        g = tensor.to_numpy(self.g)
        for i in range(g.size):
            self.assertEqual(g[i], self.np_g[i] + coefficient * self.w[i])


if __name__ == '__main__':
    unittest.main()

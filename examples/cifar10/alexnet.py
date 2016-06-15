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
""" CIFAR10 dataset is at https://www.cs.toronto.edu/~kriz/cifar.html.
It includes 5 binary dataset, each contains 10000 images. 1 row (1 image)
includes 1 label & 3072 pixels.  3072 pixels are 3 channels of a 32x32 image
"""

import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src/python'))
import layer
import tensor
import device
import optimizer
import metric
import loss
import cPickle
import initializer


def load_dataset(filepath):
    print 'Loading data file %s' % filepath
    with open(filepath, 'rb') as fd:
        cifar10 = cPickle.load(fd)
    image = cifar10['data'].astype(dtype=np.uint8)
    label = np.asarray(cifar10['labels'], dtype=np.uint8)
    label = label.reshape(label.size, 1)
    return image, label


def load_train_data(dir_path):
    labels = []
    batchsize = 10000
    images = np.empty((5 * batchsize, 3, 32, 32), dtype=np.uint8)
    for did in range(1, 6):
        fname_train_data = dir_path + "/data_batch_{}".format(did)
        image, label = load_dataset(fname_train_data)
        images[(did - 1) * batchsize:did * batchsize] = image
        labels.extends(label)
    images = np.array(images, dtype=np.float)
    labels = np.array(labels, dtype=np.int)
    return images, labels


def load_test_data(dir_path):
    images, labels = load_dataset(dir_path + "/test_batch")
    return np.array(images, dtype=np.float), np.array(labels, dtype=np.int)


def create_net(device):
    net = FeedForwardNetNet(loss.SoftmaxCrossEntropy(), metric.Accuracy())
    W0_specs = {'init': 'gaussian', 'std': 0.0001}
    W1_specs = {'init': 'gaussian', 'std': 0.01}
    W2_specs = {'init': 'gaussian', 'std': 0.01, 'decay_mult': 250}
    b_specs = {'init': 'constant', 'value': 0, 'lt_mult': 2}
    net.add(layer.Convolution('conv1', 32, 5, 1, W0_specs, b_specs, pad=2))
    net.add(layer.MaxPooling2D('pool1', 3, 2, pad=1))
    net.add(layer.Activation('relu1'))
    net.add(layer.LRN(name='lrn1'))
    net.add(layer.Convolution('conv2', 32, 5, 1, W1_specs, b_specs, pad=2))
    net.add(layer.Activation('relu2'))
    net.add(layer.MaxPooling2D('pool2', 3, 2, pad=1))
    net.add(layer.LRN('lrn2'))
    net.add(layer.Convolution('conv3', 64, 5, 1, W1_specs, b_specs, pad=2))
    net.add(layer.Activation('relu3'))
    net.add(layer.MaxPooling2D('pool3', 3, 2, pad=1))
    net.add(layer.Flatten('flat'))
    net.add(layer.Dense('dense', 10, W2_specs, b_specs))
    net.to_device(device)
    return net


def get_lr(epoch):
    if epoch < 120:
        return 0.001
    elif epoch < 130:
        return 0.0001
    elif epoch < 140:
        return 0.00001


def train(data_dir, net, num_epoch=140, batch_size=100):
    net.init_params()
    opt = optimizer.SGD(get_lr, momentum=0.9, weight_decay=0.004)
    for (p, specs) in zip(net.param_values, net.param_specs):
        opt.register(specs, p)
    train_x, train_y = load_train_data(data_dir)
    test_x, test_y = load_test_data(data_dir)
    mean = np.average(train_x, axis=0)
    train_x -= mean
    test_x -= mean

    dev = device.CudaGPU()
    tx = tensor.Tensor((batch_size, 3, 32, 32), dev)
    ty = tensor.Tensor((batch_size,), dev)
    num_train_batch = train_x.shape[0] / batch_size
    num_test_batch = test_x.shape[0] / batch_size
    for epoch in range(num_epoch):
        loss, acc = 0.0, 0.0
        print 'Epoch %d' % epoch
        for b in range(num_train_batch):
            x = train_x[b * batch_size: (b + 1) * batch_size]
            y = train_y[b * batch_size: (b + 1) * batch_size]
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            (params, grads), (l, a) = net.train(tx, ty)
            loss += l
            acc += a
            for (p, g) in zip(params, grads):
                opt.apply(epoch, p, g)
            # update progress bar
            info = 'training loss = %f, training accuracy = %f' \
                % (epoch, loss / num_train_batch, acc / num_train_batch)

            utils.update_progress(float(b) / float(num_train_batch), info)

        loss, acc = 0.0, 0.0
        for b in range(num_test_batch):
            x = test_x[b * batch_size: (b + 1) * batch_size]
            y = test_y[b * batch_size: (b + 1) * batch_size]
            tx.from_array(x)
            ty.from_array(y)
            l, a = net.evaluate(x, y)
            loss += l
            acc += a

        print 'test loss = %f, test accuracy = %f' \
            % (epoch, loss / num_test_batch, acc / num_test_batch)


if __name__ == '__main__':
    data_dir = 'cifar-10-batches-py'
    assert os.path.exists(data_dir), \
        'Pls download the cifar10 dataset via "download_data.py py"'
    net = create_net()
    train(data_dir, net)

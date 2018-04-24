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

"""
Train mnist, see more explanation at http://mxnet.io/tutorials/python/mnist.html
"""
import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, fit
from common.util import download_file
import mxnet as mx
import numpy as np
import gzip, struct

def read_data(label, image):
    """
    download and read data into numpy
    """
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    with gzip.open(download_file(base_url+label, os.path.join('data',label))) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(download_file(base_url+image, os.path.join('data',image)), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)


def to4d(img):
    """
    reshape to 4D arrays
    """
    return img.reshape(img.shape[0], 1, 32, 32).astype(np.float32)/255

def swap(data):
    """
    reshape to 2nd and 4th axes
    """
    res = data.swapaxes(2, 3)
    res = res.swapaxes(1, 2)
    res = res.astype(np.float32)/255
    return res

def transform(data):
    #data = mx.image.imresize(data, 32, 32)
    res = data.transpose((2,0,1))
    #data = mx.nd.swapaxes(data, 0, 2)
    res = res.astype(np.float32)
    return res

def get_mnist_iter(args, kv):
    """
    create data iterator with NDArrayIter
    """
    # (train_lbl1, train_img1) = read_data(
    #         'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz')
    # (val_lbl, val_img) = read_data(
    #         't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')
    # print (train_img1.shape)
    #
    # train = mx.gluon.data.DataLoader(mx.gluon.data.vision.CIFAR10(root='~/.mxnet/datasets/cifar10', train=True, transform=transform),
    #                         args.batch_size, shuffle=True, last_batch='rollover')
    # val = mx.gluon.data.DataLoader(mx.gluon.data.vision.CIFAR10(root='~/.mxnet/datasets/cifar10', train=False, transform=transform),
    #                         args.batch_size, shuffle=False, last_batch='rollover')

    train_cifar10 = mx.gluon.data.vision.CIFAR10(root='~/.mxnet/datasets/cifar10', train=True, transform=transform)
    val_cifar10 = mx.gluon.data.vision.CIFAR10(root='~/.mxnet/datasets/cifar10', train=False, transform=transform)
    print ("original: ", train_cifar10._data.shape)
    swapped = swap(train_cifar10._data)
    swappedval = swap(val_cifar10._data)
    print ("swap: ", swapped.shape)
    #print ("transform: ", transform(train_cifar10._data).shape)
    train = mx.io.NDArrayIter(
        swapped, train_cifar10._label, args.batch_size, shuffle=True)
    val = mx.io.NDArrayIter(
        swappedval, val_cifar10._label, args.batch_size)

    return (train, val)

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="train mnist",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-classes', type=int, default=10,
                        help='the number of classes')
    parser.add_argument('--num-examples', type=int, default=50000,
                        help='the number of training examples')

    parser.add_argument('--add_stn',  action="store_true", default=False, help='Add Spatial Transformer Network Layer (lenet only)')

    fit.add_fit_args(parser)
    parser.set_defaults(
        # network
        network        = 'lenet',
        # train
        gpus           = None,
        batch_size     = 64,
        disp_batches   = 100,
        num_epochs     = 20,
        lr             = .01,
        lr_step_epochs = '10'
    )
    args = parser.parse_args()

    # load network
    from importlib import import_module
    net = import_module('symbols.'+args.network)
    sym = net.get_symbol(**vars(args))

    # train
    fit.fit(args, sym, get_mnist_iter)

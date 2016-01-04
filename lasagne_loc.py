#!/usr/bin/env python
import json
import numpy as np
import sklearn.cross_validation

import matplotlib.pyplot as plt

import os
import pickle

import lasagne
import theano
import theano.tensor as T

from lasagne.layers import InputLayer
from lasagne.layers.dnn import Conv2DDNNLayer as Conv
from lasagne.layers.dnn import Pool2DDNNLayer as Pool
from lasagne.layers import Upscale2DLayer
from lasagne.layers import ConcatLayer

from lasagne.nonlinearities import softmax

from PIL import Image

DATA_DIR = './data/train320'

d = pickle.load(open('./data/train320data.pkl'))
fns = d['filenames']
p1ts = d['point1']
p2ts = d['point2']

SEED = int(os.environ['SEED'])

np.random.seed(SEED)

train_ix, val_ix = sklearn.cross_validation.train_test_split(
    np.arange(len(fns)), test_size=0.2)


def batch(ix, N, seed):
    X = np.zeros((N, 3, 320, 320)).astype('float32')
    Mask = np.zeros((N, 2, 320, 320)).astype('float32')

    rng = np.random.RandomState(seed)
    ix = ix[rng.randint(0, len(ix), N)]
    for n, i in enumerate(ix):
        fn = fns[i]

        im = plt.imread('{}/{}'.format(DATA_DIR, fn))
        im = im/255. - 0.5
        mask = np.zeros(im.shape[:2])

        x1, y1 = p1ts[i]
        x1 = int(x1)
        y1 = int(y1)
        try:
            mask[y1, x1] = 1
        except IndexError:
            pass

        Mask[n, 0] = mask

        mask = np.zeros(im.shape[:2])
        x1, y1 = p2ts[i]
        x1 = int(x1)
        y1 = int(y1)
        try:
            mask[y1, x1] = 1
        except IndexError:
            pass

        Mask[n, 1] = mask
        X[n] = im.transpose(2, 0, 1)
    return X, Mask


def maxloc(mask):
    a = mask.reshape((-1, IMAGE_W*IMAGE_W)).argmax(-1)
    y = a % IMAGE_W
    x = a // IMAGE_W
    return np.array((y, x)).T


def l2dist(x1, x2):
    return np.sqrt(((x1 - x2)**2).sum(-1))


def build_net(IMAGE_W):
    net = {}
    l = InputLayer((None, 3, IMAGE_W, IMAGE_W))
    net['input'] = l

    l = Conv(l, 16, 3, pad='same')
    net['T4'] = l

    l = Conv(Pool(l, 2), 32, 3, pad='same')
    net['T3'] = l

    l = Conv(Pool(l, 2), 48, 3, pad='same')
    net['T2'] = l

    l = Conv(Pool(l, 2), 48, 3, pad='same')
    net['T1'] = l

    l = Conv(Conv(net['T1'], 48, 3, pad='same'), 48, 3, pad='same')
    l = Upscale2DLayer(l, 2)
    net['M1'] = l

    l = ConcatLayer((net['T2'], net['M1']))
    l = Conv(Conv(l, 48, 3, pad='same'), 32, 3, pad='same')
    l = Upscale2DLayer(l, 2)
    net['M2'] = l

    l = ConcatLayer((net['T3'], net['M2']))
    l = Conv(Conv(l, 32, 3, pad='same'), 16, 3, pad='same')
    l = Upscale2DLayer(l, 2)
    net['M3'] = l

    l = ConcatLayer((net['T4'], net['M3']))
    l = Conv(Conv(l, 16, 3, pad='same'), 2, 3, pad='same', nonlinearity=None)

    l = lasagne.layers.ReshapeLayer(l, (-1, IMAGE_W*IMAGE_W))
    l = lasagne.layers.NonlinearityLayer(l, softmax)
    l = lasagne.layers.ReshapeLayer(l, (-1, 2, IMAGE_W, IMAGE_W))
    net['M4'] = l

    return net


# Build net
IMAGE_W = 320

lasagne.random.set_rng(np.random.RandomState(SEED))
net = build_net(IMAGE_W)

X = T.tensor4()
Y = T.tensor4()
output = lasagne.layers.get_output(net['M4'], X)

loss = lasagne.objectives.binary_crossentropy(
    output.reshape((-1, IMAGE_W*IMAGE_W)),
    Y.reshape((-1, IMAGE_W*IMAGE_W))
    )
loss = T.mean(loss)

params = lasagne.layers.get_all_params(net['M4'])

LR = theano.shared(np.array(0.0001).astype('float32'))
updates = lasagne.updates.adam(loss, params, learning_rate=LR)

f_predict = theano.function([X], output)
f_train = theano.function([X, Y], loss, updates=updates)
f_val = theano.function([X, Y], [loss, output])


# TRAIN
rng = np.random.RandomState(SEED)

for epoch in range(100):
    train_loss = 0
    for _ in range(10):
        Xb, Mb = batch(train_ix, 24, rng.randint(4294967295))
        train_loss += f_train(Xb, Mb)
    train_loss /= 10
    val_dist = []
    val_loss = 0
    for _ in range(10):
        Xb, Mb = batch(val_ix, 24, rng.randint(4294967295))
        loss, Pb = f_val(Xb, Mb)
        val_loss += loss
        val_dist.append(l2dist(maxloc(Mb), maxloc(Pb)))
    val_dist = np.concatenate(val_dist)
    val_loss /= 10
    print('Epoch {:03}: loss train (val) {:.07f} ({:.07f}) mean dist {:.01f}, nearby {:.03f}'.format(
            epoch, train_loss, val_loss, val_dist.mean(), (val_dist < 10).mean()))

if np.isnan(train_loss) or np.isnan(val_loss):
    raise ValueError('NaN loss!')

pvt = lasagne.layers.get_all_param_values(net['M4'])
pickle.dump(pvt, open(
    './models/loc2_seed_{}_trained_100epoch.pkl'.format(SEED), 'w'))

# TEST
TEST_DATA_DIR = './data/test320'
ORIG_DATA_DIR = './data/test'

fns = os.listdir(TEST_DATA_DIR)

d = {}
for i in range(len(fns)):
    im = np.array(Image.open('{}/{}'.format(TEST_DATA_DIR, fns[i])))
    im0 = Image.open('{}/{}'.format(ORIG_DATA_DIR, fns[i]))

    im = im/255. - 0.5
    im = im.transpose(2, 0, 1)[np.newaxis]
    im = im.astype('float32')

    ps = maxloc(f_predict(im))

    s = im0.size[0] / 320.
    t = (320 - im0.size[1] / s) / 2.
    ps[:, 1] = ps[:, 1] - t
    ps = ps * s
    ps = ps * (384, 384)/im0.size[:2]

    d[fns[i]] = ps

testpoints = json.load(open('./testpoints/testpoints1_filtered.json'))
for p in testpoints:
    x, y = d[p['filename']][0]
    p['annotations'][0]['x'] = int(x)
    p['annotations'][0]['y'] = int(y)
json.dump(
    testpoints,
    open('./testpoints/lasagne_loc_seed_{}_point1.json'.format(SEED), 'w'),
    indent=0)

testpoints = json.load(open('./testpoints/testpoints2_filtered.json'))
for p in testpoints:
    x, y = d[p['filename']][1]
    p['annotations'][0]['x'] = int(x)
    p['annotations'][0]['y'] = int(y)

json.dump(
    testpoints,
    open('./testpoints/lasagne_loc_seed_{}_point2.json'.format(SEED), 'w'),
    indent=0)

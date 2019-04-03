# %matplotlib inline
import d2lzh as d2l
from mxnet.gluon import data as gdata
import sys
import time

mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)
print(len(mnist_train), len(mnist_test))

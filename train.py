import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

import matplotlib.pyplot as plt
import six

import data
import model as M 

mnist = data.load_mnist_data()

x_all = mnist['data'].astype(np.float32) / 255
y_all = mnist['target'].astype(np.int32)
x_train, x_test = np.split(x_all, [60000])
y_train, y_test = np.split(y_all, [60000])

model = L.Classifier(M.MNISTModel())
optimizer = optimizers.SGD()
optimizer.setup(model)

accuracy_data = []

batchsize = 100
datasize = 60000  
for epoch in range(20):
  print('epoch %d' % epoch)
  indexes = np.random.permutation(datasize)
  for i in range(0, datasize, batchsize):
    x = Variable(x_train[indexes[i : i + batchsize]])
    t = Variable(y_train[indexes[i : i + batchsize]])
    optimizer.update(model, x, t)
    accuracy_data.append(model.accuracy.data)

sum_loss, sum_accuracy = 0, 0

plt.plot(accuracy_data, 'k--')
plt.show()
plt.savefig("accuracy.png")

for i in range(0, 10000, batchsize):
  x = Variable(x_test[i : i + batchsize])
  t = Variable(y_test[i : i + batchsize])
  loss = model(x, t)
  sum_loss += loss.data * batchsize
  sum_accuracy += model.accuracy.data * batchsize

mean_loss     = sum_loss / 10000
mean_accuracy = sum_accuracy / 10000

print('mean_loss %.2f' % mean_loss)
print('mean_accuracy %d' % (mean_accuracy * 100))

if (mean_accuracy * 100) > 90:
  serializers.save_hdf5('trained.model', model)
  print "model has saved, it has enough quality as trained model :)"

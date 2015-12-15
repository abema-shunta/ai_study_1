import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from PIL import Image
from PIL import ImageOps

import argparse
import model as M 

parser = argparse.ArgumentParser(
    description='A Neural Algorithm of Artistic Style')
parser.add_argument('--img', '-i', default='',
                    help='path of input image')
args = parser.parse_args()

model = L.Classifier(M.MNISTModel())
serializers.load_hdf5('trained.model', model)

img = Image.open(args.img)
img = ImageOps.grayscale(img)
img = ImageOps.invert(img)

x = np.asarray(img.resize((28,28))).reshape((1,784)).astype(np.float32) / 20.
x = Variable(x)

answer = [0,0.]
for i in range(10):
  loss = model(x, Variable(np.array([i], dtype='int32')))
  accuracy = model.accuracy.data
  print "Probability of ", i, " : ", accuracy*100, "%" 
  if answer[1] < accuracy:
    answer = [i, accuracy]

print "-------------------------"
print "Image should be ", answer[0]

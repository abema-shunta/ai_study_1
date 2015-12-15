import numpy as np 
import chainer 
from chainer import Variable

# 変数定義。初期値に5を設定。型は浮動小数32bit
x_data = np.array([5], dtype=np.float32)
x = Variable(x_data)

# 準伝搬計算
y = x**2 - 2 * x + 1

# 値の確認
y.data
# >> array([ 16.], dtype=float32)

# 誤差電波
y.backward()
# 変化量の表示
x.grad
# >> array([ 8.], dtype=float32)

# 行列を用いた計算
x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
y = x**2 - 2*x + 1
y.grad = np.ones((2, 3), dtype=np.float32)
y.backward()
x.grad
# array([[  0.,   2.,   4.],
#        [  6.,   8.,  10.]], dtype=float32)

# リンクの読み込み
import chainer.functions as F
import chainer.links as L

# 線形計算。全結合。
f = F.Linear(3, 2)

# 重み
f.W.data
# array([[ 1.01847613,  0.23103087,  0.56507462],
#        [ 1.29378033,  1.07823515, -0.56423163]], dtype=float32)

# バイアス
f.b.data
# array([ 0.,  0.], dtype=float32)

# Linear を用いた計算
x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
y = f(x)
y.data
# array([[ 3.1757617 ,  1.75755572],
#        [ 8.61950684,  7.18090773]], dtype=float32)

# 勾配の初期化
y.zerograds()

# Linearでもちゃんと逆伝搬します。
y.grad = np.ones((2, 2), dtype=np.float32)
y.backward()
f.W.grad
# array([[ 5.,  7.,  9.],
       # [ 5.,  7.,  9.]], dtype=float32)
f.b.grad
# array([ 2.,  2.], dtype=float32)

# Linkをつなげたものがチェーン
l1 = L.Linear(4, 3)
l2 = L.Linear(3, 2)

def my_forward(x):
    h = l1(x)
    return l2(h)

# 再利用性を高めるために、Chainクラスからサブクラスを作ります。
from chainer import Chain

class MyChain(Chain):
     def __init__(self):
         super(MyChain, self).__init__(
             l1=L.Linear(4, 3),
             l2=L.Linear(3, 2),
         )
     def __call__(self, x):
         h = self.l1(x)
         return self.l2(h)

# パラメータの最適化にはオプティマイザを使う。
from chainer import optimizers

model = MyChain()
optimizer = optimizers.SGD()
optimizer.setup(model)

model.zerograds()
optimizer.update()

# モデルやオプティマイザを保存するのに、シリアライザを使う。
from chainer import serializers

serializers.save_hdf5('my.model', model)
serializers.load_hdf5('my.model', model)

serializers.save_hdf5('my.state', optimizer)
serializers.load_hdf5('my.state', optimizer)




import chainer
import chainer.links as L
import chainer.functions as F
from chainer.functions.array.reshape import reshape


def _global_average_pooling_2d(x):
    n, channel, rows, cols = x.shape
    h = F.average_pooling_2d(x, (rows, cols), stride=1)
    h = reshape(h, (n, channel))
    return h


class ResNet50_Fine(chainer.Chain):
    def __init__(self, output=8):
        super(ResNet50_Fine, self).__init__()

        with self.init_scope():
            self.base = L.ResNet50Layers()
            self.fc = L.Linear(None, output)

    def __call__(self, x):
        h = self.base(x, layers=['res5'])['res5']
        self.cam = h
        h = _global_average_pooling_2d(h)
        h = self.fc(h)

        return h

import chainer
import chainer.functions as F
import chainer.links as L


class ConvBNR(chainer.Chain):
    def __init__(self, ch0, ch1, sample='down', use_bn=True, dropout=False):
        self.sample = sample
        self.use_bn = use_bn
        self.dropout = dropout
        w = chainer.initializers.HeNormal()

        super(ConvBNR, self).__init__()
        with self.init_scope():
            if self.sample == 'down':
                self.c = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)
            else:
                self.c = L.Convolution2D(ch0, ch1, 3, 1, 1, initialW=w)
            if use_bn:
                self.bn = L.BatchNormalization(ch1)

    def __call__(self, x):
        h = x
        if self.sample == 'down':
            h = self.c(h)
            if self.use_bn:
                h = self.bn(h)
            if self.dropout:
                h = F.dropout(h)
            h = F.leaky_relu(h)

        else:
            h = F.unpooling_2d(h, 2, 2, cover_all=False)
            h = self.c(h)
            if self.use_bn:
                h = self.bn(h)
            if self.dropout:
                h = F.dropout(h)
            h = F.relu(h)

        return h


class DconvBNR(chainer.Chain):
    def __init__(self, ch0, ch1, sample='down', use_bn=True, dropout=False):
        self.sample = sample
        self.use_bn = use_bn
        self.dropout = dropout
        w = chainer.initializers.HeNormal()

        super(DconvBNR, self).__init__()
        with self.init_scope():
            if self.sample == 'down':
                self.c = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)
            else:
                self.c = L.Deconvolution2D(ch0, ch1, 4, 2, 1, initialW=w)
            if use_bn:
                self.bn = L.BatchNormalization(ch1)

    def __call__(self, x):
        h = x
        if self.sample == 'down':
            h = self.c(h)
            if self.use_bn:
                h = self.bn(h)
            if self.dropout:
                h = F.dropout(h)
            h = F.leaky_relu(h)

        else:
            h = self.c(h)
            if self.use_bn:
                h = self.bn(h)
            if self.dropout:
                h = F.dropout(h)
            h = F.relu(h)

        return h

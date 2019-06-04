
import chainer
import chainer.functions as F
import chainer.links as L


class ConvBNR(chainer.Chain):
    def __init__(self, ch0, ch1, sample='down', use_bn=False, dropout=False):
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


Blocks = {'conv': ConvBNR, 'deconv': DconvBNR}


class Generator(chainer.Chain):
    def __init__(self, in_ch, out_ch, upsample):
        w = chainer.initializers.HeNormal()
        Block = Blocks[upsample]
        super(Generator, self).__init__()
        with self.init_scope():
            self.e0 = L.Convolution2D(in_ch, 64, 3, 1, 1, initialW=w)
            self.e1 = Block(64, 128, sample='down')
            self.e2 = Block(128, 256, sample='down')
            self.e3 = Block(256, 512, sample='down')
            self.e4 = Block(512, 512, sample='down')
            self.e5 = Block(512, 512, sample='down')
            self.e6 = Block(512, 512, sample='down')
            self.e7 = Block(512, 512, sample='down')

            self.d0 = Block(512, 512, sample='up')
            self.d1 = Block(1024, 512, sample='up')
            self.d2 = Block(1024, 512, sample='up')
            self.d3 = Block(1024, 512, sample='up')
            self.d4 = Block(1024, 256, sample='up')
            self.d5 = Block(512, 128, sample='up')
            self.d6 = Block(256, 64, sample='up')
            self.d7 = L.Convolution2D(128, out_ch, 3, 1, 1, initialW=w)

    def __call__(self, x):

        h1 = F.leaky_relu(self.e0(x))
        h2 = self.e1(h1)
        h3 = self.e2(h2)
        h4 = self.e3(h3)
        h5 = self.e4(h4)
        h6 = self.e5(h5)
        h7 = self.e6(h6)
        h8 = self.e7(h7)

        h = self.d0(h8)
        inp = F.concat([h, h7], axis=1)
        h = self.d1(inp)
        inp = F.concat([h, h6], axis=1)
        h = self.d2(inp)
        inp = F.concat([h, h5], axis=1)
        h = self.d3(inp)
        inp = F.concat([h, h4], axis=1)
        h = self.d4(inp)
        inp = F.concat([h, h3], axis=1)
        h = self.d5(inp)
        inp = F.concat([h, h2], axis=1)
        h = self.d6(inp)
        inp = F.concat([h, h1], axis=1)
        h = self.d7(inp)

        out = F.tanh(h)

        return out

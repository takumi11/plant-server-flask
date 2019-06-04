
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


class Attention_block(chainer.Chain):
    def __init__(self, l_ch, g_ch, int_ch, pad):
        w = chainer.initializers.HeNormal()
        super(Attention_block, self).__init__()
        with self.init_scope():
            self.W_x = L.Convolution2D(l_ch, int_ch, 1, 1, 0, initialW=w)
            self.bn_x = L.BatchNormalization(int_ch)

            self.W_g = L.Convolution2D(g_ch, int_ch, 1, 1, pad, initialW=w)
            self.bn_g = L.BatchNormalization(int_ch)

            self.psi = L.Convolution2D(int_ch, 1, 1, 1, 0, initialW=w)
            self.bn_psi = L.BatchNormalization(1)

    def __call__(self, x, g):

        x1 = self.W_x(x)
        g1 = self.W_g(g)
        after_relu = F.relu(g1+x1)

        psi = F.sigmoid(self.psi(after_relu))

        return x * psi


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
            self.a0 = Attention_block(512, 512, 256, 1)

            self.d1 = Block(1024, 512, sample='up')
            self.a1 = Attention_block(512, 1024, 256, 2)

            self.d2 = Block(1024, 512, sample='up')
            self.a2 = Attention_block(512, 1024, 256, 4)

            self.d3 = Block(1024, 512, sample='up')
            self.a3 = Attention_block(512, 1024, 256, 8)

            self.d4 = Block(1024, 256, sample='up')
            self.a4 = Attention_block(256, 1024, 128, 16)

            self.d5 = Block(512, 128, sample='up')
            self.a5 = Attention_block(128, 512, 64, 32)

            self.d6 = Block(256, 64, sample='up')
            self.a6 = Attention_block(64, 256, 32, 64)

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
        attn0 = self.a0(x=h7, g=h8)
        h9 = F.concat([h, attn0], axis=1)

        h = self.d1(h9)
        attn1 = self.a1(x=h6, g=h9)
        h10 = F.concat([h, attn1], axis=1)

        h = self.d2(h10)
        attn2 = self.a2(x=h5, g=h10)
        h11 = F.concat([h, attn2], axis=1)

        h = self.d3(h11)
        attn3 = self.a3(x=h4, g=h11)
        h12 = F.concat([h, attn3], axis=1)

        h = self.d4(h12)
        attn4 = self.a4(x=h3, g=h12)
        h13 = F.concat([h, attn4], axis=1)

        h = self.d5(h13)
        attn5 = self.a5(x=h2, g=h13)
        h14 = F.concat([h, attn5], axis=1)

        h = self.d6(h14)
        attn6 = self.a6(x=h1, g=h14)
        h15 = F.concat([h, attn6], axis=1)

        out = self.d7(h15)
        out = F.tanh(out)

        return out

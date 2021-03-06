import chainer
import chainer.functions as F
import chainer.links as L
from training.model.base_block import ConvBNR, DconvBNR

network_name = 'attention_unet'
Blocks = {'conv': ConvBNR, 'deconv': DconvBNR}


class Attention_block(chainer.Chain):
    def __init__(self, g_ch, l_ch, int_ch):
        w = chainer.initializers.HeNormal()
        super(Attention_block, self).__init__()
        with self.init_scope():
            self.W_g = L.Convolution2D(g_ch, int_ch, 1, 1, 0, initialW=w)
            self.bn_g = L.BatchNormalization(int_ch)

            self.W_x = L.Convolution2D(g_ch, int_ch, 1, 1, 0, initialW=w)
            self.bn_x = L.BatchNormalization(int_ch)

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
            self.a0 = Attention_block(512, 512, 256)

            self.d1 = Block(1024, 512, sample='up')
            self.a1 = Attention_block(512, 512, 256)

            self.d2 = Block(1024, 512, sample='up')
            self.a2 = Attention_block(512, 512, 256)

            self.d3 = Block(1024, 512, sample='up')
            self.a3 = Attention_block(512, 512, 256)

            self.d4 = Block(1024, 256, sample='up')
            self.a4 = Attention_block(256, 256, 128)

            self.d5 = Block(512, 128, sample='up')
            self.a5 = Attention_block(128, 128, 64)

            self.d6 = Block(256, 64, sample='up')
            self.a6 = Attention_block(64, 64, 32)

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

        h9 = self.d0(h8)

        attn0 = self.a0(x=h7, g=h9)
        con = F.concat([h9, attn0], axis=1)
        h10 = self.d1(con)

        attn1 = self.a1(x=h6, g=h10)
        con = F.concat([h10, attn1], axis=1)
        h11 = self.d2(con)

        attn2 = self.a2(x=h5, g=h11)
        con = F.concat([h11, attn2], axis=1)
        h12 = self.d3(con)

        attn3 = self.a3(x=h4, g=h12)
        con = F.concat([h12, attn3], axis=1)
        h13 = self.d4(con)

        attn4 = self.a4(x=h3, g=h13)
        con = F.concat([h13, attn4], axis=1)
        h14 = self.d5(con)

        attn5 = self.a5(x=h2, g=h14)
        con = F.concat([h14, attn5], axis=1)
        h15 = self.d6(con)

        attn6 = self.a6(x=h1, g=h15)
        con = F.concat([h15, attn6], axis=1)
        out = self.d7(con)

        out = F.tanh(out)

        return out

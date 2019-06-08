# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
from model.base_block import ConvBNR, DconvBNR

network_name = 'unet'
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

            self.d7 = Block(512, 512, sample='up')
            self.d6 = Block(1024, 512, sample='up')
            self.d5 = Block(1024, 512, sample='up')
            self.d4 = Block(1024, 512, sample='up')
            self.d3 = Block(1024, 256, sample='up')
            self.d2 = Block(512, 128, sample='up')
            self.d1 = Block(256, 64, sample='up')
            self.d0 = L.Convolution2D(128, out_ch, 3, 1, 1, initialW=w)

    def __call__(self, x):

        h1 = F.leaky_relu(self.e0(x))
        h2 = self.e1(h1)
        h3 = self.e2(h2)
        h4 = self.e3(h3)
        h5 = self.e4(h4)
        h6 = self.e5(h5)
        h7 = self.e6(h6)
        h8 = self.e7(h7)

        h = self.d7(h8)
        h9 = F.concat([h, h7], axis=1)
        h = self.d6(h9)
        h10 = F.concat([h, h6], axis=1)
        h = self.d5(h10)
        h11 = F.concat([h, h5], axis=1)
        h = self.d4(h11)
        h12 = F.concat([h, h4], axis=1)
        h = self.d3(h12)
        h13 = F.concat([h, h3], axis=1)
        h = self.d2(h13)
        h14 = F.concat([h, h2], axis=1)
        h = self.d1(h14)
        h15 = F.concat([h, h1], axis=1)
        out = self.d0(h15)

        out = F.tanh(out)

        return out

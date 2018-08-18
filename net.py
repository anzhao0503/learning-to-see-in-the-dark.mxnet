import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn
import mxnet.ndarray as F


def conv(channels, kernel_size):
    out = nn.Sequential()
    out.add(nn.Conv2D(channels, kernel_size=kernel_size, strides=(1,1),  padding=(1, 1), use_bias=False),
            nn.LeakyReLU(0.2)
            )
    return out


def DownConv(channels, kernel_size):
    out = nn.Sequential()
    out.add(conv(channels, kernel_size),
            conv(channels, kernel_size)
            )
    return out


class UpConv(nn.Block):
    def __init__(self, out_channels):
        super(UpConv, self).__init__()
        self.out_channels = out_channels
        self.up_conv = nn.Conv2DTranspose(self.out_channels, kernel_size=2, strides=2, use_bias=False)
        self.conv3_0 = conv(self.out_channels, kernel_size=3)
        self.conv3_1 = conv(self.out_channels, kernel_size=3)

    def forward(self, x, s):
        x = self.up_conv(x)
        x = F.Crop(*[x, s], center_crop=True)
        x = s + x
        x = self.conv3_0(x)
        x = self.conv3_1(x)
        return x


class PixelShuffle(nn.Block):
    '''
    sub-pixel convolution layer
    ref: https://arxiv.org/abs/1609.05158
    '''
    def __init__(self, up_scale):
        super(PixelShuffle, self).__init__()
        self.up_scale = up_scale

    def forward(self, x):
        batch_size, c, h, w = x.shape
        if batch_size is None:
            batch_size = -1
        out_c = c // (self.up_scale * self.up_scale)
        out_h = h * self.up_scale
        out_w = w * self.up_scale
        out = F.reshape(x, (batch_size, self.up_scale, self.up_scale, out_c, h, w))
        out = F.transpose(out, (0, 3, 4, 1, 5, 2))
        out = F.reshape(out, (batch_size, out_c, out_h, out_w))
        return out


class UNet(nn.Block):
    def __init__(self, out_channels, up_scale):
        '''
        ref: https://arxiv.org/abs/1505.04597
        :param out_channels: 12 for Sony, 27 for Fuji
        :param up_scale: 2 for Sony, 3 for Fuji
        '''
        super(UNet, self).__init__()
        self.out_channels = out_channels
        self.up_scale = up_scale
        with self.name_scope():
            self.conv1 = DownConv(32, 3)
            self.pool1 = nn.MaxPool2D(pool_size=(2, 2), padding=1)
            self.conv2 = DownConv(64, 3)
            self.pool2 = nn.MaxPool2D(pool_size=(2, 2), padding=1)
            self.conv3 = DownConv(128, 3)
            self.pool3 = nn.MaxPool2D(pool_size=(2, 2), padding=1)
            self.conv4 = DownConv(256, 3)
            self.pool4 = nn.MaxPool2D(pool_size=(2, 2), padding=1)
            self.conv5 = DownConv(512, 3)

            self.up6 = UpConv(256)
            self.up7 = UpConv(128)
            self.up8 = UpConv(64)
            self.up9 = UpConv(32)
            self.conv10 = nn.Conv2D(self.out_channels, kernel_size=1)
            self.out = PixelShuffle(self.up_scale)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        conv5 = self.conv5(pool4)

        up6 = self.up6(conv5, conv4)
        up7 = self.up7(up6, conv3)
        up8 = self.up8(up7, conv2)
        up9 = self.up9(up8, conv1)
        conv10 = self.conv10(up9)
        out = self.out(conv10)
        return out


# if __name__ == '__main__':
#     model = UNet(12, 2)
#     model.initialize()
#     x = F.array(np.random.random((1, 4, 256, 256)))
#     out = model(x)
#     print out

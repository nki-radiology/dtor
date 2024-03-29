import math
from torch import nn as nn
from dtor.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


class Model(nn.Module):
    def __init__(self, in_channels=3, conv_channels=8, prelim=65536, dry=False): #65536): # (64 for MNIST 3D)
        super().__init__()

        self.tail_batchnorm = nn.BatchNorm3d(in_channels)

        self.block1 = LTPBlock(in_channels, conv_channels)
        self.block2 = LTPBlock(conv_channels, conv_channels * 2)
        self.block3 = LTPBlock(conv_channels * 2, conv_channels * 4)
        self.block4 = LTPBlock(conv_channels * 4, conv_channels * 8)

        self.head_linear = nn.Linear(prelim, 2)
        self.head_softmax = nn.Softmax(dim=1)
        self.dry = dry
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        #print(input_batch.shape)
        bn_output = self.tail_batchnorm(input_batch)
        #print(bn_output.shape)

        block_out = self.block1(bn_output)
        #print(block_out.shape)
        block_out = self.block2(block_out)
        #print(block_out.shape)
        block_out = self.block3(block_out)
        #print(block_out.shape)
        block_out = self.block4(block_out)
        #print(block_out.shape)

        conv_flat = block_out.view(
            block_out.size(0),
            -1,
        )
        #print(conv_flat.shape)
        if self.dry:
            return block_out
        linear_output = self.head_linear(conv_flat)

        return linear_output, self.head_softmax(linear_output)


class LTPBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels, conv_channels, kernel_size=3, padding=1, bias=True,
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            conv_channels, conv_channels, kernel_size=3, padding=1, bias=True,
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(2, 2)
        #self.maxpool = nn.MaxPool3d(4, 4)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return self.maxpool(block_out)

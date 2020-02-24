from torch import nn
import torch
from torchvision import models
import torchvision
from torch.nn import functional as F


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class UNet11(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            vgg - encoder pre-trained with VGG11
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.num_classes = num_classes

        self.encoder = models.vgg11(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[3],
                                   self.relu)

        self.conv3 = nn.Sequential(
            self.encoder[6],
            self.relu,
            self.encoder[8],
            self.relu,
        )
        self.conv4 = nn.Sequential(
            self.encoder[11],
            self.relu,
            self.encoder[13],
            self.relu,
        )

        self.conv5 = nn.Sequential(
            self.encoder[16],
            self.relu,
            self.encoder[18],
            self.relu,
        )

        self.center = DecoderBlock(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv=is_deconv)
        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv=is_deconv)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 4, is_deconv=is_deconv)
        self.dec3 = DecoderBlock(256 + num_filters * 4, num_filters * 4 * 2, num_filters * 2, is_deconv=is_deconv)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv=is_deconv)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))
        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out_ = self.final(dec1)
            x_out  = torch.tanh(x_out_) 

        return x_out


class UNet16(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained: if encoder uses pre-trained weigths from VGG16
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   self.relu,
                                   self.encoder[26],
                                   self.relu,
                                   self.encoder[28],
                                   self.relu)

        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8, is_deconv=is_deconv)

        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv=is_deconv)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv=is_deconv)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv=is_deconv)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv=is_deconv)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out_ = self.final(dec1)
            x_out  = torch.tanh(x_out_) 

        return x_out
    
class Conv3BN(nn.Module):
    def __init__(self, in_: int, out: int, bn=False):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out) if bn else None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x


class UNetModule(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.l1 = Conv3BN(in_, out)
        self.l2 = Conv3BN(out, out)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x

    
    
class AlbuNet34(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder
        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
        """

    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes
        print('loading AlbuNet34')
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.resnet34(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlock(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlock(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlock(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            logits = self.final(dec0)
            x_out  = torch.tanh(logits)
            
        return x_out

    

class UNet(nn.Module):
    """
    Vanilla UNet.

    Implementation from https://github.com/lopuhin/mapillary-vistas-2017/blob/master/unet_models.py
    """
    output_downscaled = 1
    module = UNetModule

    def __init__(self,
                 input_channels: int = 3,
                 filters_base: int = 32,
                 down_filter_factors=(1, 2, 4, 8, 16),
                 up_filter_factors=(1, 2, 4, 8, 16),
                 bottom_s=4,
                 num_classes=1,
                 add_output=True):
        super().__init__()
        self.num_classes = num_classes
        assert len(down_filter_factors) == len(up_filter_factors)
        assert down_filter_factors[-1] == up_filter_factors[-1]
        down_filter_sizes = [filters_base * s for s in down_filter_factors]
        up_filter_sizes = [filters_base * s for s in up_filter_factors]
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        self.down.append(self.module(input_channels, down_filter_sizes[0]))
        for prev_i, nf in enumerate(down_filter_sizes[1:]):
            self.down.append(self.module(down_filter_sizes[prev_i], nf))
        for prev_i, nf in enumerate(up_filter_sizes[1:]):
            self.up.append(self.module(
                down_filter_sizes[prev_i] + nf, up_filter_sizes[prev_i]))
        pool = nn.MaxPool2d(2, 2)
        pool_bottom = nn.MaxPool2d(bottom_s, bottom_s)
        upsample = nn.Upsample(scale_factor=2)
        upsample_bottom = nn.Upsample(scale_factor=bottom_s)
        self.downsamplers = [None] + [pool] * (len(self.down) - 1)
        self.downsamplers[-1] = pool_bottom
        self.upsamplers = [upsample] * len(self.up)
        self.upsamplers[-1] = upsample_bottom
        self.add_output = add_output
        if add_output:
            self.conv_final = nn.Conv2d(up_filter_sizes[0], num_classes, 1)

    def forward(self, x):
        xs = []
        for downsample, down in zip(self.downsamplers, self.down):
            x_in = x if downsample is None else downsample(xs[-1])
            x_out = down(x_in)
            xs.append(x_out)

        x_out = xs[-1]
        for x_skip, upsample, up in reversed(
                list(zip(xs[:-1], self.upsamplers, self.up))):
            x_out = upsample(x_out)
            x_out = up(torch.cat([x_out, x_skip], 1))

        if self.add_output:
            x_out = self.conv_final(x_out)
            if self.num_classes > 1:
                x_out = F.log_softmax(x_out, dim=1)
        return torch.tanh(x_out)
    

class EncDec(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder
        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
        """

    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = torchvision.models.resnet34(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)

        # Encoder 
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        # Decoder 
        self.dec5 = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlock(num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlock(num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlock(num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder 
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        # Decoder 
        dec5 = self.dec5(conv5)
        dec4 = self.dec4(dec5)
        dec3 = self.dec3(dec4)
        dec2 = self.dec2(dec3)
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out_ = self.final(dec0)
            x_out  = torch.tanh(x_out_)

        return x_out
    
    
class MDeNet(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder
        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
        """

    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        print("Loading MDeNet")
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = torchvision.models.resnet34(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)

        # Encoder 
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        # Decoder 1
        self.dec1_5 = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec1_4 = DecoderBlock(num_filters * 8, num_filters * 8 * 2, num_filters * 4, is_deconv)
        self.dec1_3 = DecoderBlock(num_filters * 4, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec1_2 = DecoderBlock(num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1_1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec1_0 = ConvRelu(num_filters, num_filters)
        
        # Decoder 2
        self.dec2_4 = DecoderBlock(256, num_filters * 8 * 2, num_filters * 4, is_deconv)
        self.dec2_3 = DecoderBlock(num_filters * 4, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2_2 = DecoderBlock(num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec2_1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec2_0 = ConvRelu(num_filters, num_filters)
            
        # Decoder 3
        self.dec3_3 = DecoderBlock(128, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec3_2 = DecoderBlock(num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec3_1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec3_0 = ConvRelu(num_filters, num_filters)
            
        # Decoder 4
        self.dec4_2 = DecoderBlock(64, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec4_1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec4_0 = ConvRelu(num_filters, num_filters)
        
        # Decoder 5
        self.dec5_2 = DecoderBlock(64, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec5_1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec5_0 = ConvRelu(num_filters, num_filters)
        
        # Concatenation layer 
        self.final = nn.Conv2d(5*num_filters, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder 
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        # Decoder 1
        dec1_5 = self.dec1_5(conv5)
        dec1_4 = self.dec1_4(dec1_5)
        dec1_3 = self.dec1_3(dec1_4)
        dec1_2 = self.dec1_2(dec1_3)
        dec1_1 = self.dec1_1(dec1_2)
        dec1_0 = self.dec1_0(dec1_1)

        # Decoder 2
        dec2_4 = self.dec2_4(conv4)
        dec2_3 = self.dec2_3(dec2_4)
        dec2_2 = self.dec2_2(dec2_3)
        dec2_1 = self.dec2_1(dec2_2)
        dec2_0 = self.dec2_0(dec2_1)
        
        # Decoder 3
        dec3_3 = self.dec3_3(conv3)
        dec3_2 = self.dec3_2(dec3_3)
        dec3_1 = self.dec3_1(dec3_2)
        dec3_0 = self.dec3_0(dec3_1)
                
        # Decoder 4
        dec4_2 = self.dec4_2(conv2)
        dec4_1 = self.dec4_1(dec4_2)
        dec4_0 = self.dec4_0(dec4_1)
                        
        # Decoder 5
        dec5_2 = self.dec5_2(conv1)
        dec5_1 = self.dec5_1(dec5_2)
        dec5_0 = self.dec5_0(dec5_1)
        
        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(torch.cat([dec1_0, dec2_0, dec3_0, dec4_0, dec5_0], 1)), dim=1)
        else:
            x_out_ = self.final(torch.cat([dec1_0, dec2_0, dec3_0, dec4_0, dec5_0], 1))
            x_out  = torch.tanh(x_out_)

        return x_out
    
    
class MDeNetplus(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder
        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
        """

    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        print("Loading MDeNetplus")
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = torchvision.models.resnet34(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)

        # Encoder 
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        # Decoder 1
        self.dec1_5 = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec1_4 = DecoderBlock(num_filters * 8, num_filters * 8 * 2, num_filters * 4, is_deconv)
        self.dec1_3 = DecoderBlock(num_filters * 4, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec1_2 = DecoderBlock(num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1_1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec1_0 = ConvRelu(num_filters, num_filters)
        
        # Decoder 2
        self.dec2_4 = DecoderBlock(256, num_filters * 8 * 2, num_filters * 4, is_deconv)
        self.dec2_3 = DecoderBlock(num_filters * 4, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2_2 = DecoderBlock(num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec2_1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec2_0 = ConvRelu(num_filters, num_filters)
            
        # Decoder 3
        self.dec3_3 = DecoderBlock(128, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec3_2 = DecoderBlock(num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec3_1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec3_0 = ConvRelu(num_filters, num_filters)
            
        # Decoder 4
        self.dec4_2 = DecoderBlock(64, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec4_1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec4_0 = ConvRelu(num_filters, num_filters)
        
        # Decoder 5
        self.dec5_2 = DecoderBlock(64, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec5_1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec5_0 = ConvRelu(num_filters, num_filters)
        
        # Concatenation layer 
        self.final = nn.Conv2d(5*num_filters, num_classes, kernel_size=1)
        self.final_ = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder 
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        # Decoder 1
        dec1_5 = self.dec1_5(conv5)
        dec1_4 = self.dec1_4(dec1_5)
        dec1_3 = self.dec1_3(dec1_4)
        dec1_2 = self.dec1_2(dec1_3)
        dec1_1 = self.dec1_1(dec1_2)
        dec1_0 = self.dec1_0(dec1_1)

        # Decoder 2
        dec2_4 = self.dec2_4(conv4 +dec1_5)
        dec2_3 = self.dec2_3(dec2_4+dec1_4)
        dec2_2 = self.dec2_2(dec2_3+dec1_3)
        dec2_1 = self.dec2_1(dec2_2+dec1_2)
        dec2_0 = self.dec2_0(dec2_1+dec1_1)
        
        # Decoder 3
        dec3_3 = self.dec3_3(conv3 +dec2_4+dec1_4)
        dec3_2 = self.dec3_2(dec3_3+dec2_3+dec1_3)
        dec3_1 = self.dec3_1(dec3_2+dec2_2+dec1_2)
        dec3_0 = self.dec3_0(dec3_1+dec2_1+dec1_1)
                
        # Decoder 4
        dec4_2 = self.dec4_2(conv2 +dec3_3+dec2_3+dec1_3)
        dec4_1 = self.dec4_1(dec4_2+dec3_2+dec2_2+dec1_2)
        dec4_0 = self.dec4_0(dec4_1+dec3_1+dec2_1+dec1_1)
                        
        # Decoder 5
        dec5_2 = self.dec5_2(conv1 +conv2 +dec3_3+dec2_3+dec1_3)
        dec5_1 = self.dec5_1(dec5_2+dec4_2+dec3_2+dec2_2+dec1_2)
        dec5_0 = self.dec5_0(dec5_1+dec4_1+dec3_1+dec2_1+dec1_1)
        
        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(torch.cat([dec1_0, dec2_0, dec3_0, dec4_0, dec5_0], 1)), dim=1)
        else:
            x_out_ = self.final(torch.cat([dec1_0, dec2_0, dec3_0, dec4_0, dec5_0], 1))
            x_out  = torch.tanh(x_out_)
            
            x_out_1 = self.final_(dec1_0)
            x_out_2 = self.final_(dec2_0)
            x_out_3 = self.final_(dec3_0)
            x_out_4 = self.final_(dec4_0)
            x_out_5 = self.final_(dec5_0)
            
            x_out1 = torch.tanh(x_out_1)
            x_out2 = torch.tanh(x_out_2)
            x_out3 = torch.tanh(x_out_3)
            x_out4 = torch.tanh(x_out_4)
            x_out5 = torch.tanh(x_out_5)

        return x_out, x_out1, x_out2, x_out3, x_out4, x_out5

    
    
    
class hourglass(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder
        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
        """

    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes
        print("Loading hourglass")
        
        ############################# First Layer ###################################
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = torchvision.models.resnet34(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)

        # Encoder 
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4
        
        # Center layer 
        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        # Decoder 
        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlock(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlock(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlock(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        
        
        ############################ Connection  #####################################
        # 1x1 convolution layer 
        self.conv1x1 = nn.Conv2d(num_filters, num_filters, kernel_size=1)
        
        ############################# Second Layer ###################################
        self.encoder_2 = torchvision.models.resnet34(pretrained=True)
        self.encoder_2.conv1 = nn.Conv2d(num_filters, 64, kernel_size=7, stride=2, padding=3)
        
        # Encoder
        self.conv1_2 = nn.Sequential(self.encoder_2.conv1,
                                   self.encoder_2.bn1,
                                   self.encoder_2.relu,
                                   self.pool)
        self.conv2_2 = self.encoder_2.layer1
        self.conv3_2 = self.encoder_2.layer2
        self.conv4_2 = self.encoder_2.layer3
        self.conv5_2 = self.encoder_2.layer4
        
        self.center_2 = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8, is_deconv)
        
        # Decoder 
        self.dec5_2 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4_2 = DecoderBlock(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3_2 = DecoderBlock(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2_2 = DecoderBlock(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1_2 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0_2 = ConvRelu(num_filters, num_filters)
        
        #################################################################################
        # Final layer for both networks 
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        
    def forward(self, x):
        ############################# First stage #######################
        # Encoder 1
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
                

        # Center layer 1
        center = self.center(self.pool(conv5))
        
        # Decoder 2
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            logits = self.final(dec0)
            x_out1  = torch.tanh(logits)

        ############################# Connection ########################
        conv1x1 = self.conv1x1(dec0)
        ############################# Second stage #######################
        # Encoder 1
        conv11 = self.conv1_2(conv1x1)
        conv22 = self.conv2_2(conv11)
        conv33 = self.conv3_2(conv22)
        conv44 = self.conv4_2(conv33)
        conv55 = self.conv5_2(conv44)

        # Center layer 2
        center_2 = self.center_2(self.pool(conv55))
        
        # Decoder 2
        dec55 = self.dec5_2(torch.cat([center_2, conv55], 1))
        dec44 = self.dec4_2(torch.cat([dec55, conv44], 1))
        dec33 = self.dec3_2(torch.cat([dec44, conv33], 1))
        dec22 = self.dec2_2(torch.cat([dec33, conv22], 1))
        dec11 = self.dec1_2(dec22)
        dec00 = self.dec0_2(dec11)
        
        if self.num_classes > 1:
            x_out2 = F.log_softmax(self.final(dec00), dim=1)
        else:
            logits1 = self.final(dec00)
            x_out2  = torch.tanh(logits1)
                     
        return x_out1, x_out2

    
class hourglass_trash(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder
        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
        """

    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        print("Loading hourglass_trash")
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = torchvision.models.resnet34(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)

        # Encoder
        
        
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        # Decoder 1
        self.dec1_5 = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec1_4 = DecoderBlock(num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec1_3 = DecoderBlock(num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec1_2 = DecoderBlock(num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1_1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec1_0 = ConvRelu(num_filters, num_filters)
        
        # Decoder 2
        self.dec2_4 = DecoderBlock(256, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec2_3 = DecoderBlock(num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2_2 = DecoderBlock(num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec2_1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec2_0 = ConvRelu(num_filters, num_filters)
            
        # Decoder 3
        self.dec3_3 = DecoderBlock(128, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec3_2 = DecoderBlock(num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec3_1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec3_0 = ConvRelu(num_filters, num_filters)
            
        # Decoder 4
        self.dec4_2 = DecoderBlock(64, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec4_1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec4_0 = ConvRelu(num_filters, num_filters)
        
        # Decoder 5
        self.dec5_2 = DecoderBlock(64, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec5_1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec5_0 = ConvRelu(num_filters, num_filters)
        
        # Concatenation layer 
        self.final = nn.Conv2d(5*num_filters, num_classes, kernel_size=1)
        
        
        #################################################################################
        # 1x1 convolution layer 
        self.conv1x1 = nn.Conv2d(5*num_filters, 5*num_filters, kernel_size=1)
        self.encoder_2 = torchvision.models.resnet34(pretrained=False)
        self.encoder_2.conv1 = nn.Conv2d(5*num_filters, 64, kernel_size=3, stride=1, padding=1)
        
        # Encoder
        self.conv1_2 = nn.Sequential(self.encoder_2.conv1,
                                   self.encoder_2.bn1,
                                   self.encoder_2.relu,
                                   self.pool)
        self.conv2_2 = self.encoder_2.layer1
        self.conv3_2 = self.encoder_2.layer2
        self.conv4_2 = self.encoder_2.layer3
        self.conv5_2 = self.encoder_2.layer4

        
    def forward(self, x):
        
        # Encoder 
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        # Decoder 1
        dec1_5 = self.dec1_5(conv5)
        dec1_4 = self.dec1_4(dec1_5)
        dec1_3 = self.dec1_3(dec1_4)
        dec1_2 = self.dec1_2(dec1_3)
        dec1_1 = self.dec1_1(dec1_2)
        dec1_0 = self.dec1_0(dec1_1)

        # Decoder 2
        dec2_4 = self.dec2_4(conv4)
        dec2_3 = self.dec2_3(dec2_4)
        dec2_2 = self.dec2_2(dec2_3)
        dec2_1 = self.dec2_1(dec2_2)
        dec2_0 = self.dec2_0(dec2_1)

        # Decoder 3
        dec3_3 = self.dec3_3(conv3)
        dec3_2 = self.dec3_2(dec3_3)
        dec3_1 = self.dec3_1(dec3_2)
        dec3_0 = self.dec3_0(dec3_1)

        # Decoder 4
        dec4_2 = self.dec4_2(conv2)
        dec4_1 = self.dec4_1(dec4_2)
        dec4_0 = self.dec4_0(dec4_1)

        # Decoder 5
        dec5_2 = self.dec5_2(conv1)
        dec5_1 = self.dec5_1(dec5_2)
        dec5_0 = self.dec5_0(dec5_1)
        
        con_cat = torch.cat([dec1_0, dec2_0, dec3_0, dec4_0, dec5_0], 1)
        x_out_  = self.final(con_cat)
        x_out1  = torch.tanh(x_out_)

        
        ###################################################################################
        # Second model 
        conv1x1 = self.conv1x1(con_cat)
        conv11 = self.conv1_2(conv1x1)
        conv22 = self.conv2_2(conv11)
        conv33 = self.conv3_2(conv22)
        conv44 = self.conv4_2(conv33)
        conv55 = self.conv5_2(conv44)

        # Decoder 1
        dec1_55 = self.dec1_5(conv55)
        dec1_44 = self.dec1_4(dec1_55)
        dec1_33 = self.dec1_3(dec1_44)
        dec1_22 = self.dec1_2(dec1_33)
        dec1_11 = self.dec1_1(dec1_22)
        dec1_00 = self.dec1_0(dec1_11)

        # Decoder 2
        dec2_44 = self.dec2_4(conv44)
        dec2_33 = self.dec2_3(dec2_44)
        dec2_22 = self.dec2_2(dec2_33)
        dec2_11 = self.dec2_1(dec2_22)
        dec2_00 = self.dec2_0(dec2_11)

        # Decoder 3
        dec3_33 = self.dec3_3(conv33)
        dec3_22 = self.dec3_2(dec3_33)
        dec3_11 = self.dec3_1(dec3_22)
        dec3_00 = self.dec3_0(dec3_11)

        # Decoder 4
        dec4_22 = self.dec4_2(conv22)
        dec4_11 = self.dec4_1(dec4_22)
        dec4_00 = self.dec4_0(dec4_11)

        # Decoder 5
        dec5_22 = self.dec5_2(conv11)
        dec5_11 = self.dec5_1(dec5_22)
        dec5_00 = self.dec5_0(dec5_11)

        x_out__ = self.final(torch.cat([dec1_00, dec2_00, dec3_00, dec4_00, dec5_00], 1))
        x_out2  = torch.tanh(x_out__)

        return x_out1, x_out2
  
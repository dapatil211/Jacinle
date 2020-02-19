import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from jactorch.io import load_state_dict

import sys

sys.path.append("..")
import ipdb
import os

st = ipdb.set_trace
import functools


def _conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class ResidualConvBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResidualConvBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, incl_gap=False, num_classes=1000):
        super(ResNet, self).__init__()

        self.incl_gap = incl_gap
        self.incl_cls = self.incl_gap and num_classes is not None

        self.inplanes = 64
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if self.incl_gap:
            self.avgpool = nn.AvgPool2d(7, stride=1)
        if self.incl_cls:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

    def reset_parameters(self):
        return reset_resnet_parameters(self)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.incl_gap:
            x = self.avgpool(x)

        if self.incl_cls:
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x


cfgs = {
    "resnet18": (ResidualConvBlock, [2, 2, 2, 2]),
    "resnet34": (ResidualConvBlock, [3, 4, 6, 3]),
    "resnet50": (ResidualConvBottleneck, [3, 4, 6, 3]),
    "resnet101": (ResidualConvBottleneck, [3, 4, 23, 3]),
    "resnet152": (ResidualConvBottleneck, [3, 8, 36, 3]),
}

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def make_resnet(
    net_id, pretrained, incl_gap=True, num_classes=1000, model_location=None
):
    model = ResNet(*cfgs[net_id], incl_gap=incl_gap, num_classes=num_classes)
    if pretrained:
        if model_location:
            model = FeatNet2DWrapper()
            ckpt_names = os.listdir(model_location)
            steps = [int((i.split("-")[1]).split(".")[0]) for i in ckpt_names]
            if len(ckpt_names) > 0:
                step = max(steps)
                model_name = "model-%d.pth" % (step)
                path = os.path.join(model_location, model_name)
                print("...found checkpoint %s" % (path))
                checkpoint = torch.load(path)
                model.load_state_dict(checkpoint["model_state_dict"])
                model = model.featnet2D.resnet
            else:
                print("No checkpoint found")
        else:
            pretrained_model = model_zoo.load_url(model_urls[net_id])
            if num_classes != 1000:
                del pretrained_model["fc.weight"]
                del pretrained_model["fc.bias"]

            try:
                load_state_dict(model, pretrained_model)
            except KeyError:
                pass  # Intentionally ignore the key error.
    return model


def make_resnet_contructor(net_id):
    func = functools.partial(make_resnet, net_id=net_id)
    func.__name__ = net_id
    func.__doc__ = net_id.replace("resnet", "ResNet-")
    return func


resnet18 = make_resnet_contructor("resnet18")
resnet34 = make_resnet_contructor("resnet34")
resnet50 = make_resnet_contructor("resnet50")
resnet101 = make_resnet_contructor("resnet101")
resnet152 = make_resnet_contructor("resnet152")


def reset_resnet_parameters(m, fc_std=0.01, bfc_std=0.001):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, fc_std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Bilinear):
        m.weight.data.normal_(0, bfc_std)
        if m.bias is not None:
            m.bias.data.zero_()
    else:
        for sub in m.modules():
            if m != sub:
                reset_resnet_parameters(sub, fc_std=fc_std, bfc_std=bfc_std)


class FeatNet2DWrapper(nn.Module):
    def __init__(self):
        super(FeatNet2DWrapper, self).__init__()
        self.featnet2D = FeatNet2D()

    def forward(self, depth_g, rgb_g, summ_writer):
        pass


class FeatNet2D(nn.Module):
    def __init__(self):
        super(FeatNet2D, self).__init__()

        print("FeatNet2D...")

        self.resnet = resnet34(pretrained=False, incl_gap=False, num_classes=None)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                32, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)
            ),
        )

        self.net = nn.Sequential(self.resnet, self.decoder)
        self.rgb_layer = nn.Conv2d(
            in_channels=32, out_channels=3, kernel_size=1, stride=1, padding=0
        ).cuda()
        self.depth_layer = nn.Conv2d(
            in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0
        ).cuda()

    def forward(self, depth_g, rgb_g, summ_writer):
        pass

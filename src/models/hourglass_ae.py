import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['HourglassNet', 'hg']


class ConvAct(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride=1):
        super(ConvAct, self).__init__()
        padding = kernel_size / 2

        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = x * self.sigmoid(x)

        return x

class ActConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride=1):
        super(ActConv, self).__init__()
        padding = kernel_size / 2

        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)

    def forward(self, x):
        x = x * self.sigmoid(x)
        x = self.conv(x)

        return x


class Conv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride=1):
        super(Conv, self).__init__()
        padding = kernel_size / 2

        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)

    def forward(self, x):
        x = self.conv(x)

        return x

def make_conv(in_f, out_f, kernel_size, stride=1, mode='ConvAct'):
    if mode == "ConvAct":
        return nn.Sequential(
                ConvAct(in_f, out_f, kernel_size, stride)
                )
    elif mode == "Conv":
        return nn.Sequential(
                Conv(in_f, out_f, kernel_size, stride)
                )
    elif mode == "ActConv":
        return nn.Sequential(
                ActConv(in_f, out_f, kernel_size, stride)
                )


class Hourglass(nn.Module):
    def __init__(self, f):
        super(Hourglass, self).__init__()
        self.n = 4
        self.kernel_size = 3
        self.f = f
        self.g = f >> 1

        self.upsample = nn.Upsample(scale_factor=2)
        self.hg = self._make_hg(self.n)

    def _make_conv(self, in_f, out_f, kernel_size):
        return nn.Sequential(
                ActConv(in_f, out_f, kernel_size)
                )

    def _make_hg(self, n = 4):
        hg = []
        f3 = self.f
        f2 = self.f + self.g
        f1 = self.f + self.g * 2
        f0 = self.f + self.g * 3
        ff = self.f + self.g * 4

        f_difc = {
                3: (f3, f2),
                2: (f2, f1),
                1: (f1, f0)
                }
        for i in range(n):
            tmp = []
            if i == 0:
                tmp.append(self._make_conv(f0, f0, self.kernel_size))
                tmp.append(self._make_conv(f0, ff, self.kernel_size))
                tmp.append(self._make_conv(ff, f0, self.kernel_size))
                tmp.append(self._make_conv(ff, ff, self.kernel_size))
            else:
                tmp.append(self._make_conv(f_difc[i][0], f_difc[i][0], self.kernel_size))
                tmp.append(self._make_conv(f_difc[i][0], f_difc[i][1], self.kernel_size))
                tmp.append(self._make_conv(f_difc[i][1], f_difc[i][0], self.kernel_size))
            hg.append(nn.ModuleList(tmp))
        return nn.ModuleList(hg)

    def _hg_forward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low = F.max_pool2d(x, 2, stride=2)
        low = self.hg[n-1][1](low)

        if n > 1:
            low = self._hg_forward(n-1, low)
        else:
            low = self.hg[n-1][3](low)

        low = self.hg[n-1][2](low)
        up2 = self.upsample(low)

        return up1 + up2

    def forward(self, x):
        return self._hg_forward(self.n, x)

class StageBase(nn.Module):
    def __init__(self, f):
        super(StageBase, self).__init__()
        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.conv1 = make_conv(3, 64, 7, stride=2, mode='ConvAct')
        self.conv2 = make_conv(64, 128, 3, mode="ConvAct")

        self.conv3 = make_conv(128, 128, 3, mode='ConvAct')
        self.conv4 = make_conv(128, 128, 3, mode='ConvAct')

        self.conv5 = make_conv(128, 256, 3, mode='ConvAct')
        self.conv6 = make_conv(256, f, 3, mode='Conv')

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool(x)

        x = self.conv5(x)
        x = self.conv6(x)

        return x

class HourglassNet(nn.Module):

    def __init__(self, f=256, num_stacks=2 , num_classes=14, embedding_len=14):
        super(HourglassNet, self).__init__()

        self.num_stacks = num_stacks

        self.stage_base = self._make(StageBase(f))

        hg, fc1, fc2, score, embedding, fc_, score_ = [], [], [], [], [], [], []

        for i in range(num_stacks):
            hg.append(Hourglass(f))
            fc1.append(make_conv(f, f, kernel_size=3, mode="ActConv"))
            fc2.append(make_conv(f, f, kernel_size=1, mode="ActConv"))
            score.append(make_conv(f, num_classes, kernel_size=1, mode="Conv"))
            embedding.append(make_conv(f, embedding_len, kernel_size=1, mode="Conv"))
            if i < num_stacks-1:
                fc_.append(make_conv(f, f, kernel_size=1, mode="Conv"))
                score_.append(make_conv(num_classes+embedding_len, f, kernel_size=1, mode="Conv"))

        self.hg = nn.ModuleList(hg)
        self.fc1 = nn.ModuleList(fc1)
        self.fc2 = nn.ModuleList(fc2)
        self.score = nn.ModuleList(score)
        self.embedding = nn.ModuleList(embedding)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    def _make(self, target):
        return nn.Sequential(
                target
                )

    def forward(self, x):
        out = []

        x = self.stage_base(x)

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.fc1[i](y)
            y = self.fc2[i](y)
            score = self.score[i](y)
            embedding = self.embedding[i](y)
            out.append((score, embedding))
            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](torch.cat((score, embedding), dim=1))
                x = x + fc_ + score_

        return out


def hg(**kwargs):
    model = HourglassNet(f=kwargs['f'], num_stacks=kwargs['num_stacks'], embedding_len=kwargs['embedding_len'],
                         num_classes=kwargs['num_classes'])
    return model

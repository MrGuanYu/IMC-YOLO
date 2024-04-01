import math

import torch.nn as nn
import torch
import torch.nn.functional as F
from pytorch_wavelets import DWTForward

from ultralytics.nn.modules import Conv


class myChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16,local_size=5, local_weight=0.5):
        super(myChannelAttention, self).__init__()

        self.local_size = local_size
        self.local_weight = local_weight


        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.local_avg_pool = nn.AdaptiveAvgPool2d(local_size)
        self.local_max_pool = nn.AdaptiveAvgPool2d(local_size)


        # t = int(abs(math.log(in_planes, 2) + self.b) / self.gamma)  # eca  gamma=2
        # k = t if t % 2 else t + 1

        # self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # global_avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        local_avg_out = self.fc2(self.relu1(self.fc1(self.local_avg_pool(x))))

        # global_max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        local_max_out = self.fc2(self.relu1(self.fc1(self.local_max_pool(x))))

        # global_avg_out = F.adaptive_avg_pool2d(global_avg_out, [self.local_size, self.local_size])
        # global_max_out = F.adaptive_max_pool2d(global_max_out, [self.local_size, self.local_size])

        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))

        # avg_out = global_avg_out * (1-self.local_weight) + local_avg_out * self.local_weight
        # max_out = global_max_out * (1-self.local_weight) + local_max_out * self.local_weight

        # out = avg_out + max_out
        out = local_avg_out + local_max_out
        return self.sigmoid(out)


class mySpatialAttention(nn.Module):
    def __init__(self):
        super(mySpatialAttention, self).__init__()

    def __init__(self, kernel_size=7):
        super(mySpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class mutiSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(mutiSpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)


        return self.sigmoid(x)


class mutiChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(mutiChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mid_avg_pool = nn.AdaptiveAvgPool2d(3)
        self.mid_max_pool = nn.AdaptiveMaxPool2d(3)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.fc3 = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=3, stride=1,padding=1)
        self.fc4 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=3, stride=1,padding=1)

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))

        print("zzz")
        print(avg_out.size())

        mid_avg_out = self.fc4(self.relu1(self.fc3(self.mid_avg_pool(x))))
        mid_max_out = self.fc4(self.relu1(self.fc3(self.mid_max_pool(x))))

        out1 = avg_out + max_out

        out2 = mid_avg_out + mid_max_out


        return self.sigmoid(out1),self.sigmoid(out2)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class myCBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(myCBAM, self).__init__()
        # self.ca = myChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
        self.ca = ChannelAttention(in_planes, ratio)
        # self.myca = myChannelAttention(in_planes, ratio)

        self.wt = DWTForward(J=1, mode='zero', wave='haar')

    def forward(self, x):
        # out = x * self.ca(x)
        yL, yH = self.wt(x)
        # print(yL.size())
        # print(yH.size())
        y_HL = yH[0][:, :, 0, ::]
        # print(y_HL.shape)
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]

        # ca_temp =  F.adaptive_avg_pool2d(self.ca(x), (x.size()[2],x.size()[3]))
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


class mycbamBottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.cbam = myCBAM(c1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cbam(self.cv2(self.cv1(x))) if self.add else self.cbam(self.cv2(self.cv1(x)))


class mycbamC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(mycbamBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

if __name__ == '__main__':
    x = torch.randn(1, 64, 20, 20)
    b, c, h, w = x.shape
    net = mycbamC2f(c,c)
    y = net(x)
    print(y.size())



if __name__ == "__main__":
    # attention = myChannelAttention(32)
    # inputs = torch.randn((16, 32, 64, 64))
    # result = attention(inputs)
    # print(result.shape)

    net = mutiChannelAttention(32)
    inputs = torch.randn((16, 32, 64, 64))
    out = net(inputs)
    print(out)

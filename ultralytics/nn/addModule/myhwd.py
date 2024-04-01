import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse

from ultralytics.nn.modules import Conv


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


class myDown_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(myDown_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.wi = DWTInverse(mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),  # 将 inplace 设置为 False
        )
        self.ca = ChannelAttention(out_ch)
        self.sa = SpatialAttention()

    def forward(self, x):
        yL, yH = self.wt(x)
        # print(yL.size())
        # print(yH[0].size())


        y_HL = yH[0][:, :, 0, :]
        y_LH = yH[0][:, :, 1, :]
        y_HH = yH[0][:, :, 2, :]

        yL = yL * self.ca(yL)
        y_HL = y_HL * self.ca(y_HL)
        y_LH = y_LH * self.ca(y_LH)
        y_HH = y_HH * self.ca(y_HH)

        yH_reconstructed = torch.cat([y_HL.unsqueeze(2), y_LH.unsqueeze(2), y_HH.unsqueeze(2)], dim=2)

        x = self.wi((yL,[yH_reconstructed]))

        x = x * self.sa(x)

        x = self.conv_bn_relu(x)

        return x


class hwdBottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.hwd = myDown_wt(c1, c1)

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        # return  if self.add else self.hwd(self.cv2(self.cv1(x))))
        if self.add:
            return x + self.hwd(self.cv2(self.cv1(x)))
        else:
            out1 = self.cv1(x)
            out2 = self.cv2(out1)
            out3 = self.hwd(out2)

            return out3

class hwdC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(hwdBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

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


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = myDown_wt(64, 64)  # 输入通道数，输出通道数
    input = torch.rand(16, 64, 64, 64)
    output = block(input)
    print(output.size())

    net = hwdC2f(64,64)
    rlt = net(input)
    print(rlt.size())

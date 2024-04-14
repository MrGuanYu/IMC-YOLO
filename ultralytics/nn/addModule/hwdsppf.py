
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse

from ultralytics.nn.modules import Conv


class myhwdAttention(nn.Module):
    def __init__(self):
        super(myhwdAttention, self).__init__()

        self.dwt = DWTForward(J=1, mode='zero', wave='haar')
        self.iwd = DWTInverse(mode='zero', wave='haar')

        self.relu1 = nn.ReLU()

        self.linear1 = nn.Linear(3, 1)
        self.linear2 = nn.Linear(1, 3)

    def forward(self, x):
        yL, yH = self.dwt(x)

        b, c, t, w, h = yH[0].size()

        yH0_permute_view = yH[0].permute(0, 1, 4, 3, 2).reshape(b, -1, t).contiguous()


        att = self.linear2(self.relu1(self.linear1(yH0_permute_view))).reshape(b, c, h, w, t).permute(0, 1, 4, 3, 2)

        yH[0] = yH[0] * att

        out = self.iwd((yL, [yH[0]]))

        out = F.pad(out, (0, x.size(-1) - out.size(-1), 0, x.size(-2) - out.size(-2)))

        return out

class myhwdSPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.hwd = myhwdAttention()

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.hwd(x)
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))




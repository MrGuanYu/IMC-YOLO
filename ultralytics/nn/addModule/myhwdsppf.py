
import torch.nn as nn
import torch
from pytorch_wavelets import DWTForward, DWTInverse


class myhwdAttention(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(myhwdAttention, self).__init__()

        self.dwt = DWTForward(J=1, mode='zero', wave='haar')
        self.iwd = DWTInverse(mode='zero', wave='haar')

        self.relu1 = nn.ReLU()

        self.linear1 = nn.Linear(3, 1)
        self.linear2 = nn.Linear(1, 3)

    def forward(self, x):
        yL, yH = self.dwt(x)

        b, c, t, w, h = yH[0].size()

        yH0_permute_view = yH[0].permute(0, 1, 4, 3, 2).reshape(b, -1, t)

        yH_permute_view = yH0_permute_view

        att = self.linear2(self.relu1(self.linear1(yH_permute_view))).reshape(b, c, h, w, t).permute(0, 1, 4, 3, 2)

        yH[0] = yH[0] * att

        out = self.iwd((yL, [yH[0]]))

        return out
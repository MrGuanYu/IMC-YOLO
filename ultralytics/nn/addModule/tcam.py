import torch.nn as nn
import torch


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out


class myChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(myChannelAttention, self).__init__()

        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)


        self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.max_pool_w = nn.AdaptiveMaxPool2d((1, None))
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.max_pool_h = nn.AdaptiveMaxPool2d((None, 1))

        # t = int(abs(math.log(in_planes, 2) + self.b) / self.gamma)  # eca  gamma=2
        # k = t if t % 2 else t + 1

        # self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # global_avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))

        # global_max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))

        # global_avg_out = F.adaptive_avg_pool2d(global_avg_out, [self.local_size, self.local_size])
        # global_max_out = F.adaptive_max_pool2d(global_max_out, [self.local_size, self.local_size])

        out1 = self.fc2(self.relu1(self.fc1(self.max_pool_w(self.avg_pool_h(x)))))
        out2 = self.fc2(self.relu1(self.fc1(self.max_pool_h(self.avg_pool_w(x)))))

        # avg_out = global_avg_out * (1-self.local_weight) + local_avg_out * self.local_weight
        # max_out = global_max_out * (1-self.local_weight) + local_max_out * self.local_weight

        out = out1 + out2

        return self.sigmoid(out)



class TCAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(TCAM, self).__init__()
        self.ca = myChannelAttention(in_planes, ratio)
        self.ta = TripletAttention(no_spatial=False)

    def forward(self, x):
        out = x * self.ca(x)
        # out1 = out * self.sa(out)
        result = self.ta(out)
        # result = self.hwd(out1)
        return result

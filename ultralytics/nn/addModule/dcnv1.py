import numpy as np
import torch.nn as nn
import torch


class DeformConv2D(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, bias=None):
        super(DeformConv2D, self).__init__()
        self.kernel_size = kernel_size  # 卷积核的大小
        self.padding = padding  # 填充大小
        self.zero_padding = nn.ZeroPad2d(padding)  # 用0去填充
        self.conv_kernel = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)  # 二维卷积

    def forward(self, x, offset):
        '''
        x: 原始输入 [b,c,h,w]
        offset: 每个像素点的偏移 [b,2*N,h,w]
        N:kernel中元素的个数 = k*k
        offset 和 x的宽高相同，表示的是对应位置像素点的偏移
        offset 的第二个维度大小是2N, 当N=9时，排列顺序是（x1, y1, x2, y2, ...x18，y18）
        xi表示一个大小为[h,w]的张量，表示对于原始图像中的每个点，对于kernel中的第i个元素，在x轴方向的偏移量
        yi表示一个大小为[h,w]的张量，表示对于原始图像中的每个点，对于kernel中的第i个元素，在y轴方向的偏移量
        '''
        dtype = offset.data.type()  # 获取偏移 offset 张量的数据类型，通常会与输入 x 的数据类型相匹配
        ks = self.kernel_size
        N = offset.size(1) // 2

        '''
        下面这段代码的整体功能：将offset中第二维度的顺序从[x1, y1, x2, y2, ...] 变成[x1, x2, .... y1, y2, ...]
        '''

        # 创建一个索引张量 offsets_index，用于重新排列 offset 张量中的偏移项的顺序，将 x 和 y 分量分开排列
        offsets_index = torch.Tensor(torch.cat([torch.arange(0, 2 * N, 2), torch.arange(1, 2 * N + 1, 2)]),
                                     requires_grad=False).type_as(x).long()
        # torch.arange(0, 2*N, 2)：这部分代码创建了一个从 0 到 2*N-1 的整数序列，步长为 2。这个序列包含了偏移项的 x 分量的索引，因为在 offset 张量中，x 和 y 分量是交替存储的。所以这部分代码创建了一个形如 [0, 2, 4, ...] 的整数序列。
        # torch.arange(1, 2*N+1, 2)：这部分代码创建了一个从 1 到 2*N 的整数序列，步长为 2。这个序列包含了偏移项的 y 分量的索引，也是交替存储的。所以这部分代码创建了一个形如 [1, 3, 5, ...] 的整数序列。
        # torch.cat([..., ...])：torch.cat 函数用于将两个张量连接在一起，这里将上面两个整数序列连接起来，得到一个包含 x 和 y 分量索引的整数序列。结果形如 [0, 2, 4, ..., 1, 3, 5, ...]。
        # Variable(...)：将上述整数序列转换为 PyTorch 的 Variable 对象，这是为了能够在 PyTorch 中进行计算。requires_grad=False 表示这个 Variable 对象不需要计算梯度。
        # .type_as(x)：将数据类型设置为与输入张量 x 相同的数据类型，以确保数据类型一致。
        # .long()：将整数类型转换为长整数类型，以适应后续的索引操作。
        # 最终，offsets_index 是一个包含了偏移项 x 和 y 分量的索引的张量，它的大小为 [1, 2*N, 1, 1]，其中 N 表示偏移项的数量，通常是卷积核的大小。这个索引张量将在后续代码中用于重新排列 offset 张量的顺序，以方便后续计算。

        # 当b=1,N=9时，offsets_index=[ 0,  2,  4,  6,  8, 10, 12, 14, 16,  1,  3,  5,  7,  9, 11, 13, 15, 17]
        # offsets_index的大小为[18]
        offsets_index = offsets_index.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1).expand(
            *offset.size())  # 将 offsets_index 调整为与 offset 张量相同的形状
        # offsets_index.unsqueeze(dim=0)：这一步在 offsets_index 上应用 unsqueeze 操作，将维度 0 扩展（增加）一次。这使得 offsets_index 从原来的形状 (18,) 变为 (1, 18)，其中 1 是新的维度。
        # offsets_index.unsqueeze(dim=-1)：接下来，在 offsets_index 上再次应用 unsqueeze 操作，但这次是在最后一个维度上扩展。这将使 offsets_index 的形状从 (1, 18) 变为 (1, 18, 1)。
        # offsets_index.unsqueeze(dim=-1).expand(*offset.size())：最后，使用 expand 函数将 offsets_index 扩展到与 offset 相同的形状。这通过在 offsets_index 上进行广播操作，使其形状变为 (batch_size, 18, height, width)，其中 batch_size 是输入 offset 张量的批处理大小，而 18 是因为偏移项有 18 个元

        # 然后unsqueeze扩展维度，offsets_index大小为[1,18,1,1]
        # expand后，offsets_index的大小为[1,18,h,w]
        offset = torch.gather(offset, dim=1, index=offsets_index)  # 重新排列 offset 张量的维度顺序，将偏移项的 x 和 y 分量排列在一起，而不是交替排列
        # offset: 原始的偏移张量，其形状为 [batch_size, 2*N, height, width]，其中 N 是偏移项的数量，每个偏移项包括 x 和 y 两个分量。
        # dim=1: 这是 torch.gather 函数中的维度参数，表示在哪个维度上进行索引和收集操作。在这里，dim=1 表示我们要在 offset 的第二维度（从0开始计数）上进行索引和收集操作。
        # index=offsets_index: 这是用于索引的索引张量，它告诉 torch.gather 函数应该如何重新排列原始的 offset 张量。offsets_index 的形状为 [1, 18, height, width]，其中每个元素是一个整数索引，用于指定如何重新排列原始偏移项的顺序。这个索引张量的值控制了 x 和 y 分量的排列顺序，使它们排列在一起

        # 根据维度dim按照索引列表index将offset重新排序，得到[x1, x2, .... y1, y2, ...]这样顺序的offset
        # ------------------------------------------------------------------------

        # 对输入x进行padding
        if self.padding:
            x = self.zero_padding(x)

        # p表示求偏置后，每个点的位置
        p = self._get_p(offset, dtype)  # (b, 2N, h, w)
        # p.contiguous(): 这一步是为了确保张量 p 在内存中是连续的。PyTorch 中的张量可以以不同的存储方式存在，有些情况下可能不是连续的。这一步会重新排列存储顺序，以确保数据是按顺序排列的，这在后续操作中往往是必要的
        p = p.contiguous().permute(0, 2, 3, 1)  # (b,h,w,2N)

        q_lt = torch.Tensor(p.data, requires_grad=False).floor()  # floor是向下取整     因为在程序中（0，0）点在左上角
        q_rb = q_lt + 1  # 上取整
        # +1相当于向上取整，这里为什么不用向上取整函数呢？是因为如果正好是整数的话，向上取整跟向下取整就重合了，这是我们不想看到的。

        # q_lt[..., :N]代表x方向坐标，大小[b,h,w,N], clamp将值限制在0~h-1
        # q_lt[..., N:]代表y方向坐标, 大小[b,h,w,N], clamp将值限制在0~w-1
        # cat后，还原成原大小[b,h,w,2N]
        # 确保左上角点q_lt和右下q_rb的 x 和 y 坐标不超出输入图像的范围
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()  # 将q_lt中的值控制在图像大小范围内 [b,h,w,2N]
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()  # 将q_rt中的值控制在图像大小范围内 [b,h,w,2N]
        '''
        获取采样后的点周围4个方向的像素点
        q_lt:  left_top 左上
        q_rb:  right_below 右下
        q_lb:  left_below 左下
        q_rt:  right_top 右上
        '''
        # 获得lb   左上角x坐标与右下角y坐标拼接
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], -1)  # [b,h,w,2N]
        # 获得rt
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], -1)  # [b,h,w,2N]
        '''
        插值的时候需要考虑一下padding对原始索引的影响 
        p[..., :N]  采样点在x方向(h)的位置   大小[b,h,w,N]
        p[..., :N].lt(self.padding) : p[..., :N]中小于padding 的元素，对应的mask为true
        p[..., :N].gt(x.size(2)-1-self.padding): p[..., :N]中大于h-1-padding 的元素，对应的mask为true 

        图像的宽(或高)度我们假设为W，填充值我们设为pad，填充后图像的实际宽度为 W+2*pad。因此小于pad大于填充后图像的实际宽度-pad-1的就是在原始图像外的东西
        如图像宽度W=5，填充pad=1，那么填充后图像宽度为5+2*1=7，原图像点索引范围是1-5；当索引大于7-1-1=5时,就超出了原图像边界

           p[..., N:]  采样点在y方向(w)的位置   大小[b,h,w,N]
        p[..., N:].lt(self.padding) : p[..., N:]中小于padding 的元素，对应的mask为true
        p[..., N:].gt(x.size(2)-1-self.padding): p[..., N:]中大于w-1-padding 的元素，对应的mask为true 
        cat之后，大小为[b,h,w,2N]
        '''
        mask = torch.cat([p[..., :N].lt(self.padding) + p[..., :N].gt(x.size(2) - 1 - self.padding),
                          p[..., N:].lt(self.padding) + p[..., N:].gt(x.size(3) - 1 - self.padding)], dim=-1).type_as(
            p)  #
        # mask不需要反向传播
        mask = mask.detach()
        # p - (p - torch.floor(p))相当于torch.floor(p)
        floor_p = p - (p - torch.floor(p))

        '''
        mask为1的区域就是padding的区域
        p*(1-mask) : mask为0的 非padding区域的p被保留
        floor_p*mask: mask为1的  padding区域的floor_p被保留

        可变形卷积引入了一个新的因素，即采样点的偏移。偏移后的采样点可能会落在填充区域内，这时应该如何处理这些点的位置信息呢？
        对于非填充区域的采样点，我们希望保持原有的位置信息，因为这些点是图像中的有效信息。
        对于填充区域的采样点，由于它们落在填充区域内，直接使用原始位置信息可能不合适，因为这些点在填充区域可能没有意义。此时，取整后的位置信息更符合填充区域的特性，因为它将采样点约束在填充区域内的整数坐标上，以适应卷积操作。
        '''
        p = p * (1 - mask) + floor_p * mask
        # 修正坐标信息 p，以确保采样位置不会超出输入图像的边界
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # 双线性插值的系数 大小均为 (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        # (1+左上角的点x - 原始采样点x)*(1+左上角的点y - 原始采样点y)           代表左上角的权重
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        # (1-(右下角的点x - 原始采样点x))*(1-(右下角的点y - 原始采样点y))       代表右下角的权重
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        # (1+左下角的点x - 原始采样点x)*(1+左上角的点y - 原始采样点y)           代表左下角的权重
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
        # (1-(右上角的点x - 原始采样点x))*(1-(右上角的点y - 原始采样点y))       代表右上角的权重

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)  # 左上角的点在原始图片中对应的真实像素值
        x_q_rb = self._get_x_q(x, q_rb, N)  # 右下角的点在原始图片中对应的真实像素值
        x_q_lb = self._get_x_q(x, q_lb, N)  # 左下角的点在原始图片中对应的真实像素值
        x_q_rt = self._get_x_q(x, q_rt, N)  # 右上角的点在原始图片中对应的真实像素值

        # 双线性插值算法
        # x_offset : 偏移后的点再双线性插值后的值  大小(b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt
        '''
        偏置点含有九个方向的偏置，_reshape_x_offset() 把每个点9个方向的偏置转化成 3×3 的形式，
        于是就可以用 3×3 stride=3 的卷积核进行 Deformable Convolution，
        它等价于使用 1×1 的正常卷积核（包含了这个点9个方向的 context）对原特征直接进行卷积。
        '''

        x_offset = self._reshape_x_offset(x_offset, ks)  # (b,c,h*ks,w*ks)

        out = self.conv_kernel(x_offset)

        return out

    # 功能：求每个点的偏置方向.在可变形卷积中，每个像素点需要学习的是对应卷积核的若干个方向的偏置，这些偏置方向在 _get_p_n 方法中生成
    def _get_p_n(self, N, dtype):
        # N=kernel_size*kernel_size
        # 生成了 p_n_x 和 p_n_y，它们表示了一个二维网格的 x 和 y 坐标值。这个网格的大小是 (kernel_size, kernel_size)，并且它的中心点是 (0, 0)。meshgrid 函数用于生成这个坐标网格
        p_n_x, p_n_y = np.meshgrid(range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
                                   range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1), indexing='ij')
        # (2N, 1)
        # 通过 np.concatenate 将 p_n_x 和 p_n_y 沿着指定的轴（默认为第0轴，即按行连接）连接在一起，形成一个大小为 (2N, 1) 的一维数组 p_n。这个一维数组包含了所有方向的偏置
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten()))
        # 使用 np.reshape 将 p_n 重新形状为 (1, 2*N, 1, 1)，使其符合 PyTorch 张量的形状要求。这是一个四维张量，第一个维度为 1，表示 batch size，第二个维度为 2*N，表示偏置方向的个数，而后两个维度为 1，表示空间维度
        p_n = np.reshape(p_n, (1, 2 * N, 1, 1))
        # 将 p_n 转换为 PyTorch 张量，并使用 Variable 包装它。dtype 参数用于指定张量的数据类型，而 requires_grad 设置为 False 表示这个张量不需要梯度计算
        p_n = torch.Tensor(torch.from_numpy(p_n).type(dtype), requires_grad=False)

        return p_n  # [1,2*N,1,1]

    @staticmethod
    # 功能：求每个点的坐标
    def _get_p_0(h, w, N, dtype):
        # 通过 np.meshgrid 创建两个网格，其中一个包含从 1 到 h 的整数，另一个包含从 1 到 w 的整数。其中 p_0_x 包含了高度方向上的坐标，p_0_y 包含了宽度方向上的坐标。这两个变量形成了图像上每个像素点的 x 和 y 坐标信息
        p_0_x, p_0_y = np.meshgrid(range(1, h + 1), range(1, w + 1), indexing='ij')
        # 这两行将坐标信息展平，并在相应的维度上重复 N 次，以便与偏移的维度匹配
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w).repeat(N, axis=1)  # (1,N,h,w)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w).repeat(N, axis=1)  # (1,N,h,w)
        # 将 p_0_x 和 p_0_y 沿着第一个维度（axis=1，即通道维度）拼接在一起，得到形状为 (1, 2*N, h, w) 的张量 p_0
        p_0 = np.concatenate((p_0_x, p_0_y), axis=1)
        # 将 p_0 转换为 PyTorch 的张量，并设置其数据类型为 dtype。使用 Variable 包装这个张量，以便在 PyTorch 中使用它，并设置 requires_grad 为 False，表示不需要计算梯度
        p_0 = torch.Tensor(torch.from_numpy(p_0).type(dtype), requires_grad=False)

        return p_0  # (1,2*N,h,w)

    # 求最后的偏置后的点=每个点的坐标+偏置方向+偏置
    def _get_p(self, offset, dtype):
        '''
        offset: 每个像素点的偏移 [b,2*N,h,w]
        N:kernel中元素的个数 = k*k
        offset 和 x的宽高相同，表示的是对应位置像素点的偏移
        offset 的第二个维度大小是2N, 当N=9时，排列顺序是（x1, y1, x2, y2, ...x18，y18）
        xi表示一个大小为[h,w]的张量，表示对于原始图像中的每个点，对于kernel中的第i个元素，在x轴方向的偏移量
        yi表示一个大小为[h,w]的张量，表示对于原始图像中的每个点，对于kernel中的第i个元素，在y轴方向的偏移量
        '''
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        p_n = self._get_p_n(N, dtype)  # 偏置方向：(1, 2N, 1, 1)
        p_0 = self._get_p_0(h, w, N, dtype)  # 每个点的坐标：(1, 2N, h, w)
        p = p_0 + p_n + offset  # 最终点的位置
        return p  # (1,2N,h,w)

    # 求出p点周围四个点的像素
    # 获取偏移后的点在原始图像中对应的真实像素值，即根据偏移后的位置信息，获取原始图像中相应点的像素值
    def _get_x_q(self, x, q, N):
        # x:[b,c,h',w']
        # q:[b,h,w,2N]
        # q可能为q_lt,q_rt,q_lb,q_rb
        b, h, w, _ = q.size()
        padded_w = x.size(3)  # w'
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)  # (b, c, h*w)
        # 将图片压缩到1维，方便后面的按照index索引提取

        # q[...,:N]  (b,h,w,N) 原始图像中(h_i,w_j)的点在偏移后，向左上角取整对应的点，在N个区域中，x方向的偏移量
        # q[...,N:]  (b,h,w,N) 原始图像中(h_i,w_j)的点在偏移后，向左上角取整对应的点，在N个区域中，y方向的偏移量
        # index:  (b,h,w,N) 原始图像中(h_i,w_j)的点在偏移后，向左上角取整对应的点，在N个区域中，x*w + y
        index = q[..., :N] * padded_w + q[..., N:]  # 大小(b, h, w, N)
        # 这个目的就是将index索引均匀扩增到图片一样的h*w大小

        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        '''
        unsqueeze后 (b,1,h,w,N)
        expand后 (b,c,h,w,N)
        view后 (b, c, h*w*N) 其中每一个值对应一个index
        '''
        # 双线性插值法就是4个点再乘以对应与 p 点的距离。获得偏置点 p 的值，这个 p 点是 9 个方向的偏置所以最后的 x_offset 是 b×c×h×w×9。
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        # x :(b,c,h*w)
        # gather后： (b,c,h*w*N)
        # view后：(b,c,h,w,N)
        return x_offset  # (b,c,h,w,N) 左上角的点在原始图像中对应的像素值

    # _reshape_x_offset() 把每个点9个方向的偏置转化成 3×3 的形式
    # 将 x_offset 张量中的像素值重新排列，使每个像素点周围都包含了 ks*ks 个方向的像素值，以便进行可变形卷积操作
    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        # x_offset : (b, c, h, w, N)
        # ks: kernel_size
        # N=ks*ks
        b, c, h, w, N = x_offset.size()
        '''
        当ks=3,N=9时:
        s=0 [...,0:3]  (b,c,h,w,3)->(b,c,h,w*3)
        s=3 [...,3:6]  (b,c,h,w,3)->(b,c,h,w*3)
        s=6 [...,6:9]  (b,c,h,w,3)->(b,c,h,w*3)
        cat 后 (b,c,h,w*9)
        view 后(b,c,h*3,w*3)
        '''
        # x_offset[..., s:s + ks] 表示在前面的维度（通常是前三维）保持不变的情况下，对最后一个维度进行切片操作
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)  # (b,c,h*3,w*3)

        return x_offset  # (b,c,h*ks,w*ks)
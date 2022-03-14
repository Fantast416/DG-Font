# coding=gbk
from copyreg import constructor
from numpy import size
import torch
import torch.nn.functional as F
from torch import nn


class ResBlocks(nn.Module):
    """
        一组ResBlock的集合
    """

    def __init__(self, num_blocks, dim, norm, act, pad_type, use_sn=False):
        """
        Args:
            num_blocks ([type]): [包含的ResBlock的数目]
            dim ([type]): [特征图数量、维度]
            norm (str, optional): [正则化方式,可选的有bn/in/adain/none四种]. 
            act (str, optional): [激活函数,可选的有relu/lrelu/tanh/none]. 
            pad_type (str, optional): [填充方式,可选的有reflect/replicate/zero].
            use_sn (bool, optional): [是否采用谱归一化spectral_norm，其实一种训练GAN时的技巧]. Defaults to False.
        """
        super(ResBlocks, self).__init__()
        self.model = nn.ModuleList()
        for i in range(num_blocks):
            self.model.append(ResBlock(dim, norm=norm, act=act, pad_type=pad_type, use_sn=use_sn))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    """
        两个kernal_size=3,stride=1,padding=1的2D卷积层组成的一组残差模块
        input--->Conv2D--->Conv2D---> + ---> output
          |____>_____>______>_____>___|
    """

    def __init__(self, dim, norm='in', act='relu', pad_type='zero', use_sn=False):
        """
        Args:
            dim ([type]): [特征图数量]
            norm (str, optional): [正则化方式,可选的有bn/in/adain/none四种]. 
            act (str, optional): [激活函数,可选的有relu/lrelu/tanh/none]. 
            pad_type (str, optional): [填充方式,可选的有reflect/replicate/zero].
            use_sn (bool, optional): [是否采用谱归一化spectral_norm，其实一种训练GAN时的技巧]. Defaults to False.
        """
        super(ResBlock, self).__init__()
        self.model = nn.Sequential(Conv2dBlock(dim, dim, 3, 1, 1,
                                               norm=norm,
                                               act=act,
                                               pad_type=pad_type, use_sn=use_sn),
                                   Conv2dBlock(dim, dim, 3, 1, 1,
                                               norm=norm,
                                               act='none',
                                               pad_type=pad_type, use_sn=use_sn))

    def forward(self, x):
        x_org = x
        residual = self.model(x)
        out = x_org + 0.1 * residual
        return out


class ActFirstResBlk(nn.Module):
    """
        一系列综合操作，得看用在哪儿？
    """

    def __init__(self, dim_in, dim_out, downsample=True):
        """
        Args:
            dim_in ([type]): [用于内置卷积层的，输入特征图数量]
            dim_out ([type]): [用于内置卷积层的，输出特征图数量]
            downsample (bool, optional): [是否下采样,如果为True的话,会有平均池化层夹杂]. Defaults to True.
        """
        super(ActFirstResBlk, self).__init__()
        self.norm1 = FRN(dim_in)
        self.norm2 = FRN(dim_in)
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.downsample = downsample
        self.learned_sc = (dim_in != dim_out)  # learned_sc 为 false === 输入输出维度一致
        if self.learned_sc:  ## ？？
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)  # 加一层 1*1 的卷积层，用来减少参数？

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        x = self.norm1(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        x = self.norm2(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x1 = torch.rsqrt(torch.tensor(2.0))  # sqrt(2)
        x2 = x1 * self._shortcut(x)
        x3 = x2 + torch.rsqrt(torch.tensor(2.0))
        return x3 * self._residual(x)


class LinearBlock(nn.Module):
    """
        整个类就是起到了一个整合封装多样式全连接层FC的作用，将输入先过fc，
        然后如果有norm操作就做norm，然后如果有activation操作就做activation，最后输出
    """

    def __init__(self, in_dim, out_dim, norm='none', act='relu', use_sn=False):
        """
        Args:
            in_dim ([int]): [输入的特征的维度数量]
            out_dim ([int]): [输出的特征的维度数量]
            norm (str, optional): [正则化方式,可选的有bn/in/adain/none四种]. Defaults to 'none'.
            act (str, optional): [激活函数,可选的有relu/lrelu/tanh/none]. Defaults to 'relu'.
            use_sn (bool, optional): [是否采用谱归一化spectral_norm，其实一种训练GAN时的技巧]. Defaults to False.
        """
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.Linear(in_dim, out_dim, bias=use_bias)
        if use_sn:
            self.fc = nn.utils.spectral_norm(self.fc)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if act == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif act == 'tanh':
            self.activation = nn.Tanh()
        elif act == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(act)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class Conv2dBlock(nn.Module):
    """
        整个类就是起到了一个整合多样式卷积Block的作用，将输入先padding，然后再卷积，
        然后如果有norm操作就做norm，然后如果有activation操作就做activation，最后输出
    """

    def __init__(self, in_dim, out_dim, ks, st, padding=0,
                 norm='none', act='relu', pad_type='zero',
                 use_bias=True, use_sn=False):
        """
        Args:
            in_dim ([int]): [输入特征图的数量]
            out_dim ([int]): [输出特征图的数量]
            ks ([type]): [kernel_size卷积核大小]
            st ([type]): [stride卷积步长]
            padding (int, optional): [卷积周边填充]. Defaults to 0.
            norm (str, optional): [正则化方式,可选的有bn/in/adain/none四种]. Defaults to 'none'.
            act (str, optional): [激活函数,可选的有relu/lrelu/tanh/none]. Defaults to 'relu'.
            pad_type (str, optional): [填充方式,可选的有reflect/replicate/zero]. Defaults to 'zero'.
            use_bias (bool, optional): [是否启用Bias偏置项]. Defaults to True.
            use_sn (bool, optional): [是否采用谱归一化spectral_norm，其实一种训练GAN时的技巧]. Defaults to False.
        """
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias

        # initialize padding,定义self.pad 为
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'adain':
            self.norm = AdaIN2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if act == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif act == 'tanh':
            self.activation = nn.Tanh()
        elif act == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(act)

        self.conv = nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias)
        if use_sn:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class FRN(nn.Module):
    """
        FRN模块，输入为[B,C,H,W] , 输出为[B,C,H,W]
    """

    def __init__(self, num_features, eps=1e-6):
        super(FRN, self).__init__()
        self.tau = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.eps = eps

    def forward(self, x):
        # input x  shape [B,C,H,W] (batch_size , channel , height , width)
        x1 = torch.mean(x ** 2, dim=[2, 3], keepdim=True)
        # x1 shape [B,C,1,1] (batch_size , channel , 1 , 1)
        x2 = x1 + self.eps
        # x2 shape [B,C,1,1] (batch_size , channel , 1 , 1),  【+ 为逐元素相加，不改变维度】
        x3 = torch.rsqrt(x2)
        # x3 shape [B,C,1,1] (batch_size , channel , 1 , 1),  【sqrt 为逐元素操作，不改变维度】
        x4 = x * x3
        # x4 shape [B,C,H,W] (batch_size , channel , height , width)
        # 广播机制,逐元素相乘  [B,C,H,W] * [B,C,1,1] = [B,C,H,W]
        output = torch.max(self.gamma * x4 + self.beta, self.tau)

        ## output shape [B,C,H,W] (batch_size , channel , height , width)  torch.max 逐元素操作
        return output


class AdaIN2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=True):
        """
        Args:
            num_features ([type]): [特征通道数目C]
            eps ([type], optional): [description]. Defaults to 1e-5.
            momentum (float, optional): [description]. Defaults to 0.1.
            affine (bool, optional): [是否定义映射参数]. Defaults to False.
            track_running_stats (bool, optional): [description]. Defaults to True.
        """
        super(AdaIN2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))  # 定义可训练的参数，用的就不是AdaIN了
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.weight = None  # 此处，实例化的时候赋值为None，之后会由外面的函数（generator.py里的assign_adain_params函数）对其进行赋值，赋的值不是可学习得参数，而是从style那边计算得到的
            self.bias = None  # 此处，实例化的时候赋值为None，之后会由外面的函数（generator.py里的assign_adain_params函数）对其进行赋值，赋的值不是可学习得参数，而是从style那边计算得到的

        if self.track_running_stats:
            # 为running_mean和running_var注册一块Buffer，Buffer作为该Module的一部分，但是并不是可训练参数。用于跟踪训练状态
            # 同时，默认注册的该Buffer在你存储模型的时候，会被写入state_dict中
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)

    def forward(self, x):
        # print(f"self.num_features:{self.num_features}")
        # print(f"self.affine:{self.affine}")
        # print(f"self.weight:{self.weight}")
        # print(f"x.shape:{x.shape}")

        # assert（断言）用于判断一个表达式，在表达式条件为 False 的时候触发异常。
        assert self.weight is not None and self.bias is not None, "AdaIN params are None"

        N, C, H, W = x.size()

        running_mean = self.running_mean.repeat(N)
        running_var = self.running_var.repeat(N)
        # input x  shape [N,C,H,W] (batch_size , channel , height , width)

        x_ = x.contiguous().view(1, N * C, H * W)
        # x_       shape [1,N*C,H*W] (1,batch_size*channel,height*width)
        # 因为此处x_ 已经是3D的张量了，所以下方的batch_norm函数，必定是BatchNorm1D，其输入应当为（N'，C'，L'）
        # 此处BatchNorm1D N' = 1, C' = N * C, L' = H * W, 然后会对于C'的每一项,计算（N',L'）的统计数据
        # 其实对应过来，就是做了一件 对每一个实例的每一个Channel。计算 (1,H*W) 的统计数据,其实就是完成了一次 Instance Norm

        normed = F.batch_norm(x_, running_mean, running_var,
                              self.weight, self.bias,
                              True, self.momentum, self.eps)

        # normed   shape [1,N*C,H*W] (1,batch_size*channel,height*width)
        return normed.view(N, C, H, W)

    def __repr__(self):
        """
          当print()的时候，输出额外的提示信息。
        """
        return self.__class__.__name__ + '(num_features=' + str(self.num_features) + ')'


if __name__ == '__main__':
    print("CALL blocks.py")

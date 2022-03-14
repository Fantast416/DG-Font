# coding=gbk
from copyreg import constructor
from numpy import size
import torch
import torch.nn.functional as F
from torch import nn


class ResBlocks(nn.Module):
    """
        һ��ResBlock�ļ���
    """

    def __init__(self, num_blocks, dim, norm, act, pad_type, use_sn=False):
        """
        Args:
            num_blocks ([type]): [������ResBlock����Ŀ]
            dim ([type]): [����ͼ������ά��]
            norm (str, optional): [���򻯷�ʽ,��ѡ����bn/in/adain/none����]. 
            act (str, optional): [�����,��ѡ����relu/lrelu/tanh/none]. 
            pad_type (str, optional): [��䷽ʽ,��ѡ����reflect/replicate/zero].
            use_sn (bool, optional): [�Ƿ�����׹�һ��spectral_norm����ʵһ��ѵ��GANʱ�ļ���]. Defaults to False.
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
        ����kernal_size=3,stride=1,padding=1��2D�������ɵ�һ��в�ģ��
        input--->Conv2D--->Conv2D---> + ---> output
          |____>_____>______>_____>___|
    """

    def __init__(self, dim, norm='in', act='relu', pad_type='zero', use_sn=False):
        """
        Args:
            dim ([type]): [����ͼ����]
            norm (str, optional): [���򻯷�ʽ,��ѡ����bn/in/adain/none����]. 
            act (str, optional): [�����,��ѡ����relu/lrelu/tanh/none]. 
            pad_type (str, optional): [��䷽ʽ,��ѡ����reflect/replicate/zero].
            use_sn (bool, optional): [�Ƿ�����׹�һ��spectral_norm����ʵһ��ѵ��GANʱ�ļ���]. Defaults to False.
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
        һϵ���ۺϲ������ÿ������Ķ���
    """

    def __init__(self, dim_in, dim_out, downsample=True):
        """
        Args:
            dim_in ([type]): [�������þ����ģ���������ͼ����]
            dim_out ([type]): [�������þ����ģ��������ͼ����]
            downsample (bool, optional): [�Ƿ��²���,���ΪTrue�Ļ�,����ƽ���ػ������]. Defaults to True.
        """
        super(ActFirstResBlk, self).__init__()
        self.norm1 = FRN(dim_in)
        self.norm2 = FRN(dim_in)
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.downsample = downsample
        self.learned_sc = (dim_in != dim_out)  # learned_sc Ϊ false === �������ά��һ��
        if self.learned_sc:  ## ����
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)  # ��һ�� 1*1 �ľ���㣬�������ٲ�����

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
        �������������һ�����Ϸ�װ����ʽȫ���Ӳ�FC�����ã��������ȹ�fc��
        Ȼ�������norm��������norm��Ȼ�������activation��������activation��������
    """

    def __init__(self, in_dim, out_dim, norm='none', act='relu', use_sn=False):
        """
        Args:
            in_dim ([int]): [�����������ά������]
            out_dim ([int]): [�����������ά������]
            norm (str, optional): [���򻯷�ʽ,��ѡ����bn/in/adain/none����]. Defaults to 'none'.
            act (str, optional): [�����,��ѡ����relu/lrelu/tanh/none]. Defaults to 'relu'.
            use_sn (bool, optional): [�Ƿ�����׹�һ��spectral_norm����ʵһ��ѵ��GANʱ�ļ���]. Defaults to False.
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
        �������������һ�����϶���ʽ���Block�����ã���������padding��Ȼ���پ����
        Ȼ�������norm��������norm��Ȼ�������activation��������activation��������
    """

    def __init__(self, in_dim, out_dim, ks, st, padding=0,
                 norm='none', act='relu', pad_type='zero',
                 use_bias=True, use_sn=False):
        """
        Args:
            in_dim ([int]): [��������ͼ������]
            out_dim ([int]): [�������ͼ������]
            ks ([type]): [kernel_size����˴�С]
            st ([type]): [stride�������]
            padding (int, optional): [����ܱ����]. Defaults to 0.
            norm (str, optional): [���򻯷�ʽ,��ѡ����bn/in/adain/none����]. Defaults to 'none'.
            act (str, optional): [�����,��ѡ����relu/lrelu/tanh/none]. Defaults to 'relu'.
            pad_type (str, optional): [��䷽ʽ,��ѡ����reflect/replicate/zero]. Defaults to 'zero'.
            use_bias (bool, optional): [�Ƿ�����Biasƫ����]. Defaults to True.
            use_sn (bool, optional): [�Ƿ�����׹�һ��spectral_norm����ʵһ��ѵ��GANʱ�ļ���]. Defaults to False.
        """
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias

        # initialize padding,����self.pad Ϊ
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
        FRNģ�飬����Ϊ[B,C,H,W] , ���Ϊ[B,C,H,W]
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
        # x2 shape [B,C,1,1] (batch_size , channel , 1 , 1),  ��+ Ϊ��Ԫ����ӣ����ı�ά�ȡ�
        x3 = torch.rsqrt(x2)
        # x3 shape [B,C,1,1] (batch_size , channel , 1 , 1),  ��sqrt Ϊ��Ԫ�ز��������ı�ά�ȡ�
        x4 = x * x3
        # x4 shape [B,C,H,W] (batch_size , channel , height , width)
        # �㲥����,��Ԫ�����  [B,C,H,W] * [B,C,1,1] = [B,C,H,W]
        output = torch.max(self.gamma * x4 + self.beta, self.tau)

        ## output shape [B,C,H,W] (batch_size , channel , height , width)  torch.max ��Ԫ�ز���
        return output


class AdaIN2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=True):
        """
        Args:
            num_features ([type]): [����ͨ����ĿC]
            eps ([type], optional): [description]. Defaults to 1e-5.
            momentum (float, optional): [description]. Defaults to 0.1.
            affine (bool, optional): [�Ƿ���ӳ�����]. Defaults to False.
            track_running_stats (bool, optional): [description]. Defaults to True.
        """
        super(AdaIN2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))  # �����ѵ���Ĳ������õľͲ���AdaIN��
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.weight = None  # �˴���ʵ������ʱ��ֵΪNone��֮���������ĺ�����generator.py���assign_adain_params������������и�ֵ������ֵ���ǿ�ѧϰ�ò��������Ǵ�style�Ǳ߼���õ���
            self.bias = None  # �˴���ʵ������ʱ��ֵΪNone��֮���������ĺ�����generator.py���assign_adain_params������������и�ֵ������ֵ���ǿ�ѧϰ�ò��������Ǵ�style�Ǳ߼���õ���

        if self.track_running_stats:
            # Ϊrunning_mean��running_varע��һ��Buffer��Buffer��Ϊ��Module��һ���֣����ǲ����ǿ�ѵ�����������ڸ���ѵ��״̬
            # ͬʱ��Ĭ��ע��ĸ�Buffer����洢ģ�͵�ʱ�򣬻ᱻд��state_dict��
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

        # assert�����ԣ������ж�һ�����ʽ���ڱ��ʽ����Ϊ False ��ʱ�򴥷��쳣��
        assert self.weight is not None and self.bias is not None, "AdaIN params are None"

        N, C, H, W = x.size()

        running_mean = self.running_mean.repeat(N)
        running_var = self.running_var.repeat(N)
        # input x  shape [N,C,H,W] (batch_size , channel , height , width)

        x_ = x.contiguous().view(1, N * C, H * W)
        # x_       shape [1,N*C,H*W] (1,batch_size*channel,height*width)
        # ��Ϊ�˴�x_ �Ѿ���3D�������ˣ������·���batch_norm�������ض���BatchNorm1D��������Ӧ��Ϊ��N'��C'��L'��
        # �˴�BatchNorm1D N' = 1, C' = N * C, L' = H * W, Ȼ������C'��ÿһ��,���㣨N',L'����ͳ������
        # ��ʵ��Ӧ��������������һ�� ��ÿһ��ʵ����ÿһ��Channel������ (1,H*W) ��ͳ������,��ʵ���������һ�� Instance Norm

        normed = F.batch_norm(x_, running_mean, running_var,
                              self.weight, self.bias,
                              True, self.momentum, self.eps)

        # normed   shape [1,N*C,H*W] (1,batch_size*channel,height*width)
        return normed.view(N, C, H, W)

    def __repr__(self):
        """
          ��print()��ʱ������������ʾ��Ϣ��
        """
        return self.__class__.__name__ + '(num_features=' + str(self.num_features) + ')'


if __name__ == '__main__':
    print("CALL blocks.py")

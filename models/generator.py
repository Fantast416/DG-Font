# coding=gbk
from copyreg import constructor
from torch import nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
import scipy.io as io
import math
import numpy as np

try:
    from models.blocks import LinearBlock, Conv2dBlock, ResBlocks
except:
    from blocks import LinearBlock, Conv2dBlock, ResBlocks

import sys

sys.path.append('..')
from modules import modulated_deform_conv


class Generator(nn.Module):
    def __init__(self, img_size=80, sty_dim=64, n_res=2, use_sn=False):
        """

        :param img_size: 输入的图像大小，但是在此份代码中毫无意义，没有被使用
        :param sty_dim: 字体风格向量的大小——Zs的维度
        :param n_res: 默认为2， ResBlock的数量
        :param use_sn: 是否使用sn
        """
        super(Generator, self).__init__()
        print("Init Generator")

        self.nf = 64
        self.nf_mlp = 256

        self.decoder_norm = 'adain'  # 在此处设置decoder_norm的类型，后续传入Decoder类中

        self.adaptive_param_getter = get_num_adain_params  # 设置计算函数
        self.adaptive_param_assign = assign_adain_params  # 设置赋值函数

        print("GENERATOR NF : ", self.nf)

        nf_dec = 256  # nf_dec decoder中输入的特征图的数量

        # ContentEncoder的目的是得到Skip1，Skip2和Zc
        self.cnt_encoder = ContentEncoder(n_res, 'in', 'relu', 'reflect')
        # Decoder是论文中的Mixer
        self.decoder = Decoder(nf_dec, n_res, self.decoder_norm, self.decoder_norm, 'relu', 'reflect',
                               use_sn=use_sn)
        # MLP用于在decode函数中，对参考的风格特征进行计算，从而获得ADAIN的参数
        self.mlp = MLP(sty_dim, self.adaptive_param_getter(self.decoder), self.nf_mlp, 3, 'none', 'relu')

        self.apply(weights_init('kaiming'))  # apply 用于当函数参数已经存在于一个元组或字典中时，间接地调用函数

    def forward(self, x_src, s_ref):
        # Generator中的前向传播，输入是内容图像，以及参考的风格图像
        # 先是内容图像经过Content Encoder，得到 Zc,skip1,skip2
        c_src, skip1, skip2 = self.cnt_encoder(x_src)
        # 然后将内容Zc,skip1,skip2 以及 参考的风格图像 输入解码函数中, 所以原论文中的Style Encoder其实在代码实现的时候，合进了self.decode函数中书写
        x_out = self.decode(c_src, s_ref, skip1, skip2)
        # 最终输出的就是 [B,3,H,W] 生成图像
        return x_out

    def decode(self, cnt, sty, skip1, skip2):
        """

        :param cnt: 内容特征 特征图 Zc [B,256,H/4,W/4]
        :param sty: 风格特征 向量Zs [B,sty_dim] sty_dim就是main.py中输入的可控参数，就是论文中 Zs向量的维度
        :param skip1: skip1 [B,256,H,W]
        :param skip2: skip2 [B,256,H/2,W/2]
        :return: 得到输出图像
        """

        # 在该函数中，首先对参考的风格特征（这个特征在外层，是由GudingNet部分对StyleImage处理得到的）经过ＭＬＰ层，获得参数
        adapt_params = self.mlp(sty)
        # print(f"adapt_params.shape:{adapt_params.shape}") # [32,2342] 个参数
        self.adaptive_param_assign(adapt_params, self.decoder)  # 将DECODER中的ADAIN2D模块的参数进行赋值

        # 得到输出，将内容图像、skip1、skip2通过Decoder的Forward，得到输出
        out, offset_loss = self.decoder(cnt, skip1, skip2) # out : [B,3,H,W]
        # print(f"out[0].shape:{out[0].shape}")

        return out, offset_loss

    def _initialize_weights(self, mode='fan_in'):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()


class Decoder(nn.Module):  # 论文中的 Mixer
    def __init__(self, nf_dec,  n_res, res_norm, dec_norm, act, pad, use_sn=False):
        """
        Args:
            nf_dec ([type]): [输入的特征图数量],外层默认为256
            n_res ([type]): [残差块的数目]
            res_norm ([type]): [在ResBlock中用的Norm函数]
            dec_norm ([type]): [在Decoder的Conv2dBlock中用的Norm函数]
            act ([type]): [激活函数]
            pad ([type]): [padding类型]
            use_sn (bool, optional): [description]. Defaults to False.
        """
        super(Decoder, self).__init__()
        print("Init Decoder")

        nf = nf_dec
        self.model = nn.ModuleList()
        self.model.append(ResBlocks(n_res, nf, res_norm, act, pad, use_sn=use_sn))

        self.model.append(nn.Upsample(scale_factor=2))
        self.model.append(Conv2dBlock(nf, nf // 2, 5, 1, 2, norm=dec_norm, act=act, pad_type=pad, use_sn=use_sn))

        self.model.append(nn.Upsample(scale_factor=2))
        self.model.append(Conv2dBlock(nf, nf // 4, 5, 1, 2, norm=dec_norm, act=act, pad_type=pad, use_sn=use_sn))

        self.model.append(Conv2dBlock(nf // 2, 3, 7, 1, 3, norm='none', act='tanh', pad_type=pad, use_sn=use_sn))
        self.model = nn.Sequential(*self.model)
        self.dcn = modulated_deform_conv.ModulatedDeformConvPack(64, 64, kernel_size=(3, 3), stride=1, padding=1,
                                                                 groups=1, deformable_groups=1, double=True).cuda()
        self.dcn_2 = modulated_deform_conv.ModulatedDeformConvPack(128, 128, kernel_size=(3, 3), stride=1, padding=1,
                                                                   groups=1, deformable_groups=1, double=True).cuda()

    def forward(self, x, skip1, skip2):
        """
        :param x: 内容特征 特征图 Zc [B,256,H/4,W/4]
        :param skip1: skip1 [B,256,H,W]
        :param skip2: skip2 [B,256,H/2,W/2]
        :return:
        """
        output = x  # [B,256,H/4,W/4]
        for i in range(len(self.model)):
            # 依次完成：
            # i=0 ResBlocks(n_res, nf, res_norm, act, pad, use_sn=use_sn)                                # [B,256,H/4,W/4]
            # i=1 nn.Upsample(scale_factor=2)                                                            # [B,256,H/2,W/2]
            # i=2 Conv2dBlock(nf, nf // 2, 5, 1, 2, norm=dec_norm, act=act, pad_type=pad, use_sn=use_sn) # [B,128,H/2,W/2]
            #     附加操作：concat + dcn                                                                  # [B,256,H/2,W/2]
            # i=3 nn.Upsample(scale_factor=2)                                                            # [B,256,H,W]
            # i=4 Conv2dBlock(nf, nf // 4, 5, 1, 2, norm=dec_norm, act=act, pad_type=pad, use_sn=use_sn) # [B,64,H,W]
            #     附加操作：concat + dcn                                                                  # [B,128,H,W]
            # i=5 Conv2dBlock(nf // 2, 3, 7, 1, 3, norm='none', act='tanh', pad_type=pad, use_sn=use_sn) # [B,3,H,W]
            output = self.model[i](output)
            print("i",i,"output.size()",output.size())
            if i == 2:
                deformable_concat = torch.cat((output, skip2), dim=1)  # 将卷积层输出的output，和skip2的内容 在channel维度concat起来
                # print(f"deformable_concat.shape{deformable_concat.shape}")
                concat_pre, offset2 = self.dcn_2(deformable_concat, skip2)
                # print(f"concat_pre.shape{concat_pre.shape}")
                # print(f"offset2.shape{offset2.shape}")
                output = torch.cat((concat_pre, output), dim=1)
                print("i=2", "output.size()", output.size())

            if i == 4:
                deformable_concat = torch.cat((output, skip1), dim=1)
                concat_pre, offset1 = self.dcn(deformable_concat, skip1)
                output = torch.cat((concat_pre, output), dim=1)
                print("i=4", "output.size()", output.size())

        # 用于在train.py中 144行，计算offset_loss用
        offset_sum1 = torch.mean(torch.abs(offset1))
        offset_sum2 = torch.mean(torch.abs(offset2))
        offset_sum = (offset_sum1 + offset_sum2) / 2
        return output, offset_sum


class ContentEncoder(nn.Module):
    """
        变形卷积 + IN + ReLU  得到Skip1
        变形卷积 + IN + ReLU  得到Skip2
        变形卷积 + IN + ReLU + N个ResBlock叠加  得到输出Zc
    """

    def __init__(self, n_res, norm, act, pad, use_sn=False):
        """
        Args:
            n_res ([type]): [ResBlock的数量]
            norm ([type]): [norm使用的方法]
            act ([type]): [使用的激活函数]
            pad ([type]): [使用的padding类型]
            use_sn (bool, optional): [description]. Defaults to False.
        """
        super(ContentEncoder, self).__init__()
        print("Init ContentEncoder")

        # nf = nf_cnt
        self.model = nn.ModuleList()
        self.model.append(ResBlocks(n_res, 256, norm=norm, act=act, pad_type=pad, use_sn=use_sn))
        self.model = nn.Sequential(*self.model)
        self.dcn1 = modulated_deform_conv.ModulatedDeformConvPack(3, 64, kernel_size=(7, 7), stride=1, padding=3,
                                                                  groups=1, deformable_groups=1).cuda()
        self.dcn2 = modulated_deform_conv.ModulatedDeformConvPack(64, 128, kernel_size=(4, 4), stride=2, padding=1,
                                                                  groups=1, deformable_groups=1).cuda()
        self.dcn3 = modulated_deform_conv.ModulatedDeformConvPack(128, 256, kernel_size=(4, 4), stride=2, padding=1,
                                                                  groups=1, deformable_groups=1).cuda()
        self.IN1 = nn.InstanceNorm2d(64)
        self.IN2 = nn.InstanceNorm2d(128)
        self.IN3 = nn.InstanceNorm2d(256)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        print("Content Encoder,input size", x.size())  # 输入为 [B,C,H,W]
        # 以下每一行注释标注的大小，都是经过改行代码后输出的Tensor大小
        x, _ = self.dcn1(x, x)  # [B,64,H,W] DCN1的kernel_size和padding保证了其不会改变特征图的大小，只变化了特征图的通道数
        x = self.IN1(x)  # [B,64,H,W] InstanceNorm不会改变特征图大小和通道数
        x = self.activation(x)  # [B,64,H,W]
        skip1 = x  # [B,64,H,W]

        x, _ = self.dcn2(x, x)  # [B,128,H/2,W/2]
        x = self.IN2(x)  # [B,128,H/2,W/2]
        x = self.activation(x)  # [B,128,H/2,W/2]
        skip2 = x  # [B,128,H/2,W/2]

        x, _ = self.dcn3(x, x)  # [B,256,H/4,W/4]
        x = self.IN3(x)  # [B,256,H/4,W/4]
        x = self.activation(x)  # [B,256,H/4,W/4]
        x = self.model(x)  # 经过两组ResBlock + IN ，最终输出的Tensor为 [B,256,H/4,W/4]

        # 假设输入的图像大小为 80 * 80
        # print(f"skip1.shape:{skip1.shape}") # [32,64,80,80]
        # print(f"skip2.shape:{skip2.shape}") # [32,128,40,40]
        # print(f"x.shape:{x.shape}")         # [32,256,20,20]
        return x, skip1, skip2


class MLP(nn.Module):
    def __init__(self, nf_in, nf_out, nf_mlp, num_blocks, norm, act, use_sn=False):
        super(MLP, self).__init__()
        self.model = nn.ModuleList()
        nf = nf_mlp
        self.model.append(LinearBlock(nf_in, nf, norm=norm, act=act, use_sn=use_sn))
        for _ in range(num_blocks - 2):
            self.model.append(LinearBlock(nf, nf, norm=norm, act=act, use_sn=use_sn))
        self.model.append(LinearBlock(nf, nf_out, norm='none', act='none', use_sn=use_sn))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaIN2d":
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2 * m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2 * m.num_features:
                adain_params = adain_params[:, 2 * m.num_features:]


def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaIN2d":
            num_adain_params += 2 * m.num_features
    return num_adain_params

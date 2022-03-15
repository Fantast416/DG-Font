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

        :param img_size: �����ͼ���С�������ڴ˷ݴ����к������壬û�б�ʹ��
        :param sty_dim: �����������Ĵ�С����Zs��ά��
        :param n_res: Ĭ��Ϊ2�� ResBlock������
        :param use_sn: �Ƿ�ʹ��sn
        """
        super(Generator, self).__init__()
        print("Init Generator")

        self.nf = 64
        self.nf_mlp = 256

        self.decoder_norm = 'adain'  # �ڴ˴�����decoder_norm�����ͣ���������Decoder����

        self.adaptive_param_getter = get_num_adain_params  # ���ü��㺯��
        self.adaptive_param_assign = assign_adain_params  # ���ø�ֵ����

        print("GENERATOR NF : ", self.nf)

        nf_dec = 256  # nf_dec decoder�����������ͼ������

        # ContentEncoder��Ŀ���ǵõ�Skip1��Skip2��Zc
        self.cnt_encoder = ContentEncoder(n_res, 'in', 'relu', 'reflect')
        # Decoder�������е�Mixer
        self.decoder = Decoder(nf_dec, n_res, self.decoder_norm, self.decoder_norm, 'relu', 'reflect',
                               use_sn=use_sn)
        # MLP������decode�����У��Բο��ķ���������м��㣬�Ӷ����ADAIN�Ĳ���
        self.mlp = MLP(sty_dim, self.adaptive_param_getter(self.decoder), self.nf_mlp, 3, 'none', 'relu')

        self.apply(weights_init('kaiming'))  # apply ���ڵ����������Ѿ�������һ��Ԫ����ֵ���ʱ����ӵص��ú���

    def forward(self, x_src, s_ref):
        # Generator�е�ǰ�򴫲�������������ͼ���Լ��ο��ķ��ͼ��
        # ��������ͼ�񾭹�Content Encoder���õ� Zc,skip1,skip2
        c_src, skip1, skip2 = self.cnt_encoder(x_src)
        # Ȼ������Zc,skip1,skip2 �Լ� �ο��ķ��ͼ�� ������뺯����, ����ԭ�����е�Style Encoder��ʵ�ڴ���ʵ�ֵ�ʱ�򣬺Ͻ���self.decode��������д
        x_out = self.decode(c_src, s_ref, skip1, skip2)
        # ��������ľ��� [B,3,H,W] ����ͼ��
        return x_out

    def decode(self, cnt, sty, skip1, skip2):
        """

        :param cnt: �������� ����ͼ Zc [B,256,H/4,W/4]
        :param sty: ������� ����Zs [B,sty_dim] sty_dim����main.py������Ŀɿز��������������� Zs������ά��
        :param skip1: skip1 [B,256,H,W]
        :param skip2: skip2 [B,256,H/2,W/2]
        :return: �õ����ͼ��
        """

        # �ڸú����У����ȶԲο��ķ�������������������㣬����GudingNet���ֶ�StyleImage����õ��ģ������̣ͣв㣬��ò���
        adapt_params = self.mlp(sty)
        # print(f"adapt_params.shape:{adapt_params.shape}") # [32,2342] ������
        self.adaptive_param_assign(adapt_params, self.decoder)  # ��DECODER�е�ADAIN2Dģ��Ĳ������и�ֵ

        # �õ������������ͼ��skip1��skip2ͨ��Decoder��Forward���õ����
        out, offset_loss = self.decoder(cnt, skip1, skip2) # out : [B,3,H,W]
        # print(f"out[0].shape:{out[0].shape}")

        return out, offset_loss

    def _initialize_weights(self, mode='fan_in'):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()


class Decoder(nn.Module):  # �����е� Mixer
    def __init__(self, nf_dec,  n_res, res_norm, dec_norm, act, pad, use_sn=False):
        """
        Args:
            nf_dec ([type]): [���������ͼ����],���Ĭ��Ϊ256
            n_res ([type]): [�в�����Ŀ]
            res_norm ([type]): [��ResBlock���õ�Norm����]
            dec_norm ([type]): [��Decoder��Conv2dBlock���õ�Norm����]
            act ([type]): [�����]
            pad ([type]): [padding����]
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
        :param x: �������� ����ͼ Zc [B,256,H/4,W/4]
        :param skip1: skip1 [B,256,H,W]
        :param skip2: skip2 [B,256,H/2,W/2]
        :return:
        """
        output = x  # [B,256,H/4,W/4]
        for i in range(len(self.model)):
            # ������ɣ�
            # i=0 ResBlocks(n_res, nf, res_norm, act, pad, use_sn=use_sn)                                # [B,256,H/4,W/4]
            # i=1 nn.Upsample(scale_factor=2)                                                            # [B,256,H/2,W/2]
            # i=2 Conv2dBlock(nf, nf // 2, 5, 1, 2, norm=dec_norm, act=act, pad_type=pad, use_sn=use_sn) # [B,128,H/2,W/2]
            #     ���Ӳ�����concat + dcn                                                                  # [B,256,H/2,W/2]
            # i=3 nn.Upsample(scale_factor=2)                                                            # [B,256,H,W]
            # i=4 Conv2dBlock(nf, nf // 4, 5, 1, 2, norm=dec_norm, act=act, pad_type=pad, use_sn=use_sn) # [B,64,H,W]
            #     ���Ӳ�����concat + dcn                                                                  # [B,128,H,W]
            # i=5 Conv2dBlock(nf // 2, 3, 7, 1, 3, norm='none', act='tanh', pad_type=pad, use_sn=use_sn) # [B,3,H,W]
            output = self.model[i](output)
            print("i",i,"output.size()",output.size())
            if i == 2:
                deformable_concat = torch.cat((output, skip2), dim=1)  # ������������output����skip2������ ��channelά��concat����
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

        # ������train.py�� 144�У�����offset_loss��
        offset_sum1 = torch.mean(torch.abs(offset1))
        offset_sum2 = torch.mean(torch.abs(offset2))
        offset_sum = (offset_sum1 + offset_sum2) / 2
        return output, offset_sum


class ContentEncoder(nn.Module):
    """
        ���ξ�� + IN + ReLU  �õ�Skip1
        ���ξ�� + IN + ReLU  �õ�Skip2
        ���ξ�� + IN + ReLU + N��ResBlock����  �õ����Zc
    """

    def __init__(self, n_res, norm, act, pad, use_sn=False):
        """
        Args:
            n_res ([type]): [ResBlock������]
            norm ([type]): [normʹ�õķ���]
            act ([type]): [ʹ�õļ����]
            pad ([type]): [ʹ�õ�padding����]
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
        print("Content Encoder,input size", x.size())  # ����Ϊ [B,C,H,W]
        # ����ÿһ��ע�ͱ�ע�Ĵ�С�����Ǿ������д���������Tensor��С
        x, _ = self.dcn1(x, x)  # [B,64,H,W] DCN1��kernel_size��padding��֤���䲻��ı�����ͼ�Ĵ�С��ֻ�仯������ͼ��ͨ����
        x = self.IN1(x)  # [B,64,H,W] InstanceNorm����ı�����ͼ��С��ͨ����
        x = self.activation(x)  # [B,64,H,W]
        skip1 = x  # [B,64,H,W]

        x, _ = self.dcn2(x, x)  # [B,128,H/2,W/2]
        x = self.IN2(x)  # [B,128,H/2,W/2]
        x = self.activation(x)  # [B,128,H/2,W/2]
        skip2 = x  # [B,128,H/2,W/2]

        x, _ = self.dcn3(x, x)  # [B,256,H/4,W/4]
        x = self.IN3(x)  # [B,256,H/4,W/4]
        x = self.activation(x)  # [B,256,H/4,W/4]
        x = self.model(x)  # ��������ResBlock + IN �����������TensorΪ [B,256,H/4,W/4]

        # ���������ͼ���СΪ 80 * 80
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

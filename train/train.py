from tqdm import trange
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tools.utils import *
from tools.ops import compute_grad_gp, update_average, copy_norm_params, queue_data, dequeue_data, \
    average_gradients, calc_adv_loss, calc_contrastive_loss, calc_recon_loss
from torchvision.utils import save_image

def trainGAN(data_loader, networks, opts, epoch, args, additional):
    """
    训练核心函数
    :param data_loader: 训练集的数据加载器 DataLoader类
    :param networks: 网络 字典，包含  C、C_EMA、D、G，字典的key 都是 nn.Module
    :param opts: 使用的优化器字典，同上
    :param epoch: 当前处于的轮次
    :param args: 外层传入的参数
    :param additional: 额外参数，外层传入的是logger
    :return:
    """
    # AverageMeter类用于计算 某个变量指标 在一个epoch中的平均值
    # reset成员函数用于重置内部变量
    # update成员函数，用于更新变量的存储记录值
    d_losses = AverageMeter()
    d_advs = AverageMeter()
    d_gps = AverageMeter()

    g_losses = AverageMeter()
    g_advs = AverageMeter()
    g_imgrecs = AverageMeter()
    g_cont = AverageMeter()

    # set nets
    D = networks['D']  # Discriminator类
    G = networks['G'] if not args.distributed else networks['G'].module # Generator类
    G_EMA = networks['G_EMA'] if not args.distributed else networks['G_EMA'].module # EMA（Exponential Moving Average）是指数移动平均值，用于训练

    # set opts
    d_opt = opts['D']
    g_opt = opts['G']

    # switch to train mode
    D.train()
    G.train()
    G_EMA.train()

    logger = additional['logger']

    # summary writer
    # dataloader本质上是一个可迭代对象，可以使用iter()进行访问，采用iter(data_loader)返回的是一个迭代器，然后可以使用next()访问。返回的是一个批次的数据

    train_iter = iter(data_loader)

    # args.iters 为参数控制传入的：每个Epoch迭代中，梯度下降参数更新的次数
    t_train = trange(0, args.iters, initial=0, total=args.iters)

    # trange 同python中的range,区别在于trange在循环执行的时候会输出打印进度条,最后的 5.00s/it 代表每次循环耗时5s
    """
     0%|                                                                                            | 0/3 [00:00<?, ?it/s]
    第1次执行
     33%|████████████████████████████                                                        | 1/3 [00:05<00:10,  5.00s/it]
    第2次执行
     67%|████████████████████████████████████████████████████████                            | 2/3 [00:10<00:05,  5.00s/it]
    第3次执行
    100%|████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:15<00:00,  5.01s/it]
    ————————————————
    """
    for i in t_train: # 最外层循环
        try:
            imgs, y_org = next(train_iter)  # 获取一个Batch的数据
        except:
            train_iter = iter(data_loader)
            imgs, y_org = next(train_iter)

        x_org = imgs

        # x_org： 原始样本 作为Content Image  y_org： x_org 对应的类别标签
        # x_org.size() : [B,C,W,H]  BatchSize,Channel,Width,Height
        # x_org.size(0) = BatchSize

        # 返回一个 0-BatchSize 打乱顺序的数组 x_ref_idx
        x_ref_idx = torch.randperm(x_org.size(0))

        # 将数据拷贝至GPU中
        x_org = x_org.cuda(args.gpu)
        y_org = y_org.cuda(args.gpu)
        x_ref_idx = x_ref_idx.cuda(args.gpu)

        # x_ref 打乱顺序后的样本，作为参考图像即 Style Image
        x_ref = x_org.clone()
        x_ref = x_ref[x_ref_idx]

        # x_org : Content_Images
        # x_ref : Style_Images
        # print("content_images", x_org.size())  # [B,C,H,W] [8,3,80,80]
        # print("style_images", x_ref.size())

        training_mode = 'GAN'

        ####################
        # BEGIN Train GANs #
        ####################

        # Train D
        with torch.no_grad():  # 以下操作不需要计算梯度，原因是在GAN的训练过程中，我们先固定G，训练D的参数，此步是在做生成，也就是G的操作部分
            # y_ref:  x_ref 对应的类别标签
            y_ref = y_org.clone()
            y_ref = y_ref[x_ref_idx]

            x_fake, _, _, _, _ = G(x_org, x_ref)
            # 注意：x_fake图像的内容应当是x_org(Content Image)的，而风格应当是x_ref(Style Image)的

        x_ref.requires_grad_() # 接下来进入D的参数训练，所以x_ref(即StyleImage)需要在计算中保留对应的梯度信息

        d_real_logit, _ = D(x_ref, y_ref) # 将 StyleImage 和 对应的真实类别标签 输入判别器，得到一个logit分数 d_real_logit

        d_fake_logit, _ = D(x_fake.detach(), y_ref) # 将生成的输出图像 和 x_ref(Style Image)对应的真实类别标签 输入判别器，得到一个logit分数 d_fake_logit

        # .detach() 会返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,
        # 不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad。

        # d_real_logit [B] 一维向量, 每个值就是 判别器给 输入样本 针对输入的y_ref类别打的分
        # d_fake_logit [B] 一维向量, 每个值就是 判别器给 输入样本 针对输入的y_ref类别打的分

        # 计算损失d_loss，由两部分组成： d_adv(生成对抗网络损失) 和 d_gp()
        d_adv_real = calc_adv_loss(d_real_logit, 'd_real')
        d_adv_fake = calc_adv_loss(d_fake_logit, 'd_fake')

        d_adv = d_adv_real + d_adv_fake

        d_gp = args.w_gp * compute_grad_gp(d_real_logit, x_ref, is_patch=False) # 这部分损失函数是 Gradient penalty （WGAN提出的训练GAN的改进方法）

        d_loss = d_adv + d_gp

        d_opt.zero_grad()  # 优化器清零梯度

        d_adv_real.backward(retain_graph=True) # 损失反向传播，计算梯度
        d_gp.backward()  # 损失反向传播，计算梯度
        d_adv_fake.backward()  # 损失反向传播，计算梯度
        if args.distributed:
            average_gradients(D)

        d_opt.step()  # 使用optimizer更新一步判别器的参数

        # Train G:
        # 固定D的参数，训练G
        # 生成图像 x_fake
        x_fake, Icc, Iss, x_org_Zc, x_fake_Zc = G(x_org, x_ref)

        # 计算Content Consistency Loss
        loss_c = calc_recon_loss(x_fake_Zc, x_org_Zc)

        # 计算对抗生成损失

        # 将 x_ref风格，x_org内容的 生成图像 x_fake ， 以及 y_ref风格类别交给判别器，得到g_fake_logit
        # 我们希望这个值越大越好,代表其骗过了生成器，泛化能力强了
        g_fake_logit, _ = D(x_fake, y_ref)
        # 将 x_org风格，x_org内容的 生成图像 x_rec， 以及 y_org风格类别交给判别器，g_rec_logit
        # 我们希望这个值越大越好，代表生成器重建原图像能力强了
        x_rec, _, _, _, _ = G(x_org, x_org)
        g_rec_logit, _ = D(x_rec, y_org)

        g_adv_fake = calc_adv_loss(g_fake_logit, 'g')
        g_adv_rec = calc_adv_loss(g_rec_logit, 'g')

        # 对抗生成损失计算，两者的和
        g_adv = g_adv_fake + g_adv_rec

        # 计算图像重建损失(StyTR-2版)
        # loss_rec = G.module.calc_content_loss(Icc, x_org) + G.module.calc_content_loss(Iss, x_ref)

        # 计算图像重建损失，输入为 重建图像 x_rec 和 原图像 x_org，我们希望两者差距越小越好（DG-Font版）
        loss_rec = calc_recon_loss(x_rec, x_org)

        # 总损失函数，与论文一致，有四项组成：对抗损失，图像重建损失，内容一致性损失
        g_loss = args.w_adv * g_adv + args.w_rec * loss_rec + args.w_cont * loss_c


        if i % 100 == 0:
            if not os.path.exists(args.res_dir + "/test"):
                os.makedirs(args.res_dir + "/test")

            output_name = '{:s}/test/{:s}{:s}'.format(
                args.res_dir, str(i), ".jpg"
            )
            x_fake = torch.cat((x_org, x_fake), 0)
            x_fake = torch.cat((x_ref, x_fake), 0)
            # print("x_fake.size()", x_fake.size())
            save_image(x_fake, output_name, nrow=args.batch_size)

        g_opt.zero_grad()
        g_loss.backward()  # 损失反向传播，计算梯度

        if args.distributed:
            average_gradients(G)

        g_opt.step()  # 更新 G 的参数

        ##################
        # END Train GANs #
        ##################


        if epoch >= args.ema_start:
            training_mode = training_mode + "_EMA"
            update_average(G_EMA, G)

        torch.cuda.synchronize()

        with torch.no_grad():
            if epoch >= args.separated:
                d_losses.update(d_loss.item(), x_org.size(0))
                d_advs.update(d_adv.item(), x_org.size(0))
                d_gps.update(d_gp.item(), x_org.size(0))

                g_losses.update(g_loss.item(), x_org.size(0))
                g_advs.update(g_adv.item(), x_org.size(0))
                g_imgrecs.update(loss_rec.item(), x_org.size(0))
                g_cont.update(loss_c.item(), x_org.size(0))


            if (i + 1) % args.log_step == 0 and (args.gpu == 0 or args.gpu == '0'):
                summary_step = epoch * args.iters + i
                add_logs(args, logger, 'D/LOSS', d_losses.avg, summary_step)
                add_logs(args, logger, 'D/ADV', d_advs.avg, summary_step)
                add_logs(args, logger, 'D/GP', d_gps.avg, summary_step)

                add_logs(args, logger, 'G/LOSS', g_losses.avg, summary_step)
                add_logs(args, logger, 'G/ADV', g_advs.avg, summary_step)
                add_logs(args, logger, 'G/IMGREC', g_imgrecs.avg, summary_step)
                add_logs(args, logger, 'G/CONT', g_cont.avg, summary_step)

                print('Epoch: [{}/{}] [{}/{}] MODE[{}] Avg Loss: D[{d_losses.avg:.2f}] G[{g_losses.avg:.2f}] '.format(epoch + 1, args.epochs, i+1, args.iters,
                                                        training_mode, d_losses=d_losses, g_losses=g_losses))

    copy_norm_params(G_EMA, G)


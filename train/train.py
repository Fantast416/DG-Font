from tqdm import trange
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tools.utils import *
from tools.ops import compute_grad_gp, update_average, copy_norm_params, queue_data, dequeue_data, \
    average_gradients, calc_adv_loss, calc_contrastive_loss, calc_recon_loss


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
    g_rec = AverageMeter()

    moco_losses = AverageMeter()

    # set nets
    D = networks['D']  # Discriminator类
    G = networks['G'] if not args.distributed else networks['G'].module # Generator类
    C = networks['C'] if not args.distributed else networks['C'].module # GuidingNet类
    G_EMA = networks['G_EMA'] if not args.distributed else networks['G_EMA'].module
    C_EMA = networks['C_EMA'] if not args.distributed else networks['C_EMA'].module

    # set opts
    d_opt = opts['D']
    g_opt = opts['G']
    c_opt = opts['C']

    # switch to train mode
    D.train()
    G.train()
    C.train()
    C_EMA.train()
    G_EMA.train()

    logger = additional['logger']

    # summary writer
    # dataloader本质上是一个可迭代对象，可以使用iter()进行访问，采用iter(data_loader)返回的是一个迭代器，然后可以使用next()访问。返回的是一个批次的数据
    train_it = iter(data_loader)

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
            imgs, y_org = next(train_it)  # 获取一个Batch的数据
        except:
            train_it = iter(data_loader)
            imgs, y_org = next(train_it)

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

        training_mode = 'GAN'

        ####################
        # BEGIN Train GANs #
        ####################

        # Train D
        with torch.no_grad():  # 以下操作不需要计算梯度，原因是在GAN的训练过程中，我们先固定G，训练D的参数，此步是在做生成，也就是G的操作部分
            # y_ref:  x_ref 对应的类别标签
            y_ref = y_org.clone()
            y_ref = y_ref[x_ref_idx]

            s_ref = C.moco(x_ref)  # 对Style Image经过GuidingNet得到 Zs向量
            c_src, skip1, skip2 = G.cnt_encoder(x_org) # 对Content Image经过Content Encoder得到 Zc,skip1,skip2
            x_fake, _ = G.decode(c_src, s_ref, skip1, skip2) # 将Zc,Zs,skip1,skip2输入G的decode函数，得到 生成的输出图像
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

        d_gp = args.w_gp * compute_grad_gp(d_real_logit, x_ref, is_patch=False) # 这部分损失函数是？

        d_loss = d_adv + d_gp

        d_opt.zero_grad()  # 优化器清零梯度

        d_adv_real.backward(retain_graph=True) # 损失反向传播，计算梯度
        d_gp.backward()  # 损失反向传播，计算梯度
        d_adv_fake.backward()  # 损失反向传播，计算梯度
        if args.distributed:
            average_gradients(D)

        d_opt.step()  # 使用optimizer更新一步判别器的参数

        # Train G 固定D的参数，训练G
        s_src = C.moco(x_org) # 将ContentImage输入得到 x_org的风格特征向量 Zs_src
        s_ref = C.moco(x_ref) # 将StyleImage输入得到 x_ref的风格特征向量 Zs_ref (Zs)

        # 将ContentImage输入Content Encoder，得到 Zc,skip1,skip2
        c_src, skip1, skip2 = G.cnt_encoder(x_org)
        # 输入Zc,Zs,skip1,skip2,得到 x_ref风格，x_org内容的 生成图像 x_fake, 以及 offset_loss
        x_fake, offset_loss = G.decode(c_src, s_ref, skip1, skip2)
        # 输入Zc,Zs_src,skip1,skip2,得到 x_org风格，x_org内容的 生成图像 x_rec
        x_rec, _ = G.decode(c_src, s_src, skip1, skip2)

        # 将 x_ref风格，x_org内容的 生成图像 x_fake ， 以及 y_ref风格类别交给判别器，得到g_fake_logit
        # 我们希望这个值越大越好,代表其骗过了生成器，泛化能力抢了
        g_fake_logit, _ = D(x_fake, y_ref)
        # 将 x_org风格，x_org内容的 生成图像 x_rec， 以及 y_org风格类别交给判别器，g_rec_logit
        # 我们希望这个值越大越好，代表生成器重建原图像能力强了
        g_rec_logit, _ = D(x_rec, y_org)

        g_adv_fake = calc_adv_loss(g_fake_logit, 'g')
        g_adv_rec = calc_adv_loss(g_rec_logit, 'g')

        # 对抗生成损失计算，两者的和
        g_adv = g_adv_fake + g_adv_rec

        # 计算图像重建损失，输入为 重建图像 x_rec 和 原图像 x_org，我们希望两者差距越小越好
        g_imgrec = calc_recon_loss(x_rec, x_org)

        # 计算内容一致性损失，即Content Consistency Loss，，我们希望两者差距越小越好
        # 其实就是在计算，x_org和x_fake转换到内容空间中后，向量的一致性。其表现了ContentEncoder能够保留内容结构的能力
        c_x_fake, _, _ = G.cnt_encoder(x_fake) # 将x_fake 输入Content Encoder，得到 c_x_fake 即Zc
        g_conrec = calc_recon_loss(c_x_fake, c_src) # 计算 x_org得到的Zc 和 x_fake得到的Zc 的差别

        # 总损失函数，与论文一致，有四项组成：对抗损失，图像重建损失，内容一致性损失，变形偏差损失
        g_loss = args.w_adv * g_adv + args.w_rec * g_imgrec + args.w_rec * g_conrec + args.w_off * offset_loss
 
        g_opt.zero_grad()
        c_opt.zero_grad()
        g_loss.backward()  # 损失反向传播，计算梯度

        if args.distributed:
            average_gradients(G)
            average_gradients(C)

        c_opt.step()  # 更新 C 的参数
        g_opt.step()  # 更新 G 的参数

        ##################
        # END Train GANs #
        ##################


        if epoch >= args.ema_start:
            training_mode = training_mode + "_EMA"
            update_average(G_EMA, G)
        update_average(C_EMA, C)

        torch.cuda.synchronize()

        with torch.no_grad():
            if epoch >= args.separated:
                d_losses.update(d_loss.item(), x_org.size(0))
                d_advs.update(d_adv.item(), x_org.size(0))
                d_gps.update(d_gp.item(), x_org.size(0))

                g_losses.update(g_loss.item(), x_org.size(0))
                g_advs.update(g_adv.item(), x_org.size(0))
                g_imgrecs.update(g_imgrec.item(), x_org.size(0))
                g_rec.update(g_conrec.item(), x_org.size(0))

                moco_losses.update(offset_loss.item(), x_org.size(0))

            if (i + 1) % args.log_step == 0 and (args.gpu == 0 or args.gpu == '0'):
                summary_step = epoch * args.iters + i
                add_logs(args, logger, 'D/LOSS', d_losses.avg, summary_step)
                add_logs(args, logger, 'D/ADV', d_advs.avg, summary_step)
                add_logs(args, logger, 'D/GP', d_gps.avg, summary_step)

                add_logs(args, logger, 'G/LOSS', g_losses.avg, summary_step)
                add_logs(args, logger, 'G/ADV', g_advs.avg, summary_step)
                add_logs(args, logger, 'G/IMGREC', g_imgrecs.avg, summary_step)
                add_logs(args, logger, 'G/conrec', g_rec.avg, summary_step)

                add_logs(args, logger, 'C/OFFSET', moco_losses.avg, summary_step)

                print('Epoch: [{}/{}] [{}/{}] MODE[{}] Avg Loss: D[{d_losses.avg:.2f}] G[{g_losses.avg:.2f}] '.format(epoch + 1, args.epochs, i+1, args.iters,
                                                        training_mode, d_losses=d_losses, g_losses=g_losses))

    copy_norm_params(G_EMA, G)
    copy_norm_params(C_EMA, C)


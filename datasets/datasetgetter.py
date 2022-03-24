import torch
from torchvision.datasets import ImageFolder
import os
import torchvision.transforms as transforms
from datasets.custom_dataset import ImageFolerRemap


class Compose(object):
    """
    定义一个集成函数操作的类，可以集成许多函数操作，简化代码
    """
    def __init__(self, tf):
        """
        实例化的时候，传入tf：即要执行的函数数组
        :param tf: 要执行的函数数组
        """
        self.tf = tf

    def __call__(self, img):
        """
        类似于重载()函数，让实例对象可以像函数一样调用。
        对输入的img图片对象，按顺序执行之前实例化时传入的操作数组
        :param img:
        :return:
        """
        for t in self.tf:
            img = t(img)
        return img


def get_dataset(args):
    """

    :param args:
    :return: train_dataset,val_dataset
    train_dataset : 如下所示的一个字典：
    {'TRAIN': tr_dataset, 'FULL': dataset}，
    val_dataset : val_dataset
    其中： tr_dataset,dataset,val_dataset 都是 pytorch 的 数据集Dataset对象
    """
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    normalize = transforms.Normalize(mean=mean, std=std)

    transform = Compose([transforms.Resize((args.img_size, args.img_size)),
                         transforms.ToTensor(),
                         normalize])

    transform_val = Compose([transforms.Resize((args.img_size, args.img_size)),
                             transforms.ToTensor(),
                             normalize])
    
    class_to_use = args.att_to_use

    print('USE CLASSES', class_to_use)

    # remap labels, 重映射表，在Dataset的getitem函数中，会对 原来的标签ID 进行重映射
    # 原来的标签ID 为： 从0开始的连续数字。即如果总共有9个类别，那么原来的标签ID即为 0-8
    remap_table = {}
    i = 0
    for k in class_to_use:
        remap_table[k] = i
        i += 1
    # 此处生成的remap映射表，相当于是无效映射表，只是做了一个等值映射
    print("LABEL MAP:", remap_table)

    img_dir = args.data_dir # 图片数据目录

    # 构造数据集
    dataset = ImageFolerRemap(img_dir, transform=transform, remap_table=remap_table)
    # 构造验证用数据集：该验证用数据集到目前为止和上一行构造的数据集无任何差别，只是之后作用不同
    dataset_val = ImageFolerRemap(img_dir, transform=transform_val, remap_table=remap_table)

    print("dataset.targets:",dataset.targets)

    # parse classes to use 将标签转换为Tensor类型的对象
    tot_targets = torch.tensor(dataset.targets)

    min_data = 99999999
    max_data = 0

    train_idx = None
    val_idx = None
    # class_to_use: range(0, 9)
    for k in class_to_use:  # 遍历所有图像的类：
        # 假设 tot_targets: [0,0,0,0,1,1,1,1,2,2,2,2] k = 0
        # nonzero函数是numpy中用于得到数组array中非零元素的位置（数组索引）的函数。
        # (tot_targets == k) : [1,1,1,1,0,0,0,0,0,0,0,0]
        # (tot_targets == k).nonzero() : [[0],[1],[2],[3]] 结果是一个二维数组，每一个元素是一个数组，当操作数数组是一维数组时，每一个元素的数组只有一个元素，描述非零元素的索引
        tmp_idx = (tot_targets == k).nonzero() # tmp_idx 为所有 属于类别k的图片的索引数组（该索引 指对于dataset.sample数组中的位置）
        # 将属于该class类别的图片在数组中的索引划分为 训练和验证两个模块
        train_tmp_idx = tmp_idx[:-args.val_num]
        val_tmp_idx = tmp_idx[-args.val_num:]
        # 如果是第一轮循环进来的时候，直接赋值，否则cat连接向量，形成最终的train_idx和val_idx数组
        if k == class_to_use[0]:
            train_idx = train_tmp_idx.clone()
            val_idx = val_tmp_idx.clone()
        else:
            train_idx = torch.cat((train_idx, train_tmp_idx))
            val_idx = torch.cat((val_idx, val_tmp_idx))

        # 统计一下每种类别的图片的个数，计算出最多图片数 和 最少图片数目
        if min_data > len(train_tmp_idx):
            min_data = len(train_tmp_idx)
        if max_data < len(train_tmp_idx):
            max_data = len(train_tmp_idx)

    # 使用util中的函数Subset依据刚才划分好的Idx，对数据集本身进行划分
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset_val, val_idx)

    # 输出 所有种类图片中，单种类最多数量图片数，和，单种类最少数量图片数
    args.min_data = min_data
    args.max_data = max_data
    print("MINIMUM DATA :", args.min_data)
    print("MAXIMUM DATA :", args.max_data)

    print("train_dataset", train_dataset)

    # 训练集本身 变为一个字典返回
    train_dataset = {'TRAIN': train_dataset, 'FULL': dataset}

    # val_dataset 是一个pytorch的SubDataset对象,可以视为是一个Dataset

    print("val_dataset",val_dataset)

    return train_dataset, val_dataset


if __name__ == '__main__':
    a = torch.tensor([[0,1,0,0],[1,0,1,1]])
    b = a.nonzero()
    print(b)
    """
    b: tensor([[0, 1],
        [1, 0],
        [1, 2],
        [1, 3]])
    """

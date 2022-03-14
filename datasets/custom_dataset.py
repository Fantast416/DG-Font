import torch.utils.data as data

from PIL import Image

import os
import os.path
import sys


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions):
    """

    :param dir: 字符串 根目录
    :param class_to_idx: 字典 示例： {'id_0': 0, 'id_1': 1, 'id_2': 2, 'id_3': 3, 'id_4': 4, 'id_5': 5, 'id_6': 6, 'id_7': 7, 'id_8': 8}
    :param extensions: 数组，需要关注的文件的扩展名列表
    :return: array，里面每一个元素是如下的元组： （图片的路径，通过class_to_idx进行映射得到的该图片所属类别的ID）
    """
    images = []
    dir = os.path.expanduser(dir) # 在linux下面,用os.path.expanduser把 ~ 目录展开．
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)  # 每一个类别的目录
        if not os.path.isdir(d): # 如果不是目录就跳过
            continue

        for root, _, file_names in sorted(os.walk(d)): # os.walk() 方法用于通过在目录树中游走输出在目录中的文件名，向上或者向下。
            for file_name in sorted(file_names):  # file_names 就是目录下所有的文件名
                if has_file_allowed_extension(file_name, extensions): # 判断每一个文件名是不是符合扩展名 匹配条件的
                    path = os.path.join(root, file_name)  # 如果符合条件的话，通过拼接，获得文件的目录
                    item = (path, class_to_idx[target]) # 构造一个元组——（路径，该图片所属类别的ID）
                    images.append(item)

    return images


class DatasetFolder(data.Dataset):
    """
    通用数据加载器，对数据的组织形式有要求，详细解释如下：
        A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext  # 注意：必须保证 中间的文件夹格式为  class_x ，下划线不可缺。其中： class是标识符，可以是'id'等，x为数字，如1、2、3等
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.  用于从外存中加载数据进内存的 可调用对象
        extensions (list[string]): A list of allowed extensions.  一个数组，里面是所有需要被考虑在内的 文件扩展名
        transform (callable, optional): A function/transform that takes in  一个可调用对象，用于对加载进入的样本进行变换
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes 一个可调用对象，用于对加载进入的标签进行变换
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        # 获取所有类别列表，以及 类别-idx的映射
        # classes示例： ['id_0', 'id_1', 'id_2', 'id_3', 'id_4', 'id_5', 'id_6', 'id_7', 'id_8'] root下文件夹名称的列表
        # class_to_idx示例：  {'id_0': 0, 'id_1': 1, 'id_2': 2, 'id_3': 3, 'id_4': 4, 'id_5': 5, 'id_6': 6, 'id_7': 7, 'id_8': 8}
        classes, class_to_idx = self._find_classes(root)

        # 获得所有的数据元信息——即图片目录和所属类别的元组数组
        samples = make_dataset(root, class_to_idx, extensions)

        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples  # 示例： [("../data/id_0/0000.png",0),("../data/id_0/0001.png",0),("../data/id_1/0000.png",1)]
        self.targets = [s[1] for s in samples]  # 对应上述sample的标签数组，示例：[0,0,1]

        self.transform = transform
        self.target_transform = target_transform

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.  在根目录下，发现所有类别的文件夹路径

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]

        # classes示例： ['id_0', 'id_1', 'id_2', 'id_3', 'id_4', 'id_5', 'id_6', 'id_7', 'id_8'] root下文件夹名称的列表

        underscore_idx = classes[0].find('_') # 找到下划线的idx，用于接下来排序
        classes.sort()
        classes.sort(key= lambda x:int(x[underscore_idx+1:]))  # 按照每个类别文件夹 下划线往后的字符串的大小进行排序。

        class_to_idx = {classes[i]: i for i in range(len(classes))}

        # class_to_idx示例：  {'id_0': 0, 'id_1': 1, 'id_2': 2, 'id_3': 3, 'id_4': 4, 'id_5': 5, 'id_6': 6, 'id_7': 7, 'id_8': 8}

        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        imgname = path.split('/')[-1].replace('.JPEG', '')
        return sample, target, imgname

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']

# ================图片数据加载===================
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    封装了两种图像获取引擎，accimage和Pillow，torchvision默认使用的是pillow
    如果使用pillow，最终返回的是一个Image对象
    :param path: 需要加载的图像的路径
    :return:
    """
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':   # 获取图像存取引擎，accimage 包使用的是因特尔(Intel) IPP 库。它的速度快于PIL,　但是并不支持很多的图像操作。
        return accimage_loader(path)
    else:
        return pil_loader(path)
# ================图片数据加载===================


class ImageFolerRemap(DatasetFolder):
    """
        DatasetFolder 是一个通用的数据加载器（对数据的组织形式有一定要求，见基类），继承了Pytorch的Dataset类，已经覆写了一些基本函数
        ImageFolderRemap 在其基础上又封装了一层
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, remap_table=None, with_idx=False):
        """
        :param root: 图片数据的根目录
        :param transform: 对样例的变换函数
        :param target_transform: 对标签的变换函数
        :param loader: 默认的加载器，在__getitem__函数中有作用
        :param remap_table: 重映射表，用于 ？
        :param with_idx:
        """
        super(ImageFolerRemap, self).__init__(root, loader, IMG_EXTENSIONS, transform=transform, target_transform=target_transform)
        # 在父类的构造函数里，完成了所有图像信息类别的获取 + 图像类别和类别ID的映射 + 图像路径和类别ID的元信息的获取
        self.imgs = self.samples  # 示例： [("../data/id_0/0000.png",0),("../data/id_0/0001.png",0),("../data/id_1/0000.png",1)]
        self.class_table = remap_table # 构造函数传入的重映射表
        self.with_idx = with_idx

    def __getitem__(self, index):
        path, target = self.samples[index]  # 将samples元组里的信息拆解出来，前半部分是 图片路径，后半部分是 图片所属类别，直接作为 标签
        sample = self.loader(path)  # 利用指定的加载器从路径中将图片从外存中加载至内存，然后将加载至内存的图片对象作为样本
        # 对样本做变换
        if self.transform is not None:
            sample = self.transform(sample)

        """
            此处我对代码做了少量改动，将对标签做变换的操作放到了重映射之后，如此才比较符合逻辑。
            （ 虽然此处改动对本份代码应当没有影响，因为本份代码没有target_transform ）
        """
        # 利用重映射表：将原先图片类别中的0-？ 进行映射，然后才得到最终的标签
        target = self.class_table[target]
        # 对标签做变换
        if self.target_transform is not None:
            target = self.target_transform(target)

        # 如果传入的with_idx，那么会多返回一个index信息
        if self.with_idx:
            return sample, index, target

        return sample, target      # 返回 实例和标签


"""
================此部分代码在本份代码中无用======================
"""
# class CrossDomainFolder(data.Dataset):
#     def __init__(self, root, data_to_use=['photo', 'monet'], transform=None, loader=default_loader, extensions='jpg'):
#         self.data_to_use = data_to_use
#         classes, class_to_idx = self._find_classes(root)
#         samples = make_dataset(root, class_to_idx, extensions)
#         if len(samples) == 0:
#             raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
#                                "Supported extensions are: " + ",".join(extensions)))
#
#         self.root = root
#         self.loader = loader
#         self.extensions = extensions
#
#         self.classes = classes
#         self.class_to_idx = class_to_idx
#         self.samples = samples
#         self.targets = [s[1] for s in samples]
#
#         self.transform = transform
#
#     def _find_classes(self, dir):
#         """
#         Finds the class folders in a dataset.
#
#         Args:
#             dir (string): Root directory path.
#
#         Returns:
#             tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
#
#         Ensures:
#             No class is a subdirectory of another.
#         """
#         if sys.version_info >= (3, 5):
#             # Faster and available in Python 3.5 and above
#             classes = [d.name for d in os.scandir(dir) if d.is_dir() and d.name in self.data_to_use]
#         else:
#             classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d)) and d in self.data_to_use]
#         classes.sort()
#         class_to_idx = {classes[i]: i for i in range(len(classes))}
#         return classes, class_to_idx
#
#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#
#         Returns:
#             tuple: (sample, target) where target is class_index of the target class.
#         """
#         path, target = self.samples[index]
#         sample = self.loader(path)
#         if self.transform is not None:
#             sample = self.transform(sample)
#         return sample, target
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __repr__(self):
#         fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
#         fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
#         fmt_str += '    Root Location: {}\n'.format(self.root)
#         tmp = '    Transforms (if any): '
#         fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
#         return fmt_str


if __name__ == '__main__':
    img = default_loader("../../data/id_0/0000.png")
    print(img.size)
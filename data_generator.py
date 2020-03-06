# -*- coding: utf-8 -*-

# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26}, 
#    number={7}, 
#    pages={3142-3155}, 
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/SaoYan/DnCNN-PyTorch
# =============================================================================

# no need to run this code separately

import glob
import cv2
import os
import pandas as pd
import numpy as np
# from multiprocessing import Pool
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import h5py # 通过h5py读写hdf5文件
import argparse
from networks import  *

from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.test_utils import LeNet5
from advertorch_examples.utils import get_mnist_train_loader
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import TRAINED_MODEL_PATH
from advertorch.attacks import CarliniWagnerL2Attack,GradientSignAttack,L2PGDAttack,SpatialTransformAttack,JacobianSaliencyMapAttack,MomentumIterativeAttack

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import args

patch_size, stride = 40, 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128

# Return network & file name
def getNetwork(args):
    if (args.net_type == 'lenet'):
        net = LeNet(args.num_classes)
        file_name = 'lenet'
    elif (args.net_type == 'vggnet'):
        net = VGG(args.depth, args.num_classes)
        file_name = 'vgg-'+str(args.depth)
    elif (args.net_type == 'resnet'):
        net = ResNet(args.depth, args.num_classes)
        file_name = 'resnet-'+str(args.depth)
    elif (args.net_type == 'wide-resnet'):
        net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, args.num_classes)
        file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)
    else:
        print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
        sys.exit(0)

    return net, file_name

def _get_test_adv(attack_method,epsilon):
    # define parameter
    # parser = argparse.ArgumentParser(description='Train MNIST')
    # parser.add_argument('--seed', default=0, type=int)
    # parser.add_argument('--mode', default="adv", help="cln | adv")
    # parser.add_argument('--sigma', default=75, type=int, help='noise level')
    # parser.add_argument('--train_batch_size', default=50, type=int)
    # parser.add_argument('--test_batch_size', default=1000, type=int)
    # parser.add_argument('--log_interval', default=200, type=int)
    # parser.add_argument('--result_dir', default='results', type=str, help='directory of test dataset')
    # parser.add_argument('--monitor', default=False, type=bool, help='if monitor the training process')
    # parser.add_argument('--start_save', default=90, type=int,
    #                     help='the threshold epoch which will start to save imgs data using in testing')

    # # attack
    # parser.add_argument("--attack_method", default="PGD", type=str,
    #                     choices=['FGSM', 'PGD', 'Momentum', 'STA'])
    #
    # parser.add_argument('--epsilon', type=float, default=8 / 255, help='if pd_block is used')
    #
    # parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/MNIST]')
    #
    # # net
    # parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
    # parser.add_argument('--depth', default=28, type=int, help='depth of model')
    # parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
    # parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
    # parser.add_argument('--num_classes', default=10, type=int)
    # args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # load basic data
    # 测试包装的loader
    test_loader = get_handled_cifar10_test_loader(num_workers=4, shuffle=False, batch_size=50)

    # 加载网络模型
    # Load checkpoint
    print('| Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/' + args.dataset + os.sep + file_name + '.t7')  # os.sep提供跨平台的分隔符
    model = checkpoint['net']

    #
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 定义对抗攻击类型：C&W
    from advertorch.attacks import LinfPGDAttack
    if attack_method == "PGD":
        adversary = LinfPGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=epsilon,
            nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
            targeted=False)
    elif attack_method == "FGSM":
        adversary = GradientSignAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            clip_min=0.0, clip_max=1.0, eps=0.007, targeted=False)  # 先测试一下不含扰动范围限制的，FGSM的eps代表的是一般的eps_iter
    elif attack_method == "Momentum":
        adversary = MomentumIterativeAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=epsilon,
            nb_iter=40, decay_factor=1.0, eps_iter=1.0, clip_min=0.0, clip_max=1.0,
            targeted=False, ord=np.inf)
    elif attack_method == "STA":
        adversary = SpatialTransformAttack(
            model, num_classes=args.num_classes, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            initial_const=0.05, max_iterations=1000, search_steps=1, confidence=0, clip_min=0.0, clip_max=1.0,
            targeted=False, abort_early=True)  # 先测试一下不含扰动范围限制的

    # generate for train.h5 | save as train_adv_attackMethod_epsilon
    test_adv = []
    test_true_target = []
    for clndata, target in test_loader:
        print("clndata:{}".format(clndata.size()))
        clndata, target = clndata.to(device), target.to(device)
        with ctx_noparamgrad_and_eval(model):
            advdata = adversary.perturb(clndata, target)
            test_adv.append(advdata.detach().cpu().numpy())
        test_true_target.append(target.cpu().numpy())
    test_adv = np.reshape(np.asarray(test_adv),[-1,3,32,32])
    test_true_target = np.reshape(np.asarray(test_true_target),[-1])
    print("test_adv.shape:{}".format(test_adv.shape))
    print("test_true_target.shape:{}".format(test_true_target.shape))
    del model

    return test_adv, test_true_target

def get_test_adv_loader(attack_method,epsilon):
    #save file
    if os.path.exists("data/test_adv_"+str(attack_method)+"_"+str(epsilon)+".h5"):
        h5_store = h5py.File("data/test_adv_"+str(attack_method)+"_"+str(epsilon)+".h5", 'r')
        test_data = h5_store['data'][:] # 通过切片得到np数组
        test_true_target=h5_store['true_target'][:]
        h5_store.close()
    else:

        test_data,test_true_target = _get_test_adv(attack_method,epsilon)
        h5_store = h5py.File("data/test_adv_"+str(attack_method)+"_"+str(epsilon)+".h5", 'w')
        h5_store.create_dataset('data' ,data= test_data)
        h5_store.create_dataset('true_target',data=test_true_target)
        h5_store.close()

    # 生成dataset的包装类
    train_data = torch.from_numpy(test_data)
    train_target = torch.from_numpy(test_true_target)  # numpy转Tensor
    train_dataset = CIFAR10Dataset(train_data, train_target)
    del train_data,train_target
    return DataLoader(dataset=train_dataset, num_workers=2, drop_last=True, batch_size=50,
                  shuffle=False)

#generate h5file for test data regarding specific attack
def generate_attackh5(save_dir="data",attack_method="PGD",epsilon=8/255):
    '''
    :param attack_method:
    :param epsilon:
    :return: the name of file where (test_adv_data, test_true_lable) is stored
    '''
    file_name = "test_"+attack_method+"_"+epsilon+".h5"
    file_path = os.path.join(save_dir,file_name)
    if not os.exists(save_dir):
        os.mkdir(save_dir)
    else:
        # get raw test data
        data,target = get_test_raw_data()




class CIFAR10Dataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): clean image patches
        sigma: noise level, e.g., 25
    """

    def __init__(self, data, target):
        super(CIFAR10Dataset, self).__init__()
        self.data = data
        self.target = target

    def __getitem__(self, index): # 该函数涉及到enumerate的返回值
        batch_x = self.data[index]
        batch_y = self.target[index]
        return batch_x, batch_y

    def __len__(self):
        return self.data.size(0)




def get_raw_cifar10_data(loader):
    '''
    @loader:传入一个DataLoader，借此获得源数据
    @return:返回原矩阵数据（将x，3,32,32调整成为x，32,32,3）
    '''
    train_data = []
    train_target = []

    # 循环得到训练数据
    for batch_idx, (data, target) in enumerate(loader):
        train_data.append(data.numpy())
        train_target.append(target.numpy())

    train_data = np.asarray(train_data)
    train_target = np.asarray(train_target)
    train_data = train_data.reshape([-1, 3, 32, 32])
    train_target = np.reshape(train_target, [-1])

    return train_data, train_target


def get_handled_cifar10_train_loader(batch_size, num_workers, shuffle=True):
    if os.path.exists("data/train.h5"):
        # h5_store = pd.HDFStore("data/train.h5", mode='r')
        h5_store = h5py.File("data/train.h5", 'r')
        train_data = h5_store['data'][:] # 通过切片得到np数组
        train_target = h5_store['target'][:]
        h5_store.close()
        # print("^_^ data loaded successfully from train.h5")
    else:
        h5_store = h5py.File("data/train.h5", 'w')

        transform = transforms.Compose([transforms.ToTensor()])
        trainset = CIFAR10(root="./data", train=True, download=True, transform=transform)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
        train_data, train_target = get_raw_cifar10_data(train_loader)


        h5_store.create_dataset('data' ,data= train_data)
        h5_store.create_dataset('target',data = train_target)
        h5_store.close()

    # 生成dataset的包装类
    train_data = torch.from_numpy(train_data)
    train_target = torch.from_numpy(train_target)  # numpy转Tensor
    train_dataset = CIFAR10Dataset(train_data, train_target)
    del train_data,train_target
    return DataLoader(dataset=train_dataset, num_workers=num_workers, drop_last=True, batch_size=batch_size,
                  shuffle=shuffle)


def get_handled_cifar10_test_loader(batch_size, num_workers, shuffle=True):
    if os.path.exists("data/test.h5"):
        # h5_store = pd.HDFStore("data/train.h5", mode='r')
        h5_store = h5py.File("data/test.h5", 'r')
        train_data = h5_store['data'][:] # 通过切片得到np数组
        train_target = h5_store['target'][:]
        h5_store.close()
        print("^_^ data loaded successfully from test.h5")

    else:
        h5_store = h5py.File("data/test.h5", 'w')

        # 加载数据集
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = CIFAR10(root="./data", train=False, download=True, transform=transform)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
        train_data, train_target = get_raw_cifar10_data(train_loader)


        h5_store.create_dataset('data' ,data= train_data)
        h5_store.create_dataset('target',data = train_target)
        h5_store.close()


    train_data = torch.from_numpy(train_data)
    train_target = torch.from_numpy(train_target)  # numpy转Tensor
    # 生成dataset的包装类
    train_dataset = CIFAR10Dataset(train_data, train_target)
    del train_data,train_target
    return DataLoader(dataset=train_dataset, num_workers=num_workers, drop_last=True, batch_size=batch_size,
                      shuffle=shuffle)


def data_aug(raw_img_data):
    '''
    首先对图片进行亮度调整，然后对图片进行随机的投射变换处理，少的地方进行补0处理
    @raw_img_data :[x,h,w,channel] numpy.array
    @return : [x,h,w,channel] numpy.array

    '''
    # print("input shape :{}, and will iterate {} times".format(raw_img_data.shape, np.size(raw_img_data, 0)))
    res = []
    for i in range(np.size(raw_img_data, 0)):
        # show raw image :
        # show(raw_img_data[i,:,:,:].reshape(32,32,3))

        # 对亮度进行随机调整：借助transforms.ColorJitter类，这是一个可调用对象，可以让类的实例像函数一样被调用,使用这个类需要将图片转成PIL Image
        img_data = raw_img_data[i, :, :, :]
        # img_data = np.transpose(img_data, [2, 0, 1])
        # 从PIL IMAGE转np得到的是[channel,height,width] np转PIL Image 也需要保证这一格式

        # print(img_data.shape)
        # print(img_data)
        # img_data = img_data.astype(np.uint8)

        img_data = transforms.ToPILImage()(torch.from_numpy(img_data))  # numpy->PIL Image # 假设是（32，32,3）而非（3,32,32）
        # img_data.show() # show raw image
        rand_brightness = np.random.rand()
        # print(rand_brightness)
        img_data = transforms.ColorJitter(brightness=rand_brightness)(img_data)  # modify brightness

        # 对图片进行透视变换
        transforms_proba = np.random.uniform(0.3, 1)  # probability of being perspectively transformed
        img_data = transforms.RandomPerspective(p=transforms_proba, distortion_scale=0.5)(img_data)

        # img_data.show()# show image augmented
        img_data = transforms.ToTensor()(img_data).numpy()  # PIL Image -> Tensor->numpy

        # reshape here
        res.append(img_data)

    # show augmented image
    # show(img_data)

    # print("^_^ data augmented finished with shape: {}".format(np.asarray(res).shape))
    return np.asarray(res)


def _transform_vector(self, width, x_shift, y_shift, im_scale, rot_in_degrees):
    """
    If one row of transforms is [a0, a1, a2, b0, b1, b2, c0, c1],
    then it maps the output point (x, y) to a transformed input point
    (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k),
    where k = c0 x + c1 y + 1.
    The transforms are inverted compared to the transform mapping input points to output points.
    """

    rot = float(rot_in_degrees) / 90. * (math.pi / 2)

    # Standard rotation matrix
    # (use negative rot because tf.contrib.image.transform will do the inverse)
    rot_matrix = np.array(
        [[math.cos(-rot), -math.sin(-rot)],
         [math.sin(-rot), math.cos(-rot)]]
    )

    # Scale it
    # (use inverse scale because tf.contrib.image.transform will do the inverse)
    inv_scale = 1. / im_scale
    xform_matrix = rot_matrix * inv_scale
    a0, a1 = xform_matrix[0]
    b0, b1 = xform_matrix[1]

    # At this point, the image will have been rotated around the top left corner,
    # rather than around the center of the image.
    #
    # To fix this, we will see where the center of the image got sent by our transform,
    # and then undo that as part of the translation we apply.
    x_origin = float(width) / 2
    y_origin = float(width) / 2

    x_origin_shifted, y_origin_shifted = np.matmul(
        xform_matrix,
        np.array([x_origin, y_origin]),
    )

    x_origin_delta = x_origin - x_origin_shifted
    y_origin_delta = y_origin - y_origin_shifted

    # Combine our desired shifts with the rotation-induced undesirable shift
    a2 = x_origin_delta - (x_shift / (2 * im_scale))
    b2 = y_origin_delta - (y_shift / (2 * im_scale))

    # Return these values in the order that tf.contrib.image.transform expects
    return np.array([a0, a1, a2, b0, b1, b2, 0, 0]).astype(np.float32)


def gen_patches(file_name):
    # get multiscale patches from a single image
    img = cv2.imread(file_name, 0)  # gray scale
    h, w = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h * s), int(w * s)
        img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled - patch_size + 1, stride):
            for j in range(0, w_scaled - patch_size + 1, stride):
                x = img_scaled[i:i + patch_size, j:j + patch_size]
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)
    return patches


def datagenerator(data_dir='data/Train400', verbose=False):
    # generate clean patches from a dataset
    file_list = glob.glob(data_dir + '/*.png')  # get name list of all .png files
    # initrialize
    data = []

    # generate patches
    for i in range(len(file_list)):
        patches = gen_patches(file_list[i])
        for patch in patches:
            data.append(patch)
        if verbose:
            print(str(i + 1) + '/' + str(len(file_list)) + ' is done ^_^')
    data = np.array(data, dtype='uint8')
    data = np.expand_dims(data, axis=3)
    discard_n = len(data) - len(data) // batch_size * batch_size  # because of batch namalization
    data = np.delete(data, range(discard_n), axis=0)
    print('^_^-training data finished-^_^')
    return data

def get_test_raw_data():
    '''
    :return: train_image ,  train_target  | tensor
    '''
    if os.path.exists("data/test.h5"):
        h5_store = h5py.File("data/test.h5", 'r')
        train_data = h5_store['data'][:] # 通过切片得到np数组
        train_target = h5_store['target'][:]
        h5_store.close()
    else:
        h5_store = h5py.File("data/test.h5", 'w')

        transform = transforms.Compose([transforms.ToTensor()])
        trainset = CIFAR10(root="./data", train=False, download=True, transform=transform)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
        train_data, train_target = get_raw_cifar10_data(train_loader)


        h5_store.create_dataset('data' ,data= train_data)
        h5_store.create_dataset('target',data = train_target)
        h5_store.close()

    # 生成dataset的包装类
    train_data = torch.from_numpy(train_data)
    train_target = torch.from_numpy(train_target)  # numpy转Tensor
    return train_data,train_target

def get_train_raw_data():
    '''
    :return: train_image ,  train_target  | tensor
    '''
    if os.path.exists("data/train.h5"):
        # h5_store = pd.HDFStore("data/train.h5", mode='r')
        h5_store = h5py.File("data/train.h5", 'r')
        train_data = h5_store['data'][:] # 通过切片得到np数组
        train_target = h5_store['target'][:]
        h5_store.close()
        # print("^_^ data loaded successfully from train.h5")
    else:
        h5_store = h5py.File("data/train.h5", 'w')

        transform = transforms.Compose([transforms.ToTensor()])
        trainset = CIFAR10(root="./data", train=True, download=True, transform=transform)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
        train_data, train_target = get_raw_cifar10_data(train_loader)


        h5_store.create_dataset('data' ,data= train_data)
        h5_store.create_dataset('target',data = train_target)
        h5_store.close()

    # 生成dataset的包装类
    train_data = torch.from_numpy(train_data)
    train_target = torch.from_numpy(train_target)  # numpy转Tensor
    return train_data,train_target

if __name__ == '__main__':
    data = datagenerator(data_dir='data/Train400')
    data,target = get_train_raw_data()
    print(data.shape)
    print(target.shape)

#    print('Shape of result = ' + str(res.shape))
#    print('Saving data...')
#    if not os.path.exists(save_dir):
#            os.mkdir(save_dir)
#    np.save(save_dir+'clean_patches.npy', res)
#    print('Done.')       
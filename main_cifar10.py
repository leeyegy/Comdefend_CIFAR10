# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


'''
test for the three adding steps:
1. using low-level gaussian noise
2. incorporating the add-de-noise into training model
3. using low-level adversarial noise
'''


from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.test_utils import LeNet5
from advertorch_examples.utils import get_mnist_train_loader
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import TRAINED_MODEL_PATH
from advertorch.attacks import CarliniWagnerL2Attack,GradientSignAttack,L2PGDAttack,SpatialTransformAttack,JacobianSaliencyMapAttack,MomentumIterativeAttack

from skimage.io import imread, imsave
from skimage.measure import compare_psnr, compare_ssim


import numpy as np
import matplotlib
import  time

from PIL import Image
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data_generator import get_handled_cifar10_train_loader,get_handled_cifar10_test_loader
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
import os, glob, datetime, time
import sys

from networks import  *
# sys.path.append("../../")
# from utils_net.net.networks import *


torch.multiprocessing.set_sharing_strategy('file_system')

# def show(x, title=None, cbar=False, figsize=None):
    # # plt.figure(figsize=figsize)
    # plt.imshow(x)
    # if title:
    #     plt.title(title)
    # if cbar:
    #     plt.colorbar()
    # plt.show()
def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)
def add_gaussian_nosie(batch_input,noise_level):
    '''
    :param batch_input: tensor.cuda| [batch,3,32,32]
    :return: image with known-level gaussian noise|tensor.cuda
    '''
    input = batch_input.cpu()
    res = []
    for i in range (list(input.size())[0]):
        np.random.seed(seed=0)  # for reproducibility
        y = input[i].numpy() + np.random.normal(0, noise_level / 255.0, input[i].numpy().shape)  # Add Gaussian noise without clipping
        y = y.astype(np.float32)
        res.append(y)
    res = torch.from_numpy(np.asarray(res)).view(-1,3,32,32).cuda()
    return res

def save_result(result, path):
    path = path if path.find('.') != -1 else path+'.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        imsave(path, np.clip(result, 0, 1))


def show(img_tensor):
    transforms.ToPILImage()(img_tensor).show()

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

class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)

#print args
def print_setting(args):
    import time
    print(args)
    time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MNIST')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--mode', default="adv", help="cln | adv")
    parser.add_argument('--sigma', default=50, type=int, help='noise level')
    parser.add_argument('--train_batch_size', default=50, type=int)
    parser.add_argument('--test_batch_size', default=1000, type=int)
    parser.add_argument('--log_interval', default=200, type=int)
    parser.add_argument('--model_dir', default=os.path.join('models', 'DnCNN_color_CIFAR10_sigma50'), help='directory of the model')
    parser.add_argument('--model_name', default='model_038.pth', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of test dataset')
    parser.add_argument('--monitor', default=False, type=bool, help='if monitor the training process')
    parser.add_argument('--start_save', default=90, type=int, help='the threshold epoch which will start to save imgs data using in testing')

    # pixel-deflection-wavelet
    parser.add_argument('--window'           , type=int,   default= 10)
    parser.add_argument('--deflections'      , type=int,   default= 200)
    parser.add_argument('--sigma_pixel_deflection', type=float, default= 0.04)
    parser.add_argument('--denoiser'         , type=str ,  default= 'wavelet',     help='options: wavelet, TVM, bilateral, deconv, NLM')
    parser.add_argument('--disable_map'      , type=bool,default=True)
    parser.add_argument('--pd_block', type = bool,default=True, help='if pd_block is used')

    # attack
    parser.add_argument("--attack_method", default="PGD", type=str,
                        choices=['FGSM', 'PGD','Momentum','STA'])

    parser.add_argument('--epsilon', type = float,default=8/255, help='if pd_block is used')

    # training mode
    parser.add_argument("--train_with_defence",action='store_true',help='if use defence block during training')

    #resume CNN
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

    # if apply our model
    parser.add_argument('--gaussian_block', type = bool,default=True, help='if gaussian_block is used, which will add a gaussian noise and then denoise')

    parser.add_argument('--defence_block',type=str,default="AverageSmoothing",help="GaussianSmoothing | ConvSmoothing | AverageSmoothing |MedianSmoothing |JPEGFilter | BitSqueezing|BinaryFilter")
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/MNIST]')

    #net
    parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
    parser.add_argument('--depth', default=28, type=int, help='depth of model')
    parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
    parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
    parser.add_argument('--num_classes', default=10, type=int)
    args = parser.parse_args()
    print_setting(args)

    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.mode == "cln":
        flag_advtrain = False
        nb_epoch = 12
        model_filename = "cifar_lenet5_clntrained.pt"
    elif args.mode == "adv":
        flag_advtrain = True
        nb_epoch = 90
        model_filename = "cifar_lenet5_advtrained.pt"
    else:
        raise

    train_loader = get_handled_cifar10_train_loader(num_workers=4,shuffle=True,batch_size=args.train_batch_size)
    test_loader = get_handled_cifar10_test_loader(num_workers=4,shuffle=False,batch_size=args.train_batch_size)

    #load net
    # model = ResNet18()
    if not args.resume:
        model, _ = getNetwork(args)
        nb_epoch = 150
    else:
        # Load checkpoint
        print('| Resuming from checkpoint...')
        assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
        _, file_name = getNetwork(args)
        checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7') #
        model = checkpoint['net']
        nb_epoch = 10
        # best_acc = checkpoint['acc']
        # start_epoch = checkpoint['epoch']

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)


    # load denoiser
    model_DnCNN = DnCNN(image_channels=3)
    if  os.path.exists(os.path.join(args.model_dir, args.model_name)):

        model_DnCNN = torch.load(os.path.join(args.model_dir, args.model_name))
        model_DnCNN = model_DnCNN.to(device)
        # load weights into new model
        log('load trained model on '+ args.model_name)
    else:
        print('no pre-trained model')

    #define attack
    if flag_advtrain:
        from advertorch.attacks import LinfPGDAttack
        if args.attack_method == "PGD":
            adversary = LinfPGDAttack(
                model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=args.epsilon,
                nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
                targeted=False)
        elif args.attack_method == "FGSM":
            adversary =GradientSignAttack(
                model,loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                clip_min=0.0, clip_max=1.0,eps=0.007,targeted=False)
        # elif args.attack_method == "JSMA":
        #     adversary =JacobianSaliencyMapAttack(
        #         model,num_classes=args.num_classes,
        #         clip_min=0.0, clip_max=1.0,gamma=0.145,theta=1)
        elif args.attack_method == "Momentum":
            adversary =MomentumIterativeAttack(
                model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=args.epsilon,
                nb_iter=40, decay_factor=1.0, eps_iter=1.0, clip_min=0.0, clip_max=1.0,
                targeted=False,ord=np.inf)
        elif args.attack_method == "STA":
            adversary =SpatialTransformAttack(
                model,num_classes=args.num_classes, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                initial_const=0.05, max_iterations=1000, search_steps=1, confidence=0, clip_min=0.0, clip_max=1.0,
                targeted=False,abort_early=True )




        # adversary =
        # adversary =  CarliniWagnerL2Attack(
        #     model,num_classes=10,confidence=0,
        #     targeted=True,learning_rate=0.01,binary_search_step=9,max_iterations=10000,
        #     abort_early=True,initial_const = 0.001,clip_min=0.0,clip_max=255.0,
        #     loss_fn=nn.CrossEntropyLoss(reduction="sum"))


    print ("pd_block:{}, while gaussian_block:{}".format(args.pd_block,args.gaussian_block))

    for epoch in range(nb_epoch):

        # train
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)


            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(
                output, target, reduction='elementwise_mean')
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx *
                    len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        #evaluate
        model.eval()
        test_clnloss = 0
        clncorrect = 0

        test_clnloss_nodefence = 0
        clncorrect_nodefence = 0


        if flag_advtrain:
            test_advloss = 0
            advcorrect = 0
            test_advloss_nodefence = 0
            advcorrect_nodefence = 0

        for clndata, target in test_loader:
            clndata, target = clndata.to(device), target.to(device)

            # clean data without defence
            with torch.no_grad():
                output = model(clndata.float())
            test_clnloss_nodefence += F.cross_entropy(
                output, target, reduction='sum').item()

            pred = output.max(1, keepdim=True)[1]
            clncorrect_nodefence += pred.eq(target.view_as(pred)).sum().item()


            # clean data with defence
            clndata_test_one = clndata


            with torch.no_grad():
                output = model(clndata_test_one.float())
            test_clnloss += F.cross_entropy(
                output, target, reduction='sum').item()

            pred = output.max(1, keepdim=True)[1]
            clncorrect += pred.eq(target.view_as(pred)).sum().item()

            if flag_advtrain:
                with ctx_noparamgrad_and_eval(model):
                    advdata = adversary.perturb(clndata, target)


                # no defence
                with torch.no_grad():
                    output = model(advdata.float())
                test_advloss_nodefence += F.cross_entropy(
                    output, target, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]
                advcorrect_nodefence += pred.eq(target.view_as(pred)).sum().item()

                # with defence

                # # gaussian_block
                if args.gaussian_block:
                    noise_data =  add_gaussian_nosie(advdata, args.sigma)
                    advdata = model_DnCNN(noise_data)

                with torch.no_grad():
                    output = model(advdata.float())
                test_advloss += F.cross_entropy(
                    output, target, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]
                advcorrect += pred.eq(target.view_as(pred)).sum().item()

        test_clnloss /= len(test_loader.dataset)
        print('\n(clean)Test set without defence: avg cln loss: {:.4f},'
              ' cln acc: {}/{} ({:.0f}%)\n'.format(
                  test_clnloss_nodefence, clncorrect_nodefence, len(test_loader.dataset),
                  100. * clncorrect_nodefence / len(test_loader.dataset)))

        print('\n(clean)Test set with defence: avg cln loss: {:.4f},'
              ' cln acc: {}/{} ({:.0f}%)\n'.format(
                  test_clnloss, clncorrect, len(test_loader.dataset),
                  100. * clncorrect / len(test_loader.dataset)))
        if flag_advtrain:
            test_advloss /= len(test_loader.dataset)
            print('(adv with defence)Test set: avg adv loss: {:.4f},'
                  ' adv acc: {}/{} ({:.0f}%)\n'.format(
                      test_advloss, advcorrect, len(test_loader.dataset),
                      100. * advcorrect / len(test_loader.dataset)))
            test_advloss_nodefence /= len(test_loader.dataset)
            print('(adv with no defence)Test set: avg adv loss: {:.4f},'
                  ' adv acc: {}/{} ({:.0f}%)\n'.format(
                      test_advloss_nodefence, advcorrect_nodefence, len(test_loader.dataset),
                      100. * advcorrect_nodefence / len(test_loader.dataset)))

        if args.monitor or epoch>args.start_save:
            for i in range(list(clndata.shape)[0]):
                # [3,32,32] -> [32,32,3]
                tmp = np.hstack((clndata[i].cpu().numpy(), advdata[i].cpu().numpy(), noise_data[i].cpu().numpy(),denoised_data[
                    i].detach().cpu().numpy()))
                # print (tmp.shape)
                tmp = np.transpose(tmp, [1, 2, 0])
                save_result(tmp, path=os.path.join(args.result_dir, "CIFAR10",
                                                   "epoch_" + str(epoch) + "batch_" + str(batch_idx) + "index_" + str(
                                                       i) + '_dncnn.png'))

    torch.save(
        model.state_dict(),
        os.path.join(TRAINED_MODEL_PATH, model_filename))
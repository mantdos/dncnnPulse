import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import DnCNN
from dataset import prepare_data, Dataset
from utils import *

import logging
from utils1 import utils_logger
from utils1 import utils_image as util
from utils1 import utils_option as option

from shutil import copyfile
from shutil import copy
import os
from sys import exit

import h5py
import numpy as np

# 高斯忙降噪的训练

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=20, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="B", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
parser.add_argument("--useNotebook", type=bool, default=False, help='use my notebook to excute')
# parser.add_argument("--h5FileHasOpen", type=bool, default=False, help='mark the .hr file is Opened or not')
opt = parser.parse_args()

#savePath:where train.log and net.pth located
#noise:set train image noise and valid image noise value as noiseS
def main(f,savePath,noise_B,noiseS):
    # ----------------------------------------
    # 修改命令参数
    # ----------------------------------------
    opt.val_noiseL = noiseS;

    noiseL_B=[noise_B[0],noise_B[1]]; # ingnored when opt.mode=='S'
    # ----------------------------------------
    # 配置logger模块,不同的参数对于不同名字的trainLog文件
    # ----------------------------------------
    logger_name = 'train_' + 'noiseB_' + str(noiseL_B[0])+'-' + str(noiseL_B[1]) + '&noiseVal_' + str(noiseS);
    utils_logger.logger_info(logger_name, os.path.join(savePath, logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    # logger.info(option.dict2str(opt))

    logger.info('*********parameters is depth={:d},noiseVal = {:d},noiseB=[{:d}-{:d}],HSize=40'.format(opt.num_of_layers,opt.val_noiseL,noiseL_B[0],noiseL_B[1]));
    # 加载训练数据集和验证数据集,加载的数据是经过提前预处理的train和val.h5文件，默认以set12作为测试集
    logger.info('Loading dataset ...')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)  #loader_train：算下来可以生成1863个batch
    logger.info('# of training samples: {:,d}'.format(int(len(dataset_train)))) #{:,d}:带,的整数 {:d}:不带,的整数
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # 建立模型
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(size_average=False)
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # 开始训练模型
    writer = SummaryWriter(opt.outf)
    #定义该参数下验证集和训练集的psnr列表
    trainPsnrList = [];
    valPsnrList = [];
    for epoch in range(opt.epochs):    # 这里共迭代50次
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        logger.info('learning rate {:f}'.format(current_lr))
        print('learning rate %f' % current_lr)
        #定义该epoch下训练集的平均psnr
        trainPsnr = 0;
        step = 0;
        # train
        for i, data in enumerate(loader_train, 0):
            # 测试使用，加快训练速度
            # if i==31:
            #     break;
            # 开始训练
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data   #img_train：当前这个batch内的128个数据
            if opt.mode == 'S':
                noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL/255.)   # 生成特定强度的噪声
            if opt.mode == 'B':
                noise = torch.zeros(img_train.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0,:,:,:].size()
                    noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)
            imgn_train = img_train + noise  # 将噪声添加到这个batch的训练图片中
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            noise = Variable(noise.cuda()) # caculate the real noise
            out_train = model(imgn_train) #获取模型的预测出来的噪声
            loss = criterion(out_train, noise) / (imgn_train.size()[0]*2)  #通过预测的噪声和真实噪声计算损失函数
            loss.backward()
            optimizer.step()
            # results
            model.eval()
            out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)#用带噪图片减去预测的结果，获得去噪后的图片
            psnr_train = batch_PSNR(out_train, img_train, 1.) #计算PSNR
            if step % 100 == 0:
                logger.info('[epoch {:d}][{:d}/{:d}] loss: {:.4f} PSNR_train: {:.4f}'.format(epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 100 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
            trainPsnr += psnr_train;
        #计算psnr并添加到列表
        trainPsnr = trainPsnr/step;
        logger.info('[epoch {:d}] average PSNR_train:{:.4f}'.format(epoch+1, trainPsnr));
        trainPsnrList.append(trainPsnr);
        ## the end of each epoch
        model.eval()
        #定义该epoch下训练集的psnr
        valPsnr = 0;
        # 结果验证
        psnr_val = 0
        for k in range(len(dataset_val)):
            img_val = torch.unsqueeze(dataset_val[k], 0)
            noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
            imgn_val = img_val + noise
            img_val, imgn_val = Variable(img_val.cuda(), volatile=True), Variable(imgn_val.cuda(), volatile=True)
            out_val = torch.clamp(imgn_val-model(imgn_val), 0., 1.)
            psnr_val += batch_PSNR(out_val, img_val, 1.)
        psnr_val /= len(dataset_val)
        # 添加valPsnr到列表
        valPsnrList.append(psnr_val);

        logger.info('[epoch {:d}] PSNR_val: {:.4f}'.format(epoch+1, psnr_val))
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        # log the images
        out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
        Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)
        # save model
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))
    # 将所有epoach下的psnr列表保存到文件中,保存名字为trainPsnr_+deepth+_+noise
    # destPsnrDir：存放psnr.h5文件的路径
    # 文件已经打开，句柄通过f传递进来
    trainPsnrName = "trainPsnr_dnoiseB_"+str(noise_B[0])+'-' + str(noise_B[1]) + '&noiseVal_' + str(noiseS);
    valPsnrName =   "validPsnr_dnoiseB_"+str(noise_B[0])+'-' + str(noise_B[1]) + '&noiseVal_' + str(noiseS);
    f[trainPsnrName] = trainPsnrList;
    f[valPsnrName] = valPsnrList;

#if __name__ == "__main__":
def oldMain():
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=40, stride=10, aug_times=1)
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=40, stride=10, aug_times=2)
    main()

#del all file in dir
def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

#del the empty dir in the root dir
def del_dir(path):
    ls = os.listdir(path)
    if(len(ls)==0): #the dir is a empty dir,del the dir
        os.removedirs(path);
        return;
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_dir(c_path);
    if (len(ls) == 0):  # after del son dir,del the dir
        os.removedirs(path);


def realmain():
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=40, stride=10, aug_times=1)
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)
    noise_B_List=[[40,80]];
    noiseList = [50];
    if opt.useNotebook:
        destPsnrDir = '/home/pc/Desktop/PycharmProjects/dncnn/result/dncnnModelsResult_Blind';
    else:
        destPsnrDir = '/home/p/PycharmProjects/dncnn/result/dncnnModelsResult_Blind/';
    # 因为h5文件只能打开修改一次，一次在外面打开
    if (not os.path.exists(destPsnrDir)):
        os.makedirs(destPsnrDir)
    destPsnrFileDir = destPsnrDir+"psnr.h5";
    f = h5py.File(destPsnrFileDir, 'a');
    for noise_B in noise_B_List:
        for noiseS in noiseList:
            if opt.useNotebook:
                sourceDir = '/home/pc/Desktop/PycharmProjects/dncnn/logs';
                dest = '/home/pc/Desktop/PycharmProjects/dncnn/result/dncnnModelsResult_Blind/noiseB_' + str(noise_B[0])+'-' + str(noise_B[1]) + '&noiseVal_' + str(noiseS) + '/';
            else:
                sourceDir = '/home/p/PycharmProjects/dncnn/logs';
                dest = '/home/p/PycharmProjects/dncnn/result/dncnnModelsResult_Blind/noiseB_' + str(noise_B[0])+'-' + str(noise_B[1]) + '&noiseVal_' + str(noiseS) + '/';
            #creat destination dir
            print(os.path.exists(dest));
            if (not os.path.exists(dest)):
                os.makedirs(dest)
            # do main
            main(f,dest,noise_B,noiseS);
            #try copy the two file
            try:
                ls = os.listdir(sourceDir)
                for i in ls:
                    c_path = os.path.join(sourceDir, i)
                    if not os.path.isdir(c_path):
                        copy(c_path, dest);
            except IOError as e:
                print('Unable to copy flie. %s' % e);
                exit(1);
            except:
                print('Unexpect Error');
                exit(1);
            print('copy has done');
            try:
                del_file(sourceDir);
                del_dir(sourceDir);
            except:
                print("Uodo del");
                exit(1);
            print('del has done');
    #最后关闭psnr.h5文件的写句柄
    f.close();
    # 读取psnr.h5文件并打印出来显示,同时保存在psnr.log中
    logger_name = 'psnr';
    utils_logger.logger_info(logger_name, os.path.join(destPsnrDir, logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    f = h5py.File(destPsnrDir+'/psnr.h5', 'r');
    for key in f.keys():
        str1 = str(f[key].value);
        str1 = str1.replace('  ',',');
        str1 = str1.replace(' ',',');
        logger.info('[psnr.h5 data] name:{:s}, data:{:s}'.format(f[key].name, str1));
        # print(f[key].name)  # 数据名称
        # print(f[key].shape)  # 数据的大小
        # print(f[key].value)  # 数据值


import os
if __name__ == '__main__':
    envpath = '/home/p/anaconda3/lib/python3.8/site-packages/cv2/qt/plugins/platforms'
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
    realmain()
import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DnCNN
from utils import *

import logging
from utils1 import utils_logger
from utils1 import utils_image as util
from utils1 import utils_option as option

from shutil import copyfile
from shutil import copy
import os
from sys import exit

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
opt = parser.parse_args()

def normalize(data):
    return data/255.

def main():
    # # ----------------------------------------
    # # 配置logger模块
    # # ----------------------------------------
    # logger_name = 'train'
    # utils_logger.logger_info(logger_name, os.path.join(savePath, logger_name + '.log'))
    # logger = logging.getLogger(logger_name)

    #确定模型深度和模型数据加载位置
    opt.num_of_layers = 17;
    opt.logdir = "result/dncnnPulseNoiseResult_spec/depth_17&noise_0.2";
    opt.test_noiseL = 50;
    # 加载模型
    print('Loading model ...\n')
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    device_ids = [0]
    #先建立一个原始模型，这时候只需要确定模型深度即可
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    #加载模型数据
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth')))
    model.eval()
    # 加载数据信息
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    files_source.sort()
    # 数据预处理
    psnr_test = 0
    for f in files_source: #一张一张图片的打印psnr
        # image
        Img = cv2.imread(f)
        Img = normalize(np.float32(Img[:,:,0]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img)
        # 生成噪声
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
        # 叠加噪声到图片中
        INoisy = ISource + noise
        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
        with torch.no_grad(): # this can save much memory
            Out = torch.clamp(INoisy-model(INoisy), 0., 1.) #获得去噪后的图片
        ## if you are using older version of PyTorch, torch.no_grad() may not be supported
        # ISource, INoisy = Variable(ISource.cuda(),volatile=True), Variable(INoisy.cuda(),volatile=True)
        # Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
        psnr = batch_PSNR(Out, ISource, 1.) #计算psnr
        psnr_test += psnr
        print("%s PSNR %f" % (f, psnr))  #输出
    psnr_test /= len(files_source)
    print("\naverage psnr %f" % psnr_test) #最终的平均psnr结果

if __name__ == "__main__":
    # # 设置需要测试的模型深度和噪声强度
    # depth = 10;
    # noiseS = 25;
    # sourceDir = '/home/p/PycharmProjects/dncnn/logs';
    # sourceModel = '/home/p/PycharmProjects/dncnn/logs/net.pth';
    # dest = '/home/p/PycharmProjects/dncnn/result/dncnnModelsResult/depth_' + str(depth) + '&noise_' + str(noiseS) + '/';
    main()

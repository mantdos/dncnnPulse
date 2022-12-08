import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import cv2
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import DnCNN
from dataset import prepare_data, Dataset
from utils import *
from add_noise import addPulseNoise_dncnn
from add_noise import addSaltNoise_dncnn as addSaltNoise
from add_noise import addThreasholdSaltNoise_dncnn as addThreasholdSaltNoise
from imageio import imwrite

import svm_cuda
import kornia
from feature import GD_blur,LTP_blur
import matplotlib.pyplot as plt

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

#**************************************************************************************
# 只使用固定噪声强度的脉冲噪声训练得到的dncnn模型，其中网络深度统一采用17层的深度
# 针对不同噪声强度的模型训练，保存带噪图片示例、每10个epoach后的测试集降噪后图片、最终模型和每种噪声下训练的psnr变化值。
# noise<1代表使用脉冲噪声，2>noise>1并且代表使用的盐噪声，noise>2代表是使用的经过threshhold处理过的盐噪声  noise>2.9代表盲降噪
# 保存路径：“./result/dncnnPulseNoiseResult_spec”
#**************************************************************************************


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN")

parser.add_argument("--normalizationMark", type=bool, default=True, help='是否进行特征的维度归一化')
parser.add_argument("--featureDim", type=int, default=4, help='采取的特征维度维数')
parser.add_argument("--trainSetSize", type=int, default=100000, help='训练集样本总数量')
parser.add_argument("--trainDataLoadMark", type=bool, default=False, help='训练集数据是否已添加')
parser.add_argument("--LTPPara", type=int, default=5, help='LTP值求解参数')
parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
parser.add_argument("--trainDataSaveDir", type=str, default="./data/trainData/", help='统一的训练集数据的保存路径')
parser.add_argument("--trainDataFileName", type=str, default="100000_10_60_yes.h5", help='要加载的训练集数据名称,个数_噪声min_噪声max_是否归一化')
parser.add_argument("--svmDir", type=str, default="./svmResult/model_", help='SVM模型所在文件的路径前缀')

parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=0.25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=0.25, help='noise level used on validation set')
parser.add_argument("--threasholdVal", type=int, default=30, help="在训练集设置时用于判断当前像素是否为impulse噪声点,要稍微比SVM的threashold大一些，解决漏判问题")
parser.add_argument("--midFiltersuffixDeal", type=bool, default=False, help="是否对svm处理后的threashold salt噪声图片进行二次中值滤波处理")
# parser.add_argument("--h5FileHasOpen", type=bool, default=False, help='mark the .hr file is Opened or not')
opt = parser.parse_args()
opt.device = torch.device(opt.device if torch.cuda.is_available() else "cpu")

# 对svm处理后的threashold salt噪声图片进行二次中值滤波处理，提高清晰度
def midFilter(imgn):
    if opt.midFiltersuffixDeal:
        imgn = kornia.median_blur(imgn, (3, 3));
    return imgn;

# normalizationPara = [[6.7051186363389785, 0.745013181815442, 0.2090354572660476, 0.8302966922874703, 19.0],
#                      [0.02352941176470602, 1.1384509407852406e-05, 1.4524328249818608e-05, 0.0, -19.0]];

# 每一维特征的归一化参数，即最大值和最小值
normalizationPara = [[],[]];
# 统一的trainData数据和标签
trainDataInfo = [];


# 对图片形式的二维输入数据进行特征维度归一化，如果还不存在归一化参数，先根据数据得到归一化参数
# 但是在dncnn中图片是三维的，形式为128*1*size*size*4  1*1*size*size*4
def normalizationImgDncnn(data,normalizationPara,logger):
    # logger.info("开始数据归一化");
    if len(normalizationPara[0])==0:
        logger.info("根据当前数据获取归一化参数，不可用，只适用于以为数据");
        exit(0);
        # for i in range(0,len(data[0])):
        #     max = data[:,i].max();
        #     min = data[:,i].min();
        #     normalizationPara[0].append(max);
        #     normalizationPara[1].append(min);
        # logger.info("归一化参数确定为：");
        # logger.info("max:"+str(normalizationPara[0]));
        # logger.info("min:"+str(normalizationPara[1]));
    for i in range(0,len(normalizationPara[0])):
        max = normalizationPara[0][i];
        min = normalizationPara[1][i];
        data[:,:,:,:,i] = (data[:,:,:,:,i]-min)/(max-min);
    return data;

# 获取训练集数据,纯净版，不加logger的
def getTrainDataPure():
    if not opt.trainDataLoadMark:
        opt.trainDataLoadMark = True;
        print("加载训练集数据");
        destFileDir = opt.trainDataSaveDir + opt.trainDataFileName;
        f = h5py.File(destFileDir, 'r');
        trainData = f["name"].value;
        trainLabel = f["label"].value;
        # 加载一些参数
        print("训练集数据文件名称：" + opt.trainDataFileName);
        opt.featureDim = f["featureDim"].value;
        opt.trainSetSize = len(trainLabel);
        print("特征维度：{:d}".format(opt.featureDim));
        print("样本个数：{:d}".format(opt.trainSetSize));
        opt.normalizationMark = f["normalizationMark"].value;
        print("是否归一化：" + str(opt.normalizationMark));
        trainDataInfo.append(trainData);
        trainDataInfo.append(trainLabel);
        print("备注：" + f["beizhu"].value);
        if opt.normalizationMark:
            normalizationPara[0] = f["max"].value;
            normalizationPara[1] = f["min"].value;
            print("归一化参数为：");
            print("max:" + str(normalizationPara[0]));
            print("min:" + str(normalizationPara[1]));
        print("***********************************************");
        f.close();
    return trainDataInfo[0], trainDataInfo[1];

# 获取训练集数据和归一化参数
def getTrainData(logger):
    if not opt.trainDataLoadMark:
        opt.trainDataLoadMark = True;
        logger.info("加载训练集数据");
        destFileDir = opt.trainDataSaveDir + opt.trainDataFileName;
        f = h5py.File(destFileDir, 'r');
        trainData = f["name"].value;
        trainLabel = f["label"].value;
        # 加载一些参数
        logger.info("训练集数据文件名称："+opt.trainDataFileName);
        opt.featureDim = f["featureDim"].value;
        logger.info("特征维度：{:d}".format(opt.featureDim));
        logger.info("样本个数：{:d}".format(len(trainLabel)));
        opt.normalizationMark = f["normalizationMark"].value;
        logger.info("是否归一化："+str(opt.normalizationMark));
        logger.info("备注："+f["beizhu"].value);
        trainDataInfo.append(trainData);
        trainDataInfo.append(trainLabel);
        if opt.normalizationMark:
            normalizationPara[0] = f["max"].value;
            normalizationPara[1] = f["min"].value;
            logger.info("归一化参数为：");
            logger.info("max:"+str(normalizationPara[0]));
            logger.info("min:"+str(normalizationPara[1]));
        logger.info("***********************************************");
    return trainDataInfo[0],trainDataInfo[1];

#特征值有5个，分别是GD,ABD,周围9个点的方差,中值
#通过torch直接计算  大大加快计算速度
#输入imgn是128*1*40*40或1*1*40*40,直接是tensor格式
def caculateFeature_cuda_dncnn(imgn,LTPPara):
    # 直接利用tensor方式计算图片的4个特征值
    # img_t: torch.Tensor = imgn  #kornia.utils.image_to_tensor(imgn)
    img_t = imgn;
    img_t_ndarray = img_t.squeeze().numpy()
    ######################中值
    img_mid = kornia.median_blur(img_t, (3, 3))
    img_mid = abs(img_mid - img_t);
    img_MID_ndarray = img_mid.numpy()
    ########################平均值
    img_av = kornia.box_blur(img_t, (3, 3))
    img_ABD = abs(img_av - img_t);
    img_ABD_ndarray = img_ABD.numpy()
    ################GD
    img_GD = abs(GD_blur(img_t))
    img_GD_ndarray = img_GD.numpy()
    ###############LTP
    img_LTP = LTP_blur(img_t, float(LTPPara / 255))
    img_LTP_ndarray = img_LTP.numpy();
    FeatureMatrix_cuda = np.array([img_GD_ndarray,img_ABD_ndarray,img_MID_ndarray,img_LTP_ndarray]);
    return FeatureMatrix_cuda.transpose(1,2,3,4,0);#返回128×1×size*size*featureDim

# 使用svm对imgn进行噪声检测，并返回处理后的盐噪声,imgn:128*1*size*size
def svmDeal_dncnn(imgn,logger):
    # # 先得到原始的image在clearMat这些位置的值
    imgnNumpy = imgn.numpy();
    # 对imgn中所有点计算5个特征,保存在data中，data：size*size*5,直接将tensor格式的噪声图片放进去计算特征
    # 返回np格式的特征 128*1*size*size*4
    feature = caculateFeature_cuda_dncnn(imgn,opt.LTPPara);
    if opt.normalizationMark:
        feature = normalizationImgDncnn(feature,normalizationPara,logger);
    # logger.info("开始预测");
    # 获得np格式的SVM预测结果
    out_label = svm_cuda.predictImgDncnn(feature);
    # 根据SVM预测结果得到SVM处理后的噪声图片，1代表是随机脉冲，-1代表不是
    out_label = np.where(out_label == 1, 0, 1);
    imgn_svm = out_label*imgnNumpy;
    # 将np格式的imgn_svm转换为tensor格式并返回
    imgn_svm = torch.tensor(imgn_svm);
    imgn_svm=imgn_svm.type(torch.FloatTensor)
    # imgn_svm = imgn_svm.type(torch.FloatTensor)
    return imgn_svm;

#savePath:where train.log and net.pth located
#noise:set train image noise and valid image noise value as noiseS
def main(f,savePath,depth,noiseS):
    # ----------------------------------------
    # 修改命令参数
    # ----------------------------------------
    opt.noiseL = noiseS;
    opt.val_noiseL = noiseS;
    opt.num_of_layers = depth;
    # ----------------------------------------
    # 配置logger模块,不同的参数对于不同名字的trainLog文件
    # ----------------------------------------
    logger_name = 'train_'+'depth_' + str(depth) + '&noise_' + str(noiseS);
    utils_logger.logger_info(logger_name, os.path.join(savePath, logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    # logger.info(option.dict2str(opt))

    logger.info('*********parameters is depth={:d},pulse noise intensive = {:f},是否开启中值滤波后处理：'.format(opt.num_of_layers,opt.noiseL)+str(opt.midFiltersuffixDeal));
    logger.info("threashold value = {:d},noise = {:.2f}".format(opt.threasholdVal,noiseS));

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
    noiseL_B=[0.1,0.6] # ingnored when opt.mode=='S'
    #定义该参数下验证集和训练集的psnr列表
    trainPsnrList = [];
    valPsnrList = [];
    # 加载SVM的训练数据集
    logger.info("加载SVM训练数据集，主要是需要归一化参数");
    getTrainData(logger);
    svmDir = opt.svmDir + str(0.3) + "/";
    logger.info("加载SVM模型参数：" + svmDir);
    # svmModel = joblib.load(svmDir);
    svm_cuda.loadSvmPara(svmDir, logger);
    # 开始训练
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
            # if i==4:
            #     break;
            # 开始训练
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data   #img_train：当前这个batch内的128个数据
            if opt.mode == 'S':
                if (opt.noiseL > 3):
                    imgn_train_origin = addPulseNoise_dncnn(img_train, opt.noiseL - 3);  # 生成特定强度的脉冲噪声
                    imgn_train = svmDeal_dncnn(imgn_train_origin,logger);
                    imgn_train = midFilter(imgn_train);
                elif (opt.noiseL > 2):
                    imgn_train = addThreasholdSaltNoise(img_train, opt.noiseL - 2,opt.threasholdVal);  # 生成特定强度的脉冲噪声,然后这个噪声被threashold直接处理
                elif(opt.noiseL>1):  #表示添加盐噪声,值-1为真实噪声强度
                    imgn_train = addSaltNoise(img_train, opt.noiseL-1);
                else:
                    imgn_train = addPulseNoise_dncnn(img_train, opt.noiseL);   # 生成特定强度的脉冲噪声
            if opt.mode == 'B':
                if (opt.noiseL > 2):
                    # logger.info("盲噪声图片生成,noise:{:.2f}".format(opt.noiseL));
                    imgn_train = addThreasholdSaltNoise(img_train, opt.noiseL - 2, opt.threasholdVal);  # 生成特定强度的脉冲噪声
                else:
                    logger.info("暂不支持非threasholdSaltNoise盲降噪");
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            out_train = model(imgn_train) #获取模型的预测出来的噪声
            residualImg = imgn_train - img_train;  # 待学习的残差结果
            loss = criterion(out_train, residualImg) / (imgn_train.size()[0]*2)  #通过真实的图片和残余噪声图片求解损失函数
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
        # 销毁上一次显示的所有cv2图片
        cv2.destroyAllWindows()
        for k in range(len(dataset_val)):
            img_val = torch.unsqueeze(dataset_val[k], 0)
            if opt.mode == 'S':
                if (opt.noiseL > 3):
                    imgn_val_origin = addPulseNoise_dncnn(img_val, opt.noiseL - 3);  # 生成特定强度的脉冲噪声
                    # imgn_val = imgn_val_origin;
                    imgn_val_svm = svmDeal_dncnn(imgn_val_origin, logger);
                    imgn_val = midFilter(imgn_val_svm);
                elif (opt.val_noiseL > 2):
                    imgn_val = addThreasholdSaltNoise(img_val, opt.val_noiseL - 2,opt.threasholdVal);  # 生成特定强度的脉冲噪声
                elif (opt.val_noiseL > 1):  # 表示添加盐噪声,值-1为真实噪声强度
                    imgn_val = addSaltNoise(img_val, opt.val_noiseL - 1);  # 生成特定强度的脉冲噪声
                else:
                    imgn_val = addPulseNoise_dncnn(img_val, opt.val_noiseL);  # 生成特定强度的脉冲噪声
            else:
                if (opt.val_noiseL > 2):
                    logger.info("盲噪声验证集图片生成,noise:{:.2f}，生成0.2和0.5的验证集噪声数据用于验证".format(opt.noiseL));
                    imgn_val = addThreasholdSaltNoise(img_val, 0.4,opt.threasholdVal);  # 生成特定强度的脉冲噪声
                else:
                    logger.info("暂不支持非threasholdSaltNoise盲降噪");
            img_val, imgn_val = Variable(img_val.cuda(), volatile=True), Variable(imgn_val.cuda(), volatile=True)

            out_val = torch.clamp(imgn_val-model(imgn_val), 0., 1.)
            # 每次测试保存一次含噪图片
            if epoch==0:
                if not opt.midFiltersuffixDeal:
                    if opt.noiseL>3:#说明可以保存原始脉冲噪声和svm处理后的图片
                        origin = imgn_val_origin.cpu().data.squeeze().numpy();  # origin
                        svmDeal = imgn_val_svm.cpu().data.squeeze().numpy();  # svm
                        if (not os.path.exists(savePath + "outImage/imagn")):
                            os.makedirs(savePath + "outImage/imagn");
                        plt.figure(dpi=100, figsize=(16, 8))
                        plt.subplot(1, 2, 1);
                        plt.imshow(origin, cmap='gray');
                        plt.title("pulse noise image");
                        plt.subplot(1, 2, 2);
                        plt.imshow(svmDeal, cmap='gray');
                        plt.title("svm suffix deal img");
                        plt.savefig(savePath + "outImage/imagn" + "/Image_" + str(k + 1) + "noise.png");
                        logger.info(
                            '带噪图片保存,name:' + savePath + "outImage/imagn" + "/Image_" + str(k + 1) + "noise.png");
                    else:
                        origin = imgn_val.cpu().data.squeeze().numpy();#origin
                        if (not os.path.exists(savePath+"outImage/imagn")):
                            os.makedirs(savePath+"outImage/imagn");
                        plt.figure(dpi=100)
                        plt.imshow(origin, cmap='gray');
                        plt.title("noise image");
                        plt.savefig(savePath+"outImage/imagn"+"/Image_"+str(k+1)+"noise.png");
                        logger.info('带噪图片保存,name:'+savePath+"outImage/imagn"+"/Image_"+str(k+1)+"noise.png");
                else:#如果开启了中值滤波，那么保存原始脉冲噪声图片、svm输出图片和中值滤波后的图片
                    origin = imgn_val_origin.cpu().data.squeeze().numpy();  # origin
                    svmDeal = imgn_val_svm.cpu().data.squeeze().numpy();  # svm
                    midFilterDeal = imgn_val.cpu().data.squeeze().numpy();  # origin
                    if (not os.path.exists(savePath + "outImage/imagn")):
                        os.makedirs(savePath + "outImage/imagn");
                    plt.figure(dpi=100, figsize=(16, 8))
                    plt.subplot(1, 3, 1);
                    plt.imshow(origin, cmap='gray');
                    plt.title("pulse noise image");
                    plt.subplot(1, 3, 2);
                    plt.imshow(svmDeal, cmap='gray');
                    plt.title("svm suffix deal img");
                    plt.subplot(1, 3, 3);
                    plt.imshow(midFilterDeal, cmap='gray');
                    plt.title("midFilter suffix deal img");
                    plt.savefig(savePath + "outImage/imagn" + "/Image_" + str(k + 1) + "noise.png");
                    logger.info('带噪图片保存,name:' + savePath + "outImage/imagn" + "/Image_" + str(k + 1) + "noise.png");
            # 保存每个epoach输出图片
            if epoch%10==0:
                zxl1 = out_val.cpu().data.squeeze().numpy();
                if (not os.path.exists(savePath+"outImage/epoach_"+str(epoch))):
                    os.makedirs(savePath+"outImage/epoach_"+str(epoch));
                plt.figure(dpi=100)
                plt.imshow(zxl1, cmap='gray');
                plt.title("dncnn out img");
                plt.savefig(savePath+"outImage/epoach_"+str(epoch) + "/Image_"+str(k+1)+"_denoise.png");
            psnr_val += batch_PSNR(out_val, img_val, 1.)
            #显示代码
            # zxl = imgn_val.cpu().data.squeeze().numpy();
            # plt.figure();
            # plt.subplot(1,2,1);
            # plt.imshow(zxl,cmap='gray');
            # plt.title("Image_"+str(k+1)+"_noise");
            # zxl1 = out_val.cpu().data.squeeze().numpy();
            # plt.subplot(1, 2, 2);
            # plt.imshow(zxl1, cmap='gray');
            # plt.title("Image_" + str(k + 1) + "_denoise");
            # plt.show();

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
    trainPsnrName = "trainPsnr_depth_"+ str(depth) + "&noise_" + str(noiseS);
    valPsnrName =   "validPsnr_depth_"+ str(depth) + "&noise_" + str(noiseS);
    # 如果存在先删除
    if trainPsnrName in f.keys():
        logger.info("键值："+trainPsnrName+"已存在，删除");
        del f[trainPsnrName];
    if valPsnrName in f.keys():
        logger.info("键值："+valPsnrName+"已存在，删除");
        del f[valPsnrName];
    f[trainPsnrName] = trainPsnrList;
    f[valPsnrName] = valPsnrList;
    # 将保存的数据打印到自己的log文件中
    str1 = str(f[trainPsnrName][()]);
    str1 = str1.replace('  ', ',');
    str1 = str1.replace(' ', ',');
    logger.info('[psnr.h5 data] name:{:s}, data:{:s}'.format(trainPsnrName, str1));
    str1 = str(f[valPsnrName][()]);
    str1 = str1.replace('  ', ',');
    str1 = str1.replace(' ', ',');
    logger.info('[psnr.h5 data] name:{:s}, data:{:s}'.format(valPsnrName, str1));

#if __name__ == "__main__":
def oldMain():
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=40, stride=10, aug_times=1)
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)
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
    depthList = [17];
    noiseList = [2.3];
    threasholdList = [40];
    # 因为h5文件只能打开修改一次，一次在外面打开
    destPsnrDir = './result/dncnnPulseNoiseResult_spec_ThreasholdTest';
    if (not os.path.exists(destPsnrDir)):
        os.makedirs(destPsnrDir)
    destPsnrFileDir = destPsnrDir+"/psnr.h5";
    f = h5py.File(destPsnrFileDir, 'a');
    for depth in depthList:
        for noiseS in noiseList:
            for threashold in threasholdList:
                opt.threasholdVal = threashold;
                if noiseS>2.9 and noiseS <3:
                    opt.mode = "B";#代表忙降噪
                sourceDir = './logs';
                dest = './result/dncnnPulseNoiseResult_spec_ThreasholdTest/threashold_' + str(threashold) + '&noise_' + str(noiseS) + '/';
                #creat destination dir
                print(os.path.exists(dest));
                if (not os.path.exists(dest)):
                    os.makedirs(dest)
                # do main
                main(f,dest,depth,noiseS);
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


import os
if __name__ == '__main__':
    envpath = '/home/p/anaconda3/lib/python3.8/site-packages/cv2/qt/plugins/platforms'
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
    realmain()
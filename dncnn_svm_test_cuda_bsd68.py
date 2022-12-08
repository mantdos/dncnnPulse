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
import joblib
import cv2
import matplotlib.pyplot as plt
from sklearn import svm
from add_noise import addPulseNoiseAndThreasholdSalt_SVM
import os
from sys import exit

import svm_cuda
import kornia

from add_noise import addThreasholdSaltNoise_dncnn as addThreasholdSaltNoise
from imageio import imwrite

import matplotlib.pyplot as plt

import logging
from utils1 import utils_logger
from utils1 import utils_image as util
from utils1 import utils_option as option

from shutil import copyfile
from shutil import copy
import os
from sys import exit
from feature import GD_blur,LTP_blur

import h5py
import numpy as np

#**************************************************************************************
# 针对训练好的dncnn和Svm结合测试效果
# 测试集选择使用那12张图片，噪声强度分别为0.2,0.3，0.5，0.7,0.8
# dncnn模型的记过保存zai"./result/dncnnPulseNoiseResult_spec/depth_17&noise_?/net.pth"中
# 测试集图片： "./data/Set12"
# 测试结果： 每种情况的psnr记录在“./result/dncnn_svm_test/noise_?/result.log”，
# noise为sigma情况下带噪图片、SVM的处理结果的误差图保存在"./result/dncnn_svm_test/SVMResult/noise_?/..."中
# noise为sigma情况下原始图片、带噪图片、3情况的输出结果保存在"./result/dncnn_svm_test/dncnnResult/noise_?/..."中
# svm模型保存在"./svmresultBalance/model_?/SVM.model"中
# 测试分两种：
# 1、dncnn+随机脉冲噪声效果
# 2、dncnn_salt+svm处理的随机脉冲噪声
# 3、dncnn_threasholdSalt+svm处理的随机脉冲噪声
#**************************************************************************************

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description="DnCNN_SVM")
parser.add_argument("--testSelect", type=int, default=1, help='设置测试那种情况，与注释保持一致')
parser.add_argument("--featureDim", type=int, default=4, help='采取的特征维度维数')
parser.add_argument("--dncnnDir", type=str, default="./result/dncnnPulseNoiseResult_spec/depth_17&noise_", help='dncnn模型所在文件的路径前缀')
parser.add_argument("--svmDir", type=str, default="./svmResult/model_", help='SVM模型所在文件的路径前缀')
parser.add_argument("--saveDir", type=str, default="./result/dncnn_svm_test_bsd68/", help='测试结果统一保存位置前缀')
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--threasholdVal", type=int, default=30, help="在训练集设置时用于判断当前像素是否为impulse噪声点")
parser.add_argument("--normalizationMark", type=bool, default=True, help='是否进行特征的维度归一化')
parser.add_argument("--trainDataSaveDir", type=str, default="./data/trainData/", help='统一的训练集数据的保存路径，在这里主要需要该训练集的归一化参数')
parser.add_argument("--trainDataFileName", type=str, default="100000_30_yes.h5", help='要加载的训练集数据名称,个数_噪声min_噪声max_是否归一化')
parser.add_argument("--trainDataLoadMark", type=bool, default=False, help='训练集数据是否已添加')
parser.add_argument("--LTPPara", type=int, default=5, help='LTP值求解参数')
opt = parser.parse_args()

# 统一的归一化参数
normalizationPara = [[],[]];
# 统一的trainData数据和标签
trainDataInfo = [];

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

# 对输入数据进行特征维度归一化，如果还不存在归一化参数，先根据数据得到归一化参数
def normalization(data,normalizationPara,logger):
    logger.info("开始数据归一化");
    if len(normalizationPara[0])==0:
        logger.info("归一化参数未确定，确定归一化参数");
        for i in range(0,len(data[0])):
            max = data[:,i].max();
            min = data[:,i].min();
            normalizationPara[0].append(max);
            normalizationPara[1].append(min);
        logger.info("归一化参数确定为：");
        logger.info("max:"+str(normalizationPara[0]));
        logger.info("min:"+str(normalizationPara[1]));
    for i in range(0,len(normalizationPara[0])):
        max = normalizationPara[0][i];
        min = normalizationPara[1][i];
        data[:,i] = (data[:,i]-min)/(max-min);
        # if i==4:
        #     data[:, i] = data[:,i]*20;
    return data;

# 对图片形式的二维输入数据进行特征维度归一化，如果还不存在归一化参数，先根据数据得到归一化参数
# 4*size*size
def normalizationImg(data,normalizationPara,logger):
    logger.info("开始数据归一化");
    if len(normalizationPara[0])==0:
        logger.info("归一化参数未确定，确定归一化参数");
        for i in range(0,len(data[0])):
            max = data[:,i].max();
            min = data[:,i].min();
            normalizationPara[0].append(max);
            normalizationPara[1].append(min);
        logger.info("归一化参数确定为：");
        logger.info("max:"+str(normalizationPara[0]));
        logger.info("min:"+str(normalizationPara[1]));
    for i in range(0,len(normalizationPara[0])):
        max = normalizationPara[0][i];
        min = normalizationPara[1][i];
        data[:,:,i] = (data[:,:,i]-min)/(max-min);
        # if i==4:
        #     data[:, i] = data[:,i]*20;
    return data;

# 判断x，y坐标是否在img的界外
def outOfBounds(img,x,y):
    if x>=0 and x<img.shape[0] and y>=0 and y<img.shape[1]:
        return False;
    else:
         return  True;

# 判断x，y坐标是否为边界点
def isEdgePoint(img,x,y):
    if x==0 or x==img.shape[0]-1 or y==0 or y==img.shape[1]-1:
        return True;
    else:
        return  False;

# 根据dncnn模型地址和模型层数获取模型
def getDncnnModel(dncnn_dir,num_of_layers):
    # 加载模型
    print('Loading model ...\n')
    net = DnCNN(channels=1, num_of_layers=num_of_layers);
    device_ids = [0]
    # 先建立一个原始模型，这时候只需要确定模型深度即可
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    # 加载模型数据
    model.load_state_dict(torch.load(os.path.join(dncnn_dir, 'net.pth')))
    model.eval()
    return model;

# 计算该点的各个特征值旧
#像素点的特征值有5个，分别是GD,ABD,周围9个点的方差,中值,是否为边缘特征
def caculateFeature(imgn,x,y):
    # data[curSampleSize, data.shape[1]-1] = nearbyPointNum(imgn,x,y);
    if isEdgePoint(imgn, x, y):  # 如果是边界点 特征为1
        isEage = 1;
    else:
        isEage = -1;
    #使用矩阵的方法计算
    tarMatrix,clearmat = getTargetMatrix(imgn,x,y);#在imgn中对应的矩阵和失效点矩阵
    #得到有效点的点数
    effectivePointCnt = np.sum(clearmat==1);
    #得到一列的矩阵
    tarMatrix1 = tarMatrix.reshape(9);
    #去除其中的负数
    tarMatrix1 = tarMatrix1[tarMatrix1>=0];
    tarMatrix1.sort();
    midVal = (tarMatrix1[(int)(effectivePointCnt / 2)] + tarMatrix1[(int)((effectivePointCnt - 1) / 2)])/2;
    midVal = abs(midVal-imgn[x, y]);
    HelpMat = np.ones((3,3));
    tarMatrix = np.where(tarMatrix==-1,0,tarMatrix);
    average = tarMatrix.sum()/effectivePointCnt;#均值
    ABD = abs(average-tarMatrix[1,1]);#ABD等于均值减去自己
    GD = (abs(tarMatrix-tarMatrix[1,1]*HelpMat)*clearmat).sum();#GD等于去除掉无效点后的所有差异和
    variance = ((tarMatrix - average)*(tarMatrix - average)*clearmat).sum()/(effectivePointCnt-1);#方差等于9个数减去均值的平方之和除以8
    # 求解LTP指标
    #定义LTP指标的冗余项k=5
    k = opt.LTPPara/255;
    diffMatrix = tarMatrix - tarMatrix[1,1]*HelpMat;#得到每个点和中心点的差值
    #将比中心点大与k的置1,小于k的置-1,其他的置0
    diffMatrix = np.where(diffMatrix>=k,1,diffMatrix);
    diffMatrix = np.where(diffMatrix<=-k,-1,diffMatrix);
    diffMatrix = np.where(abs(diffMatrix)<k,0,diffMatrix);
    #定义权值矩阵
    weightMatrx = np.array([
        [1,   2, 4],
        [128, 0, 8],
        [64, 32, 16]]);
    LTP =(weightMatrx*diffMatrix*clearmat).sum();
    # # 计算ROAD指标，取周围和中心点的差值，然后取绝对值的最大5个
    # [tarMatrix,centerPos] = getTarget5_5Matrix(imgn,x,y);
    # tarMatrix = abs(tarMatrix-tarMatrix[centerPos[0],centerPos[1]]*np.ones((tarMatrix.shape[0],tarMatrix.shape[1])));
    # reshapeMatrx = tarMatrix.reshape(tarMatrix.shape[0]*tarMatrix.shape[1]);
    # len  =tarMatrix.shape[0]*tarMatrix.shape[1]-1;
    # reshapeMatrx.sort();
    # ROAD = 0;
    # for i in range(0,3):
    #     ROAD = ROAD + reshapeMatrx[len-i];
    # return [[GD,ABD,variance,midVal,LTP,isEage],["GD","ABD","variance","midVal","LTP","isEage"]];
    # return [[GD,ABD,midVal,LTP,ROAD],["GD","ABD","midVal","LTP","ROAD"]];
    return [[GD,ABD,midVal,LTP],["GD","ABD","midVal","LTP"]];

# 获取目标点矩阵，用于解决边界点的特征计算问题
def getTargetMatrix(imgn,x,y):
    if x==0 and y==0:#左上角
        return [np.array([[-1,-1,-1],
                [-1,imgn[x,y],imgn[x,y+1]],
                [-1,imgn[x+1,y],imgn[x+1,y+1]]]),np.array([[0,0,0],[0,1,1],[0,1,1]])];
    elif x==0 and y == imgn.shape[1]-1:#右上角
        return [np.array([[-1,-1,-1],
                [imgn[x,y-1],imgn[x,y],-1],
                [imgn[x+1,y-1],imgn[x+1,y],-1]]),np.array([[0,0,0],[1,1,0],[1,1,0]])];
    elif x==imgn.shape[0]-1 and y == imgn.shape[1]-1:#右下脚
        return [np.array([[imgn[x-1,y-1],imgn[x-1,y],-1],
                [imgn[x,y-1],imgn[x,y],-1],
                [-1,-1,-1]]),np.array([[1,1,0],[1,1,0],[0,0,0]])];
    elif x==imgn.shape[0]-1 and y == 0:#左下脚
        return [np.array([[-1,imgn[x-1,y],imgn[x-1,y+1]],
                [-1,imgn[x,y],imgn[x,y+1]],
                [-1,-1,-1]]),np.array([[0,1,1],[0,1,1],[0,0,0]])];
    elif x > 0 and y == imgn.shape[1] - 1:  # 右边
        return [np.array([[imgn[x - 1, y - 1],imgn[x - 1, y], -1],
                [imgn[x, y - 1],imgn[x, y], -1],
                [imgn[x + 1, y - 1],imgn[x + 1, y],-1]]), np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0]])];
    elif x==0 and y >0:#上边
        return [np.array([[-1,-1,-1],
                [imgn[x,y-1],imgn[x,y],imgn[x,y+1]],
                [imgn[x+1,y-1],imgn[x+1,y],imgn[x+1,y+1]]]),np.array([[0,0,0],[1,1,1],[1,1,1]])];
    elif x > 0 and y == 0:  # 左边
        return [np.array([[-1,imgn[x - 1, y],imgn[x - 1, y+1]],
                [-1,imgn[x, y],imgn[x, y+1]],
                [-1,imgn[x + 1, y],imgn[x + 1, y+1]]]), np.array([[0, 1, 1], [0, 1, 1], [0, 1, 1]])];
    elif x == imgn.shape[0]-1 and y >0:  # 下边
        return [np.array([[-1,-1,-1],
                [imgn[x-1, y - 1],imgn[x-1, y],imgn[x-1, y + 1]],
                [imgn[x, y - 1],imgn[x, y],imgn[x, y + 1]]]), np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]])];
    else:
        return [imgn[x-1:x+2,y-1:y+2],np.ones((3,3))];

#特征值有5个，分别是GD,ABD,周围9个点的方差,中值
#通过torch直接计算  大大加快计算速度
def caculateFeature_cuda(imgn):
    # 直接利用tensor方式计算图片的4个特征值
    img_t: torch.Tensor = kornia.utils.image_to_tensor(imgn)
    img_t = img_t.unsqueeze(0).float()
    ######################中值
    img_mid = kornia.median_blur(img_t, (3, 3))
    img_mid = abs(img_mid - img_t);
    img_MID_ndarray = img_mid.squeeze().numpy()
    ########################平均值
    img_av = kornia.box_blur(img_t, (3, 3))
    img_ABD = abs(img_av - img_t);
    img_ABD_ndarray = img_ABD.squeeze().numpy()
    ################GD
    img_GD = abs(GD_blur(img_t))
    img_GD_ndarray = img_GD.squeeze().numpy()
    ###############LTP
    img_LTP = LTP_blur(img_t, float(5 / 255))
    img_LTP_ndarray = img_LTP.squeeze().numpy();
    FeatureMatrix_cuda = np.array([img_GD_ndarray,img_ABD_ndarray,img_MID_ndarray,img_LTP_ndarray]);
    return FeatureMatrix_cuda.transpose(1,2,0);#返回size*size*featureDim

# 获取目标点矩阵，5×5的框
def getTarget5_5Matrix(imgn,x,y):
    xc = 2;
    yc = 2;
    xl = x-2; #[x-2:x+2]
    xr = x+3;
    yl = y-2;
    yr = y+3;
    if xl<0:
        xl = 0;
        xc = x;
    if xr>imgn.shape[1]:
        xr = imgn.shape[1];
        xc = 2;
    if yl<0:
        yl = 0;
        yc = y;
    if yr>imgn.shape[0]:
        yr = imgn.shape[0];
        yc = 2;
    return [imgn[xl:xr,yl:yr],np.array([xc,yc])];

# 使用svm对imgn进行噪声检测，并返回处理后的盐噪声,imgn:size*size
def svmDeal(imgn,logger):
    # 对imgn中所有点计算5个特征,保存在data中，data：size*size*5
    feature = caculateFeature_cuda(imgn);
    if opt.normalizationMark:
        feature = normalizationImg(feature, normalizationPara,logger);
    logger.info("开始预测");
    out_label = svm_cuda.predictImg(feature);
    # 1代表是随机脉冲，-1代表不是
    out_label = np.where(out_label == 1, 0, 1);
    imgn_svm = out_label*imgn;
    return imgn_svm;

# 通过SVM输出的结果和真实的threashold盐噪声结果对比得到误差图和精度，并保存样图
# imgn:原始的脉冲噪声图片
# imgn_svm：SVM处理后的输出图片
# saveDir：SVM和Dncnn测试保存的总体地址
# i: 当前图片的名称
def svmResultCaculate(imgn_threasholdSalt,imgn_svm,imgn,noise,saveDir,logger,i):
    # 此时diffImg,0:正确  <0:漏判  >0:误判
    diffImg = imgn_threasholdSalt - imgn_svm;
    # 求解精度
    totalPoint = imgn.shape[0] * imgn.shape[1];  # 总共的点数
    correctPoint = np.sum(diffImg == 0);  # 检测正确的个数
    noisePoint = np.sum(imgn_threasholdSalt != imgn);  # 真实threasholdSalt噪声的个数
    missPoint = np.sum(diffImg < 0);  # 漏判的个数
    errorPoint = np.sum(diffImg > 0);  # 误判的个数
    logger.info("总点数:" + "{:d}".format(totalPoint));
    logger.info("正确检测点数:" + "{:d}".format(correctPoint));
    logger.info("噪声占比:" + "{:.4f}".format(noisePoint / totalPoint));
    logger.info("漏判个数:" + "{:d}".format(missPoint));
    logger.info("误判个数:" + "{:d}".format(errorPoint));
    logger.info("预测精度:" + "{:.4f}".format(correctPoint / totalPoint));
    logger.info("噪声点被检测到比例:" + "{:.4f}".format(
        1 - missPoint / noisePoint));
    logger.info("误判比例:" + "{:.4f}".format(
        errorPoint / noisePoint));
    # errorImg:误差图,表示SVM的预测和真实threasholdSalt之间的差异,you RGB通道,即红绿蓝
    errorImg = np.zeros((diffImg.shape[0], diffImg.shape[1], 3));
    errorImg[:, :, 0] = np.where(diffImg > 0, 255, 0);  # 红色误判
    errorImg[:, :, 1] = np.where(diffImg == 0, 255, 0);  # 绿色正确
    errorImg[:, :, 2] = np.where(diffImg < 0, 255, 0);  # 蓝色漏判
    # 显示所有图片的误差样图
    plt.figure(dpi=100, figsize=(16, 8))
    plt.subplot(1, 2, 1);
    plt.imshow(imgn, cmap='gray');
    plt.title("noise img,noise ratio:" + "{:.4f}".format(noisePoint / totalPoint));
    plt.axis('off');
    # plt.subplot(1,3,2);
    # noiseImg = np.where(imgn_threasholdSalt==0,255,0);
    # plt.imshow(noiseImg.astype(np.uint8),cmap='gray');
    # plt.title("threashold pulse noise");
    plt.subplot(1, 2, 2);
    plt.imshow(errorImg.astype(np.uint8));
    plt.title(
        "error img,predict accuracy:" + "{:.4f}".format(correctPoint / totalPoint) + ",detect ratio:" + "{:.4f}".format(
            1 - missPoint / noisePoint));
    plt.axis('off');
    # 变为tensor格式
    # ISource = torch.Tensor(Img)
    # 保存SVM的预测精度图片
    svmResultSaveDir = saveDir + "svmErrorResult/";
    if (not os.path.exists(svmResultSaveDir)):
        os.makedirs(svmResultSaveDir);
    plt.savefig(svmResultSaveDir + "noise_" + str(noise) + "_" + i);
    # logger.info('svm处理误差图片保存,name:' + svmResultSaveDir + "noise_" + str(noise) + "_" + i);

# 使用dncnn对imgn进行噪声检测，保存处理后的图片，返回本张图片的精度
# 此时的imgn和img都是经过panding，1*1*size*size的tensor格式的图片，同时该图片已经放入了GPU
# 需要原始图片以计算精度
# i：当前图片的名称
# suffix：保存的图片名称的后缀，用于识别是那种dncnn的结果
def dncnnDeal(img,imgn,dncnnModel,noise,i,suffix,logger):
    outVal = torch.clamp(imgn - dncnnModel(imgn), 0., 1.)  # 获得去噪后的图片
    psnr = batch_PSNR(outVal, img, 1.)  # 计算psnr
    ssim = util.calculate_ssim(outVal, img, border=1)
    # 保存含噪图片和输出图片，保存地址为saveDir+ "noise_" + str(noise)+"/dncnnOutPutResult/";
    saveDir = opt.saveDir;
    saveDir = saveDir+ "noise_" + str(noise)+"/dncnnOutPutResult/";
    savedImgn = imgn.cpu().data.squeeze().numpy();
    savedOutImg = outVal.cpu().data.squeeze().numpy();
    # 保存噪声图和误差图片
    plt.figure(dpi=100, figsize=(16, 8))
    plt.subplot(1, 2, 1);
    plt.imshow(savedImgn, cmap='gray');
    plt.title("noise img,noise intense:" + "{:.3f}".format(noise));
    plt.axis('off');
    plt.subplot(1, 2, 2);
    plt.imshow(savedOutImg, cmap='gray');
    plt.title("psnr={:.3f}".format(psnr));
    plt.axis('off');
    if (not os.path.exists(saveDir)):
        os.makedirs(saveDir);
    plt.savefig(saveDir + "noise" + str(noise) + "_" + suffix+ "_" + i);
    logger.info("psnr={:.3f}".format(psnr));
    return [psnr,ssim];

# 自动测试三种情况的结果
# noiseList:此次需要测试那些噪声的情况
def testThreeSituation(noiseList):
    for noise in noiseList:
        save_dir = opt.saveDir;
        if not os.path.exists(save_dir):
            os.mkdir(save_dir);
        averPsnrList_dncnnOrigin = [];
        averPsnrList_dncnnSalt = [];
        averPsnrList_dncnnThreasholdSalt = [];
        averPsnrList_dncnnSvmtrain = [];
        averSsimList_dncnnOrigin = [];
        averSsimList_dncnnSalt = [];
        averSsimList_dncnnThreasholdSalt = [];
        averSsimList_dncnnSvmtrain = [];
        save_dir = save_dir + "noise_" + str(noise) +"/";
        # 注册logger文件
        logger_name = "result_"+str(noise);
        logger_dir = "./result/dncnn_svm_test_bsd68/noise_"+str(noise);
        if not os.path.exists(logger_dir):
            os.mkdir(logger_dir);
        utils_logger.logger_info(logger_name, os.path.join(logger_dir, logger_name + '.log'));
        logger = logging.getLogger(logger_name);
        logger.info("加载训练集数据，主要是为了获取归一化参数");
        getTrainData(logger);
        logger.info("dncnn层数："+str(opt.num_of_layers));
        # 根据噪声强度获取三种dncnn模型
        dncnn_origin_dir = opt.dncnnDir + str(noise);
        dncnn_salt_dir = opt.dncnnDir + str(noise+1);  # 盐噪声模型
        dncnn_threasholdSalt_dir = opt.dncnnDir + str(noise+2);
        dncnn_svmtrain_dir = opt.dncnnDir + str(noise+3);
        logger.info("加载dncnnModel_origin模型："+dncnn_origin_dir);
        dncnnModel_origin = getDncnnModel(dncnn_origin_dir,opt.num_of_layers);
        logger.info("加载dncnnModel_salt模型："+dncnn_salt_dir);
        dncnnModel_salt = getDncnnModel(dncnn_salt_dir,opt.num_of_layers);
        logger.info("加载dncnnModel_threasholdSalt模型："+dncnn_threasholdSalt_dir);
        dncnnModel_threasholdSalt = getDncnnModel(dncnn_threasholdSalt_dir,opt.num_of_layers);
        logger.info("加载dncnnModel_svm联调模型："+dncnn_svmtrain_dir);
        dncnn_svmtrain_dir = getDncnnModel(dncnn_svmtrain_dir,opt.num_of_layers);
        # 根据噪声强度获取SVM模型
        svmDir = opt.svmDir+str(noise)+"/";
        logger.info("加载SVM模型参数："+svmDir);
        # svmModel = joblib.load(svmDir);
        svm_cuda.loadSvmPara(svmDir,logger);
        # 取出每张图片开始处理
        imageDir = "./data/Set68";
        ls = os.listdir(imageDir);
        ls.sort();
        for i in ls:
            image_path = os.path.join(imageDir, i);#得到图片地址
            # 加载图片
            img = cv2.imread(image_path)[:,:,0]
            img = img/255;
            img = np.expand_dims(img, 0)
            img = np.expand_dims(img, 1)
            # 为img添加普通随机脉冲噪声 img:1*1*size*size
            imgn,imgn_threasholdSalt = addPulseNoiseAndThreasholdSalt_SVM(img[0,0,:,:], noise,opt.threasholdVal);  # 生成特定强度的脉冲噪声
            # 得到使用svm处理的盐噪声  imgn:size*size
            logger.info("SVM预处理图片："+image_path+"******************************************");
            imgn_svm = svmDeal(imgn,logger);
            svmResultCaculate(imgn_threasholdSalt, imgn_svm, imgn, noise, save_dir, logger, i);
            # 开始对比三种dncnn的输出结果××××××××××××××××××××××××××××
            # 不管是imgn还是imgn_threasholdSalt，都需要先panding一下,然后转换成tensor数据
            imgn = np.expand_dims(imgn, 0)
            imgn = np.expand_dims(imgn, 1)
            imgn = torch.tensor(imgn);
            imgn = imgn.type(torch.FloatTensor);

            img = torch.tensor(img);
            img = img.type(torch.FloatTensor);

            imgn_svm = np.expand_dims(imgn_svm, 0)
            imgn_svm = np.expand_dims(imgn_svm, 1)
            imgn_svm = torch.tensor(imgn_svm);
            imgn_svm = imgn_svm.type(torch.FloatTensor);
            # 将img、imgn、imgn_svm放入到GPU中
            img, imgn, imgn_svm = Variable(img.cuda()), Variable(imgn.cuda()), Variable(imgn_svm.cuda())
            # 1、dncnn+随机脉冲噪声效果
            logger.info("对比原始dncnn+随机脉冲噪声降噪效果");
            [psnr,ssim] = dncnnDeal(img, imgn, dncnnModel_origin, noise, i, "dncnnOrigin", logger);
            averPsnrList_dncnnOrigin.append(psnr);
            averSsimList_dncnnOrigin.append(ssim);
            # 2、dncnn_salt+svm处理的随机脉冲噪声
            logger.info("对比dncnnSalt+SVM处理后噪声降噪效果");
            [psnr, ssim] =dncnnDeal(img, imgn_svm, dncnnModel_salt, noise, i, "dncnnSalt", logger)
            averPsnrList_dncnnSalt.append(psnr);
            averSsimList_dncnnSalt.append(ssim);
            # 3、dncnn_threasholdSalt+svm处理的随机脉冲噪声
            logger.info("对比dncnnThreasholdSalt+SVM处理后噪声降噪效果");
            [psnr, ssim] = dncnnDeal(img, imgn_svm, dncnnModel_threasholdSalt, noise, i, "dncnnThreasholdSalt", logger);
            averPsnrList_dncnnThreasholdSalt.append(psnr);
            averSsimList_dncnnThreasholdSalt.append(ssim);
            # 4、dncnn_svm联调模型+svm处理的随机脉冲噪声
            logger.info("对比dncnnSVMTrain+SVM处理后噪声降噪效果");
            [psnr, ssim] = dncnnDeal(img, imgn_svm, dncnn_svmtrain_dir, noise, i, "dncnn_svmtrain", logger);
            averPsnrList_dncnnSvmtrain.append(psnr);
            averSsimList_dncnnSvmtrain.append(ssim);
        logger.info("***********************************************************************");
        logger.info("原始dncnn PSNR处理结果："+str(averPsnrList_dncnnOrigin));
        logger.info("原始dncnn PSNR平均值：{:.3f}".format(np.sum(averPsnrList_dncnnOrigin)/68));
        logger.info("原始dncnn SSIM处理结果："+str(averSsimList_dncnnOrigin));
        logger.info("原始dncnn SSIM平均值：{:.3f}".format(np.sum(averSsimList_dncnnOrigin)/68));

        logger.info("dncnnSalt PSNR处理结果："+str(averPsnrList_dncnnSalt));
        logger.info("dncnnSalt PSNR平均值：{:.3f}".format(np.sum(averPsnrList_dncnnSalt)/68));
        logger.info("dncnnSalt SSIM处理结果："+str(averSsimList_dncnnSalt));
        logger.info("dncnnSalt SSIM平均值：{:.3f}".format(np.sum(averSsimList_dncnnSalt)/68));

        logger.info("dncnnThreasholdSalt PSNR处理结果："+str(averPsnrList_dncnnThreasholdSalt));
        logger.info("dncnnThreasholdSalt PSNR平均值：{:.3f}".format(np.sum(averPsnrList_dncnnThreasholdSalt)/68));
        logger.info("dncnnThreasholdSalt SSIM处理结果："+str(averSsimList_dncnnThreasholdSalt));
        logger.info("dncnnThreasholdSalt SSIM平均值：{:.3f}".format(np.sum(averSsimList_dncnnThreasholdSalt)/68));

        logger.info("dncnn_svmtrain PSNR处理结果："+str(averPsnrList_dncnnThreasholdSalt));
        logger.info("dncnn_svmtrain PSNR平均值：{:.3f}".format(np.sum(averPsnrList_dncnnSvmtrain)/68));
        logger.info("dncnn_svmtrain SSIM处理结果："+str(averSsimList_dncnnThreasholdSalt));
        logger.info("dncnn_svmtrain SSIM平均值：{:.3f}".format(np.sum(averSsimList_dncnnSvmtrain)/68));

if __name__ == '__main__':
    envpath = '/home/p/anaconda3/lib/python3.8/site-packages/cv2/qt/plugins/platforms'
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
    if not os.path.exists("./result/"):
        os.mkdir("./result/");
    testThreeSituation([0.2,0.3,0.4,0.5,0.6]);
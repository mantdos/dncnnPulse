# -*- coding:utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import skimage.filters
from sklearn import svm
import numpy as np
import matplotlib
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
from add_noise import addPulseNoise_SVM
import random
import joblib
from  add_noise import addPulseNoiseAndThreasholdSalt_SVM
from skimage.morphology import  disk
import svm_cuda
import kornia
import torch
#贝叶斯优化代码
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from ax.utils.tutorials.cnn_utils import train, evaluate

import logging
from utils1 import utils_logger
from utils1 import utils_image as util
from utils1 import utils_option as option
import h5py

import torch
import torch.nn as nn
import torch.optim as optim
from feature import GD_blur,LTP_blur

#**************************************************************************************
# 使用GD、ABD、标准差、中位值和TLP作为特征，同时在GD和ABD的计算中不使用填0操作，只对有效点求解
# 同时经过样本均均衡处理，输出结果保存在"svmResult_12_8"文件夹中
# 贝叶斯只优化3个模型参数，不优化训练集大小
# 贝叶斯的训练集和Main的训练集都采用指定数目的同样的数据集，数据集通过h5文件保存
# 贝叶斯的测试集数据采用随机的图片，main函数的测试集样本直接采用所有12张测试图片
#**************************************************************************************


#classifier = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo')  # 一对多
parser = argparse.ArgumentParser(description="SVM")
# 文件保存路径

parser.add_argument("--saveDir", type=str, default="./svmResult/", help='SVM的整体保存路径')
parser.add_argument("--trainDataSaveDir", type=str, default="./data/trainData/", help='统一的训练集数据的保存路径')
parser.add_argument("--trainDataFileName", type=str, default="100000_10_60_yes.h5", help='要加载的训练集数据名称,个数_噪声min_噪声max_是否归一化')
parser.add_argument("--createNewDataFileMark", type=bool, default=False, help='是否创建新的训练集数据并保存')
parser.add_argument("--createNewDataFileName", type=str, default="100000_10_60_yes.h5", help='要创建的新的训练集数据名称')
parser.add_argument("--createNewDataFileBeiZhu", type=str, default="没有什么特别说明", help='要创建的新的训练集数据的额外信息备注')
#参数设置
parser.add_argument("--use09Img", type=bool, default=True, help='是否引入09的图片')
parser.add_argument("--normalizationMark", type=bool, default=True, help='是否进行特征的维度归一化')
parser.add_argument("--featureDim", type=int, default=4, help='采取的特征维度维数')
parser.add_argument("--trainSetSize", type=int, default=100000, help='训练集样本总数量')
parser.add_argument("--sampleNumPerImg", type=int, default=1000, help="每种噪声强度下选择多少个样本")
parser.add_argument("--trainImg", type=str, default="./data/Set12/08.png", help='训练集使用的图片')
parser.add_argument("--testSetSize", type=int, default=30000, help='测试集样本总数量')
parser.add_argument("--testImg", type=str, default="./data/Set12/08.png", help='测试集使用的图片')
parser.add_argument("--para_C", type=float, default=62.85, help="惩罚系数")
parser.add_argument("--para_kernel", type=str, default='linear', help="SVM核函数,rbf  linear")
parser.add_argument("--para_gamma", type=float, default=17.09, help="SVMgamma系数")
parser.add_argument("--threasholdVal", type=int, default=10, help="在训练集设置时用于判断当前像素是否为impulse噪声点")
parser.add_argument("--noiseMin", type=float, default=0.1, help='噪声强度下限')
parser.add_argument("--noiseMax", type=float, default=0.6, help='噪声强度上限')
parser.add_argument("--noiseSpec", type=float, default=-1, help='是否采用固定噪声，不采用就=-1')
parser.add_argument("--trainDataLoadMark", type=bool, default=False, help='训练集数据是否已添加')
parser.add_argument("--LTPPara", type=int, default=5, help= 'LTP值求解参数')
#程序运行状态参数
parser.add_argument("--isTestMode", type=bool, default=False, help='采用什么模式运行代码')
parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
args = parser.parse_args()
args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
opt = parser.parse_args()

# normalizationPara = [[6.7051186363389785, 0.745013181815442, 0.2090354572660476, 0.8302966922874703, 19.0],
#                      [0.02352941176470602, 1.1384509407852406e-05, 1.4524328249818608e-05, 0.0, -19.0]];

# 每一维特征的归一化参数，即最大值和最小值
normalizationPara = [[],[]];
# 统一的trainData数据和标签
trainDataInfo = [];

# 获取训练集数据
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
        opt.trainSetSize = len(trainLabel);
        logger.info("特征维度：{:d}".format(opt.featureDim));
        logger.info("样本个数：{:d}".format(opt.trainSetSize));
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
        f.close();
    return trainDataInfo[0],trainDataInfo[1];

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

# 对一维的输入数据进行特征维度归一化，如果还不存在归一化参数，先根据数据得到归一化参数
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

#计算该点附近有多少个合理点  暂未采用
def nearbyPointNum(img,x,y):
    cnt = 0;
    if x==0:
        cnt += 1;
    if x==img.shape[0]-1:
        cnt += 1;
    if y==0:
        cnt += 1;
    if y==img.shape[1]-1:
        cnt += 1;
    if cnt == 0: #不是边缘点
        return 8;
    elif cnt == 1:
        return 5;
    else:
        return 3;

#像素点的特征值有5个，分别是GD,ABD,周围9个点的方差,中值,是否为边缘特征
def caculateFeature(imgn,x,y):
    # data[curSampleSize, data.shape[1]-1] = nearbyPointNum(imgn,x,y);
    # if isEdgePoint(imgn, x, y):  # 如果是边界点 特征为1
    #     isEage = 1;
    # else:
    #     isEage = -1;
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
    # variance = ((tarMatrix - average)*(tarMatrix - average)*clearmat).sum()/(effectivePointCnt-1);#方差等于9个数减去均值的平方之和除以8
    # 求解LTP指标
    #定义LTP指标的冗余项k=5
    k = opt.LTPPara/255;
    diffMatrix = tarMatrix - tarMatrix[1,1]*HelpMat;#得到每个点和中心点的差值
    #将比中心点大与k的置1,小于k的置-1,其他的置0
    diffMatrix = np.where(diffMatrix>k,1,diffMatrix);
    diffMatrix = np.where(diffMatrix<-k,-1,diffMatrix);
    diffMatrix = np.where(abs(diffMatrix)<1,0,diffMatrix);
    #定义权值矩阵
    weightMatrx = np.array([
        [1,   2, 4],
        [128, 0, 8],
        [64, 32, 16]]);
    LTP =(weightMatrx*diffMatrix*clearmat).sum();
    # ABD = 0;
    # midVal = 0;
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

# 测试特征指标，用一张30×30的图就够了
def testFeatureIndex():
    # 读取训练集图片，选择其中一个30×30的图片
    trainImg = cv2.imread(opt.trainImg)[:,:,0];
    trainImg = trainImg/255;
    selectX = 148;
    selectY = 73;
    length = 8;
    img = trainImg[selectX:selectX+length,selectY:selectY+length];
    #为testImg添加噪声
    sigma = 0.2;
    imgn = addPulseNoise_SVM(img,sigma);
    #对比噪声前后图片 获得差值
    diffImg = abs(imgn-img);
    #计算每个参数指标
    GDMatrix = np.zeros((length,length));  # 灰度值差值
    ABDMatrix = np.zeros((length,length));  # 平均背景差值
    varianceMatrix = np.zeros((length,length));#方差
    midValMatrix = np.zeros((length,length));  # 中位值
    edgeMatrix = np.zeros((length,length));  #
    for x in range(0,length):
        for y in range(0,length):
            [GD,ABD,midVal,variance,isEage] = caculateFeature(imgn,x,y);
            GDMatrix[x,y] = GD;
            ABDMatrix[x,y] = ABD;
            varianceMatrix[x,y] = variance;
            midValMatrix[x,y] = midVal;
            edgeMatrix[x,y] = isEage;
    print("haha");

# 测试整张图片的特征指标
def testFeatureIndex_wholeImg():
    # 读取训练集图片，选择其中一个30×30的图片
    img1 = cv2.imread(opt.trainImg);
    img = img1[:,:,0];
    img = img/255;
    size1 = img.shape[0];
    size2 = img.shape[1];
    #为testImg添加噪声
    sigma = 0.2;
    imgn = addPulseNoise_SVM(img,sigma);
    imgn1 = np.zeros((img1.shape[0],img1.shape[1],img1.shape[2]));
    imgn1[:,:,0] = imgn;
    imgn1[:,:,1] = imgn;
    imgn1[:,:,2] = imgn;
    imgn1 = np.uint8(imgn1*255);
    #对比噪声前后图片 获得差值
    diffImg = abs(imgn-img);
    #计算每个参数指标
    FeatureMatrix = np.zeros((opt.featureDim,size1,size2));
    for x in range(0,size1):
        for y in range(0,size2):
            ret = caculateFeature(imgn, x, y);
            FeatureMatrix[:,x,y] = ret[0];
            FeatureName = ret[1];
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
    img_ABD_ndarray=img_ABD.squeeze().numpy()
    ################GD
    img_GD = abs(GD_blur(img_t))
    img_GD_ndarray=img_GD.squeeze().numpy()
    ###############LTP
    img_LTP = LTP_blur(img_t, float(5 / 255))
    img_LTP_ndarray=img_LTP.squeeze().numpy()
    FeatureMatrix_cuda = np.zeros((opt.featureDim,size1,size2));
    FeatureMatrix_cuda = np.array([img_GD_ndarray,img_ABD_ndarray,img_MID_ndarray,img_LTP_ndarray]);
    diff = FeatureMatrix_cuda - FeatureMatrix;
    # 归一化所有特征
    # FeatureMatrix = FeatureMatrix_cuda;
    for i in range(0,opt.featureDim):
        FeatureMatrix[i,:,:] = (FeatureMatrix[i,:,:]-FeatureMatrix[i,:,:].min())/(FeatureMatrix[i,:,:].max()-FeatureMatrix[i,:,:].min());
    # 对LTP使用边缘算子
    # LTP1 = np.zeros((img1.shape[0],img1.shape[1],img1.shape[2]));
    # LTP1[:,:,0] = FeatureMatrix[4,:,:];
    # LTP1[:,:,1] = FeatureMatrix[4,:,:];
    # LTP1[:,:,2] = FeatureMatrix[4,:,:];
    # LTP1 = np.uint8(LTP1*255);
    # LTP = FeatureMatrix[4,:,:];
    # a = 0.3;
    # LTP = np.where(LTP<a,1,LTP);
    # LTP = np.where(LTP>(1-a),1,LTP);
    # LTP = np.where(LTP<1,0,LTP);
    # sobel = LTP;
    # sobel=cv2.Canny(LTP1,1,2,apertureSize=3);#
    # 显示原图和带噪图片
    plt.figure(dpi=100, figsize=(8, 4))
    plt.subplot(1, 2, 1);
    plt.imshow(img, cmap='gray');
    plt.axis('off');
    # plt.title("img");
    plt.subplot(1, 2, 2);
    # noise = imgn;#灰度图值越大越白
    # noise = np.where(noise>opt.threasholdVal/255,1,0);
    plt.imshow(imgn,cmap='gray');
    plt.axis('off');
    # plt.title("noise img,noise:"+"{:.2f}".format(sigma));
    # 显示所有特征的分布图
    plt.figure(dpi=100,figsize=(16, 8))
    for i in range(0,opt.featureDim):
        plt.subplot(2, 3, i+1);
        plt.imshow(FeatureMatrix[i,:,:],cmap='gray');
        plt.axis('off');
        # plt.title(FeatureName[i]+",noise:".format(i+1)+"{:.2f}".format(sigma));
    #
    # plt.subplot(2, 3, 6);
    # plt.imshow(sobel, cmap='gray');
    # plt.title("sobel");

    plt.show();
    print("haha");

#*******************************
# 生成用于SVM训练或验证的脉冲噪声特征和标签
# 在本版本中仅用于生成贝叶斯优化中SVM用于测试的随机样本均衡的数据
# img:经过归一化后的原始图片
# setSize:生成的样本数量
# sampleNumPerImg：每种噪声强度下选择多少个样本
# noiseMin、noiseMax:生成的噪声上下限
# noiseSpec:是否固定图片的噪声强度，为-1表示不固定
# 返回n×*和n×1的数据和标签
#*******************************
def generateSample(img,setSize,sampleNumPerImg,noiseMin,noiseMax,noiseSpec):
    #定义特征矩阵[n×m]和标签矩阵[n×1]，n：样本个数  m：特征维度
    data = np.zeros((setSize,opt.featureDim));
    label = np.zeros((setSize,1));
    #当前样本数量为0
    curSampleSize = 0;
    if noiseSpec < 0:#说明是生成随机噪声
        while curSampleSize<setSize:
            #生成随机的噪声强度
            sigma = random.uniform(noiseMin,noiseMax);
            print("噪声强度：{:f}".format(sigma));
            #根据噪声强度生成噪声图片
            imgn = addPulseNoise_SVM(img,sigma);
            #随机选sampleNumPerImg个不同的点作为训练集样本
            randomList = random.sample(range(0, img.shape[0]*img.shape[1]-1), sampleNumPerImg);
            for pos in randomList:
                x = int(pos/img.shape[1]);
                y = pos%img.shape[1];
                # 判定该点是否为随机脉冲点
                if abs(imgn[x,y]-img[x,y])>opt.threasholdVal/255:
                    # 1代表是随机脉冲，-1代表不是
                    label[curSampleSize] = 1;
                else:
                    label[curSampleSize] = -1;
                data[curSampleSize,:] = caculateFeature(imgn,x,y)[0];
                curSampleSize += 1;
                #样本足够了
                if curSampleSize>=setSize:
                    print("共产生："+str(setSize)+"个样本");
                    break;
    else:#说明是生成固定值噪声 noiseSpec>0,并且可以直接一次搞定，不需要不断循环
        sigma = noiseSpec;
        noiseLabelSize = int(setSize/2);#噪声点个数为总个数一半
        unNoiseLabelSize = setSize - noiseLabelSize;
        while curSampleSize < setSize:
            # 生成随机的噪声强度
            sigma = random.uniform(noiseMin, noiseMax);
            # 根据噪声强度生成噪声图片
            imgn = addPulseNoise_SVM(img, sigma);
            # 随机选图片中一半个不同的点作为训练集样本
            randomList = random.sample(range(0, img.shape[0] * img.shape[1] - 1), int(img.shape[0] * img.shape[1]/2));
            for pos in randomList:
                x = int(pos / img.shape[1]);
                y = pos % img.shape[1];
                # 判定该点是否为随机脉冲点
                if abs(imgn[x, y] - img[x, y]) > opt.threasholdVal / 255:
                    # 1代表是随机脉冲，-1代表不是
                    if noiseLabelSize > 0:
                        noiseLabelSize -=1;
                    else:#说明噪声点已经够了
                        continue;
                    label[curSampleSize] = 1;
                else:
                    if unNoiseLabelSize > 0:
                        unNoiseLabelSize -=1;
                    else:#说明非噪声点已经够了
                        continue;
                    label[curSampleSize] = -1;
                data[curSampleSize,:] = caculateFeature(imgn,x,y)[0];
                curSampleSize += 1;
                # 样本足够了
                if curSampleSize >= setSize:
                    print("共产生：" + str(setSize) + "个样本");
                    break;
    return data,label;

# 训练集数据初始化，并将初始化的数据通过h5文件保存
# 该数据用于所有SVM模型的训练过程
# img:图片的地址
# setSize:生成的样本数量
# sampleNumPerImg：每种噪声强度下选择多少个样本
# noiseMin、noiseMax:生成的噪声上下限
# noiseSpec:是否固定图片的噪声强度，为-1表示不固定
# h5SaveDir：文件的保存地址
# FileName：h5文件的名称
def trainDataInit(imgDir,setSize,sampleNumPerImg,noiseMin,noiseMax,noiseSpec,h5SaveDir,FileName):
    if not os.path.exists(h5SaveDir):
        os.mkdir(h5SaveDir);
    # 注册logger文件*********************************
    logger_name = "h5FileSave";
    utils_logger.logger_info(logger_name,os.path.join(h5SaveDir,logger_name + '.log'));
    logger = logging.getLogger(logger_name);
    logger.info("生成新的训练集数据××××××××××××××××××××××××××××××××××××××××");
    logger.info("文件名称："+ FileName);
    logger.info("噪声强度范围{:f}-{:f}".format(noiseMin,noiseMax));
    # 加载h5文件，直接采用复写而不是添加模式*********************************
    destH5FilePath = h5SaveDir + FileName;
    f = h5py.File(destH5FilePath, 'w');
    # 开始循环添加噪声，生成样本*********************************
    # 定义特征矩阵[n×m]和标签矩阵[n×1]，n：样本个数  m：特征维度
    data = np.zeros((setSize, opt.featureDim));
    label = np.zeros((setSize, 1));
    # 当前样本数量为0
    curSampleSize = 0;
    cnt=0;
    if noiseSpec < 0:  # 说明是生成随机噪声
        while curSampleSize < setSize:
            # 读取图片并归一化*********************************
            # if random.randint(0,100) % 2 == 0:
            #     imgDir1 = imgDir;
            # else:
            #     imgDir1 = "./data/Set12/09.png";
            imgDir1 = imgDir;
            logger.info("读取图片：" + imgDir1);
            img = cv2.imread(imgDir1)[:, :, 0]
            img = img / 255;
            # 生成随机的噪声强度
            sigma = random.uniform(noiseMin, noiseMax);
            cnt += 1;
            logger.info("第{:d}次样本抽取，噪声强度：{:.4f}".format(cnt,sigma));
            # 根据噪声强度生成噪声图片
            imgn = addPulseNoise_SVM(img, sigma);
            # 把这张图片的噪声计算出来
            feature = caculateFeature_cuda(imgn); #size*size*featureDim
            # 随机选sampleNumPerImg个不同的点作为训练集样本
            randomList = random.sample(range(0, img.shape[0] * img.shape[1] - 1), sampleNumPerImg);
            for pos in randomList:
                x = int(pos / img.shape[1]);
                y = pos % img.shape[1];
                # 判定该点是否为随机脉冲点
                if abs(imgn[x, y] - img[x, y]) > opt.threasholdVal / 255:
                    # 1代表是随机脉冲，-1代表不是
                    label[curSampleSize] = 1;
                else:
                    label[curSampleSize] = -1;
                data[curSampleSize, :] = feature[x,y,:];
                curSampleSize += 1;
                # 样本足够了
                if curSampleSize >= setSize:
                    logger.info("训练集样本生成完毕");
                    break;
    else:  # 说明是生成固定值噪声 noiseSpec>0,并且可以直接一次搞定，不需要不断循环
        while curSampleSize < setSize:
            # 读取图片并归一化*********************************
            # if random.randint(0, 100) % 2 == 0:
            #     imgDir1 = imgDir;
            # else:
            #     imgDir1 = "./data/Set12/09.png";
            imgDir1 = imgDir;
            logger.info("读取图片：" + imgDir1);
            img = cv2.imread(imgDir1)[:, :, 0]
            img = img / 255;
            # 生成随机的噪声强度
            sigma = noiseSpec;
            cnt += 1;
            logger.info("第{:d}次样本抽取，噪声强度：{:.4f}".format(cnt, sigma));
            # 根据噪声强度生成噪声图片
            imgn = addPulseNoise_SVM(img, sigma);
            # 把这张图片的噪声计算出来
            feature = caculateFeature_cuda(imgn);  # size*size*featureDim
            # 随机选sampleNumPerImg个不同的点作为训练集样本
            randomList = random.sample(range(0, img.shape[0] * img.shape[1] - 1), sampleNumPerImg);
            for pos in randomList:
                x = int(pos / img.shape[1]);
                y = pos % img.shape[1];
                # 判定该点是否为随机脉冲点
                if abs(imgn[x, y] - img[x, y]) > opt.threasholdVal / 255:
                    # 1代表是随机脉冲，-1代表不是
                    label[curSampleSize] = 1;
                else:
                    label[curSampleSize] = -1;
                data[curSampleSize, :] = feature[x, y, :];
                curSampleSize += 1;
                # 样本足够了
                if curSampleSize >= setSize:
                    logger.info("训练集样本生成完毕");
                    break;
    f["normalizationMark"] = opt.normalizationMark;
    if opt.normalizationMark:
        normalization(data,normalizationPara,logger);
        logger.info("保存归一化参数");
        f["max"] = normalizationPara[0];
        f["min"] = normalizationPara[1];
    logger.info("保存训练集数据");
    # for i in range(0, len(data[0])):
    #     fname = "feature_{:d}"+str(i+1);
    #     f[fname] = data[:, i];
    f["name"] = data;
    f["label"] = label;
    f["featureDim"] = opt.featureDim;
    f["beizhu"] =opt.createNewDataFileBeiZhu;
    logger.info("");
    logger.info("");
    logger.info("");
    logger.info("");
    f.close();

#通过贝叶斯优化找到最好的SVM参数和训练样本数量
# 输入noise为针对的噪声强度，-1为忙降噪
def bayesOption(noise):
    opt.noiseSpec = noise;
    if not os.path.exists(opt.saveDir):
        os.mkdir(opt.saveDir);
    # ----------------------------------------
    # 配置logger模块,不同的参数对于不同名字的trainLog文件
    # ----------------------------------------
    if opt.noiseSpec<0:
        logger_name = "bayesOption_noise_blind";
    else:
        logger_name = "bayesOption_noise_"+str(opt.noiseSpec);
    if not os.path.exists(opt.saveDir+"bayesOptimal"):
        os.mkdir(opt.saveDir+"bayesOptimal");
    utils_logger.logger_info(logger_name, os.path.join(opt.saveDir+'bayesOptimal/', logger_name + '.log'));
    logger = logging.getLogger(logger_name);
    logger.info("其他参数设置");
    logger.info("trainImg:"+str(opt.trainImg));
    logger.info("testSetSize:"+str(opt.testSetSize));
    logger.info("testImg:"+str(opt.testImg));
    logger.info("sampleNumPerImg:"+str(opt.sampleNumPerImg));
    logger.info("threasholdVal:"+str(opt.threasholdVal));
    logger.info("noiseMin:"+str(opt.noiseMin));
    logger.info("noiseMax:"+str(opt.noiseMax));
    logger.info("noiseSpec:"+str(opt.noiseSpec));
    cnt = [1];
    acuracyList = [0];
    detectAcuracyList = [0];
    # 贝叶斯优化，优化4种参数，训练集样本数目，惩罚系数C，核函数，gamma参数
    # 测试集样本统一采用3000
    # 脉冲噪声阈值：5
    # 噪声上下限：0-0.8
    # 总之与全局参数保持一致
    #kernel = 1,表示kernel='linear'时，为线性核，C越大分类效果越好，但有可能会过拟合（defaul C=1）
    #kernel = 2,表示kernel='rbf'时（default），为高斯核，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。
    def train_evaluate(opt_para):
        # para_trainSetSize, para_C, para_kernel, para_gamma
        para_kernel = opt_para["para_kernel"];
        para_C = opt_para["para_C"];
        para_gamma = opt_para["para_gamma"];
        if para_kernel == 1:
            para_kernel = "linear";
        else:
            para_kernel = "rbf";
        logger.info("第"+str(cnt[0])+"次bayes优化************************************************************************************");
        cnt[0] = cnt[0]+1;

        logger.info("para_C:"+str(para_C)+",para_kernel:"+str(para_kernel)+",para_gamma:"+str(para_gamma))
        # 训练集样本生成
        # 读取训练图片
        trainImg = cv2.imread(opt.trainImg)[:, :, 0];
        trainImg = trainImg / 255;
        # 获取训练集样本
        train_data,train_label = getTrainData(logger);
        # 开始训练模型
        classifier = svm.SVC(C=para_C, kernel=para_kernel, gamma=para_gamma,
                             decision_function_shape='ovo')  # 一对多
        logger.info("模型训练...");
        classifier.fit(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先
        # 读取测试图片
        testImg = cv2.imread(opt.testImg)[:, :, 0];
        testImg = testImg / 255;
        # 获取测试集样本
        logger.info("测试集样本获取...");
        test_data, test_label = generateSample(testImg, opt.testSetSize, opt.sampleNumPerImg, opt.noiseMin,
                                               opt.noiseMax,opt.noiseSpec);
        if opt.normalizationMark:
            test_data = normalization(test_data,normalizationPara,logger);

        # ××××××××××××××××××××××××××××××
        # 计算svc分类器的准确率
        # ××××××××××××××××××××××××××××××
        # logger.info("训练集验证");
        # tra_label = classifier.predict(train_data)  # 训练集的预测标签
        # trainNoiseCnt = 0;  # 训练集有多少个噪声
        # trainNoiseDetCnt = 0;  # 训练集被检测出了多少个噪声点
        # trainNoiseToalCnt = 0;  # 训练集共检测出多少噪声
        # for i in range(0, opt.trainSetSize):
        #     if train_label[i] == 1:
        #         trainNoiseCnt += 1;
        #         if (tra_label[i] == 1):
        #             trainNoiseDetCnt += 1;
        #     if (tra_label[i] == 1):
        #         trainNoiseToalCnt += 1;
        # logger.info("训练集中噪声数目：" + str(trainNoiseCnt));
        # logger.info("训练集中噪声被检测出：" + str(trainNoiseDetCnt));
        # logger.info("训练集中噪声共检测出：" + str(trainNoiseToalCnt));
        # logger.info("--------------------------------------------------------------------------------------");

        test_lable = test_label.ravel();
        accuracy = classifier.score(test_data, test_lable);
        logger.info("测试集验证");
        tes_label = classifier.predict(test_data)  # 测试集的预测标签
        testNoiseCnt = 0;  # 测试集有多少个噪声
        testNoiseDetCnt = 0;  # 测试集被检测出了多少个噪声点
        testNoiseToalCnt = 0;  # 测试集共检测出多少噪声
        for i in range(0, opt.testSetSize):
            if test_lable[i] == 1:
                testNoiseCnt += 1;
                if (tes_label[i] == 1):
                    testNoiseDetCnt += 1;
            if (tes_label[i] == 1):
                testNoiseToalCnt += 1;
        logger.info("测试集中噪声数目：" + str(testNoiseCnt));
        logger.info("测试集中噪声被检测出：" + str(testNoiseDetCnt));
        logger.info("测试集中噪声检出率：" + str(testNoiseDetCnt/testNoiseCnt));
        detectAcuracyList[0] += testNoiseDetCnt/testNoiseCnt;
        detectAcuracyList.append(testNoiseDetCnt/testNoiseCnt);
        logger.info("测试集中噪声共检测出：" + str(testNoiseToalCnt));
        logger.info("测试集精度：" + str(accuracy)+",损失函数："+str(1-accuracy));
        acuracyList[0] += accuracy;
        acuracyList.append(accuracy);
        # return 1-testNoiseDetCnt/testNoiseCnt;
        return 1-accuracy;

    best_parameters, values, experiment, model = optimize(
        parameters=[
            {
                "name": "para_C",
                "type": "range",
                "bounds": [0.01, 100],
                "value_type": "float"
            },
            {
                "name": "para_kernel",
                "type": "range",
                "bounds": [1, 2],
                "value_type": "int"
            },
            {
                "name": "para_gamma",
                "type": "range",
                "bounds": [0.01, 100],
                "value_type": "float"
            },
        ],
        evaluation_function=train_evaluate,
        objective_name='lossVal',
        total_trials=25,
        minimize=True,
    );
    logger.info("××××××××××××××××××××××××××××××××××××××××××××××××××××××××××");
    logger.info("测试集中噪声被检测出(平均值)：" + str(detectAcuracyList[0]/25));
    logger.info("测试集精度(平均值)：" + str(acuracyList[0]/25));
    logger.info("测试集中噪声被检测出的所有情况：" + str(detectAcuracyList));
    logger.info("测试集精度所有情况：" + str(acuracyList));
    logger.info(best_parameters)
    means, covariances = values
    logger.info(means)
    para_kernel = best_parameters['para_kernel'];
    if para_kernel==1:
        para_kernel = 'linear';
    else:
        para_kernel = 'rbf';
    return best_parameters['para_C'],best_parameters['para_gamma'],para_kernel;

# 输入noise为针对的噪声强度，-1为忙降噪
# cnt:测试多少次
# svmModel:传入模型，不需要自己去训练出来，浪费时间
def main(svmModelSaveDir,noise,cnt):
    opt.noiseSpec = noise;
    logger_name = "modelAccuracy_noise_"+str(noise);
    if not os.path.exists(opt.saveDir):
        os.mkdir(opt.saveDir);
    if opt.noiseSpec<0:
        modelDir = opt.saveDir+"modelBlind/";
    else:
        modelDir = opt.saveDir+"model_"+str(opt.noiseSpec)+"/";
    if not os.path.exists(modelDir):
        os.mkdir(modelDir);
    utils_logger.logger_info(logger_name, os.path.join(modelDir, logger_name + '.log'));
    logger = logging.getLogger(logger_name);
    logger.info("模型训练参数设置");
    logger.info("trainImg:" + str(opt.trainImg));
    logger.info("trainSetSize:" + str(opt.trainSetSize));
    logger.info("testSetSize:" + str(opt.testSetSize));
    logger.info("testImg:" + str(opt.testImg));
    logger.info("sampleNumPerImg:" + str(opt.sampleNumPerImg));
    logger.info("noiseMin:" + str(opt.noiseMin));
    logger.info("noiseMax:" + str(opt.noiseMax));
    logger.info("noiseSpec:" + str(opt.noiseSpec));
    logger.info("threasholdVal:"+str(opt.threasholdVal));
    logger.info("svmPara_C:"+str(opt.para_C));
    logger.info("svmPara_gamma:"+str(opt.para_gamma));
    logger.info("svmPara_kernel:"+str(opt.para_kernel));
    logger.info("--------------------------------------------------------------------------------------");
    detectAcuracyList = []
    acuracyList = []
    SVMTrainAndSave(modelSaveDir,logger);
    # 加载SVM模型参数
    svm_cuda.loadSvmPara(svmModelSaveDir,logger);
    for i in range(0,cnt):
        # 遍历所有测试图片进行测试
        imageDir = "./data/Set12";
        ls = os.listdir(imageDir);
        ls.sort();
        cnt = 0;
        # allowCnt = [9];
        for i in ls:
            # cnt += 1;
            # if not cnt in allowCnt:
            #     continue;
            saveDir = modelDir;
            image_path = os.path.join(imageDir, i);  # 得到图片地址
            # 加载图片
            img = cv2.imread(image_path)[:, :, 0]
            img = img / 255;
            img = np.expand_dims(img, 0)
            img = np.expand_dims(img, 1)
            # 为img添加普通随机脉冲噪声 img:1*1*size*size
            logger.info("添加随机脉冲噪声");
            imgn, imgn_threasholdSalt = addPulseNoiseAndThreasholdSalt_SVM(img[0, 0, :, :], noise, opt.threasholdVal);  # 生成特定强度的脉冲噪声
            # 得到使用svm处理的盐噪声  imgn:size*size
            logger.info("SVM预处理图片：" + image_path + "******************************************");
            imgn_svm = svmDeal(imgn,logger);
            # 此时diffImg,0:正确  <0:漏判  >0:误判
            diffImg = imgn_threasholdSalt - imgn_svm;
            # 求解精度
            totalPoint = img[0, 0, :, :].shape[0] * img[0, 0, :, :].shape[1];  # 总共的点数
            correctPoint = np.sum(diffImg == 0);  # 检测正确的个数
            noisePoint = np.sum(imgn_threasholdSalt != imgn);  # 真实threasholdSalt噪声的个数
            missPoint = np.sum(diffImg < 0);  # 漏判的个数
            errorPoint = np.sum(diffImg > 0);  # 误判的个数
            logger.info("总点数:"+"{:d}".format(totalPoint));
            logger.info("正确检测点数:"+"{:d}".format(correctPoint));
            logger.info("噪声占比:" + "{:.4f}".format(noisePoint / totalPoint));
            logger.info("漏判个数:"+"{:d}".format(missPoint));
            logger.info("误判个数:"+"{:d}".format(errorPoint));
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
            plt.subplot(1, 3, 1);
            plt.imshow(imgn, cmap='gray');
            # plt.title("noise img,noise ratio:" + "{:.4f}".format(noisePoint / totalPoint));
            # plt.subplot(1,3,2);
            # noiseImg = np.where(imgn_threasholdSalt==0,255,0);
            # plt.imshow(noiseImg.astype(np.uint8),cmap='gray');
            # plt.title("threashold pulse noise");
            plt.subplot(1, 3, 2);
            plt.imshow(errorImg.astype(np.uint8));
            # plt.title("error img,predict accuracy:" + "{:.4f}".format(
            #     correctPoint / totalPoint) + ",detect ratio:" + "{:.4f}".format(1 - missPoint / noisePoint));
            # SVM处理后得到的输出结果
            plt.subplot(1, 3, 3);
            plt.imshow(imgn_svm, cmap='gray');
            # plt.title("SVM img");
            # 变为tensor格式
            # ISource = torch.Tensor(Img)
            # 保存SVM的预测精度图片
            svmResultSaveDir = saveDir + "svmErrorResult/";
            if (not os.path.exists(svmResultSaveDir)):
                os.makedirs(svmResultSaveDir);
            plt.savefig(svmResultSaveDir + "noise_" + str(noise) + "_" + i);
    # 保存该SVM模型
    # joblib.dump(saveDir, 'svmResultBalance/model_blind/SVM.model');

    # 读取离线存储的模型
    # classifier1 = joblib.load('./svmResult/model/SVM.model');

# 根据现有参数训练SVM模型并保存
# 保存位置在model_0.3里面
def SVMTrainAndSave(modelSaveDir,logger):
    # 获取训练集样本
    train_data, train_label = getTrainData(logger);
    # 开始训练模型
    svm_cuda.trainPrint(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先
    # 保存该SVM模型
    svm_cuda.saveModelPara(modelSaveDir,logger);
    # joblib.dump(svmModel, modelSaveDir+"/SVM.model");

#加载SVM模型
def loadSvmModel(path):
    # ＃加载模型
    classifier = joblib.load(path);
    return classifier;

import os
if __name__ == '__main__':
    envpath = '/home/p/anaconda3/lib/python3.8/site-packages/cv2/qt/plugins/platforms'
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
    if opt.createNewDataFileMark:
        trainDataInit(opt.trainImg, opt.trainSetSize, opt.sampleNumPerImg, opt.noiseMin, opt.noiseMax, opt.noiseSpec, opt.trainDataSaveDir, opt.createNewDataFileName);
    if opt.isTestMode:
        testFeatureIndex_wholeImg();
    else:
        #***********************************************
        # 针对特定的噪声，使用贝叶斯找到最优的SVM模型参数和训练集数据大小，然后建立模型并保存
        # 保存位置是./svmResult/....
        #***********************************************
        # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        # para_C,para_gamma,para_kernel = bayesOption(-1);
        # for noise in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9]:
        # for noise in [0.1,0.2,0.3,0.4,0.5,0.6]:
        for noise in [0.05,0.1,0.2,0.3,0.4,0.5,0.6]:
            if not os.path.exists(opt.saveDir):
                os.mkdir(opt.saveDir);
            modelSaveDir = opt.saveDir + "model_{:.1f}/".format(noise);
            if not os.path.exists(modelSaveDir):
                os.mkdir(modelSaveDir);
            main(modelSaveDir,noise,1);

    # mat = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]]);
    # for i in range(0,5):
    #     for j in range(0,5):
    #         [mat1,clearmat] = getTarget5_5Matrix(mat,i,j);
    #         print("ja");

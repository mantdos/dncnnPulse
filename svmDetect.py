# -*- coding:utf-8 -*-
import cv2
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
import matplotlib
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
from add_noise import addPulseNoise_SVM
import random

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

# define converts(字典)
def Iris_label(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]

def refernce():
    # 1.读取数据集
    path = u'iris/iris.data'  # 数据文件路径
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: Iris_label})
    # converters={4:Iris_label}中“4”指的是第5列：将第五列的str转化为label(number)
    # print(data.shape)

    # 2.划分数据与标签
    x, y = np.split(data, indices_or_sections=(4,), axis=1)  # x为数据，y为标签
    x = x[:, 0:2]
    train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=0.6,
                                                                      test_size=0.4)  # sklearn.model_selection.
    # print(train_data.shape)

    # 3.训练svm分类器
    classifier = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo')  # 一对多
    classifier.fit(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先

    # 4.计算svc分类器的准确率
    print("训练集：", classifier.score(train_data, train_label))
    print("测试集：", classifier.score(test_data, test_label))

    # 也可直接调用accuracy_score方法计算准确率
    from sklearn.metrics import accuracy_score

    tra_label = classifier.predict(train_data)  # 训练集的预测标签
    tes_label = classifier.predict(test_data)  # 测试集的预测标签
    print("训练集：", accuracy_score(train_label, tra_label))
    print("测试集：", accuracy_score(test_label, tes_label))

    # 查看决策函数
    print('train_decision_function:\n', classifier.decision_function(train_data))
    print('predict_result:\n', classifier.predict(train_data))

    # 5.绘制图形
    # 确定坐标轴范围
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0维特征的范围
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1维特征的范围
    x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网络采样点
    grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
    # 指定默认字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    # 设置颜色
    cm_light = matplotlib.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = matplotlib.colors.ListedColormap(['g', 'r', 'b'])

    grid_hat = classifier.predict(grid_test)  # 预测分类值
    grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)  # 预测值得显示
    plt.scatter(x[:, 0], x[:, 1], c=y[:, 0], s=30, cmap=cm_dark)  # 样本
    plt.scatter(test_data[:, 0], test_data[:, 1], c=test_label[:, 0], s=30, edgecolors='k', zorder=2,
                cmap=cm_dark)  # 圈中测试集样本点
    plt.xlabel(u"花萼长度", fontsize=13)
    plt.ylabel(u"花萼宽度", fontsize=13)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(u"鸢尾花SVM二特征分类")
    plt.show()

#classifier = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo')  # 一对多
parser = argparse.ArgumentParser(description="SVM")
parser.add_argument("--trainSetSize", type=int, default=1000, help='训练集样本总数量')
parser.add_argument("--trainImg", type=str, default="./data/Set12/08.png", help='训练集使用的图片')
parser.add_argument("--testSetSize", type=int, default=2000, help='测试集样本总数量')
parser.add_argument("--testImg", type=str, default="./data/Set12/08.png", help='测试集使用的图片')
parser.add_argument("--para_C", type=float, default=10, help="惩罚系数")
parser.add_argument("--para_kernel", type=str, default='rbf', help="SVM核函数")
parser.add_argument("--para_gamma", type=float, default=10, help="SVMgamma系数")
parser.add_argument("--threasholdVal", type=int, default=5, help="在训练集设置时用于判断当前像素是否为impulse噪声点")
parser.add_argument("--sampleNumPerImg", type=int, default=200, help="每种噪声强度下选择多少个样本")
parser.add_argument("--noiseMin", type=float, default=0.2, help='噪声强度下限')
parser.add_argument("--noiseMax", type=float, default=0.8, help='噪声强度上限')
parser.add_argument("--noiseSpec", type=float, default=-1, help='是否采用固定噪声，不采用就=-1')
opt = parser.parse_args()

#*******************************
# 生成用于SVM训练或验证的脉冲噪声特征和标签
# 像素点的特征值共3个，分别是GD,ABD,ACD
# img:经过归一化后的原始图片
# setSize:生成的样本数量
# sampleNumPerImg：每种噪声强度下选择多少个样本
# noiseMin、noiseMax:生成的噪声上下限
# noiseSpec:是否固定图片的噪声强度，为-1表示不固定
# 返回n×3和n×1的数据和标签
#*******************************
def generateSample(img,setSize,sampleNumPerImg,noiseMin,noiseMax,noiseSpec):
    #定义特征矩阵[n×m]和标签矩阵[n×1]，n：样本个数  m：特征维度
    data = np.zeros((setSize,3));
    label = np.zeros((setSize,1));
    #当前样本数量为0
    curSampleSize = 0;
    while curSampleSize<setSize:
        #生成随机的噪声强度
        sigma = random.uniform(noiseMin,noiseMax);
        if not noiseSpec < 0:#说明使用固定的噪声强度
            sigma = noiseSpec;
        #根据噪声强度生成噪声图片
        imgn = addPulseNoise_SVM(img,sigma);
        #随机选opt.sampleNumPerImg个不同的点作为训练集样本
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
            #计算该点的各个特征值，共4个，分别是GD,ABD,ACD
            GD = 0;#灰度值差值
            ABD = 0;#平均背景差值
            average = 0;#周围点的均值
            ACD = 0;#复杂度积累差值
            # 给出(x,y)点周围5×5的矩阵值.patchVal[2,2]为中心点,如果没超过界就都算上，否则是0
            patchVal = np.zeros((5,5));
            for i in range(-3, 3):
                for j in range(-3, 3):
                    if x+i>=0 and x+i<img.shape[0] and y+j>=0 and y+j<img.shape[1]:
                        patchVal[i+2,j+2] = imgn[x+i,y+j];
            for i in range(-1,2):
                for j in range(-1,2):
                    if not (i==0 and j==0):
                        GD += abs(patchVal[2,2]-patchVal[2+i,2+j]);
                        average += patchVal[2+i,2+j];
                    #计算每个点的复杂度误差
                    ACDVal = 4*patchVal[2+i,2+j];
                    ACDVal -= patchVal[2+i-1,2+j];
                    ACDVal -= patchVal[2+i+1,2+j];
                    ACDVal -= patchVal[2+i,2+j-1];
                    ACDVal -= patchVal[2+i,2+j+1];
                    ACD += abs(ACDVal);
            ABD = abs(patchVal[2,2] - average/8.0);
            data[curSampleSize,0] = GD;
            data[curSampleSize,1] = ABD;
            data[curSampleSize,2] = ACD;
            curSampleSize += 1;
            #样本足够了
            if curSampleSize>=setSize:
                print("共产生："+str(setSize)+"个样本");
                break;
    return data,label;
    # print(data);
    # print(lable);

def main():
    # 训练集样本生成
    #读取训练图片
    trainImg = cv2.imread(opt.trainImg)[:,:,0];
    trainImg = trainImg/255;
    #获取训练集样本
    train_data,train_label = generateSample(trainImg, opt.trainSetSize, opt.sampleNumPerImg, opt.noiseMin, opt.noiseMax, opt.noiseSpec);
    #开始训练模型
    classifier = svm.SVC(C=opt.para_C, kernel=opt.para_kernel, gamma=opt.para_gamma, decision_function_shape='ovo')  # 一对多
    classifier.fit(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先
    # 读取测试图片
    testImg = cv2.imread(opt.testImg)[:, :, 0];
    testImg = testImg / 255;
    # 获取测试集样本
    test_data,test_label = generateSample(testImg, opt.testSetSize, opt.sampleNumPerImg, opt.noiseMin, opt.noiseMax, opt.noiseSpec);

    #××××××××××××××××××××××××××××××
    #计算svc分类器的准确率
    #××××××××××××××××××××××××××××××
    train_lable = train_label.ravel();
    print("训练集精度：", classifier.score(train_data, train_lable))
    tra_label = classifier.predict(train_data)  # 训练集的预测标签
    trainNoiseCnt = 0;#训练集有多少个噪声
    trainNoiseDetCnt = 0;#训练集被检测出了多少个噪声点
    trainNoiseToalCnt = 0; #训练集共检测出多少噪声
    for i in range(0,opt.trainSetSize):
        if train_lable[i] == 1:
            trainNoiseCnt += 1;
            if(tra_label[i] == 1):
                trainNoiseDetCnt += 1;
        if(tra_label[i] == 1):
            trainNoiseToalCnt += 1;
    print("训练集中噪声数目："+str(trainNoiseCnt));
    print("训练集中噪声被检测出："+str(trainNoiseDetCnt));
    print("训练集中噪声共检测出："+str(trainNoiseToalCnt));

    test_lable = test_label.ravel();
    print("测试集精度：", classifier.score(test_data, test_lable))
    tes_label = classifier.predict(test_data)  # 测试集的预测标签
    testNoiseCnt = 0;#测试集有多少个噪声
    testNoiseDetCnt = 0;#测试集被检测出了多少个噪声点
    testNoiseToalCnt = 0; #测试集共检测出多少噪声
    for i in range(0,opt.testSetSize):
        if test_lable[i] == 1:
            testNoiseCnt += 1;
            if(tes_label[i] == 1):
                testNoiseDetCnt += 1;
        if(tes_label[i] == 1):
            testNoiseToalCnt += 1;
    print("测试集中噪声数目："+str(testNoiseCnt));
    print("测试集中噪声被检测出："+str(testNoiseDetCnt));
    print("测试集中噪声共检测出："+str(testNoiseToalCnt));



#通过贝叶斯优化找到最好的SVM参数和训练样本数量
def bayesOption():
    # ----------------------------------------
    # 配置logger模块,不同的参数对于不同名字的trainLog文件
    # ----------------------------------------
    logger_name = "bayesOption_noise_blindGai";
    if not os.path.exists("./svmTest"):
        os.mkdir("./svmTest");
    utils_logger.logger_info(logger_name, os.path.join('./svmTest/', logger_name + '.log'))
    logger = logging.getLogger(logger_name);
    logger.info("其他参数设置");
    logger.info("trainImg:"+str(opt.trainImg));
    logger.info("testSetSize:"+str(opt.testSetSize));
    logger.info("testImg:"+str(opt.testImg));
    logger.info("sampleNumPerImg:"+str(opt.sampleNumPerImg));
    logger.info("noiseMin:"+str(opt.noiseMin));
    logger.info("noiseMax:"+str(opt.noiseMax));
    logger.info("noiseSpec:"+str(opt.noiseSpec));
    cnt = [1];
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
        para_trainSetSize = opt_para["para_trainSetSize"];
        para_C = opt_para["para_C"];
        para_gamma = opt_para["para_gamma"];
        if para_kernel == 1:
            para_kernel = "linear";
        else:
            para_kernel = "rbf";
        logger.info("第"+str(cnt[0])+"次bayes优化************************************************************************************");
        cnt[0] = cnt[0]+1;

        logger.info("训练集数量:"+str(para_trainSetSize)+",para_C:"+str(para_C)+",para_kernel:"+str(para_kernel)+",para_gamma:"+str(para_gamma))
        # 训练集样本生成
        # 读取训练图片
        trainImg = cv2.imread(opt.trainImg)[:, :, 0];
        trainImg = trainImg / 255;
        # 获取训练集样本
        train_data, train_label = generateSample(trainImg, para_trainSetSize, opt.sampleNumPerImg, opt.noiseMin,
                                                 opt.noiseMax, opt.noiseSpec);
        # 开始训练模型
        classifier = svm.SVC(C=para_C, kernel=para_kernel, gamma=para_gamma,
                             decision_function_shape='ovo')  # 一对多
        classifier.fit(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先
        # 读取测试图片
        testImg = cv2.imread(opt.testImg)[:, :, 0];
        testImg = testImg / 255;
        # 获取测试集样本
        test_data, test_label = generateSample(testImg, opt.testSetSize, opt.sampleNumPerImg, opt.noiseMin,
                                               opt.noiseMax,opt.noiseSpec);

        # ××××××××××××××××××××××××××××××
        # 计算svc分类器的准确率
        # ××××××××××××××××××××××××××××××
        train_lable = train_label.ravel();
        tra_label = classifier.predict(train_data)  # 训练集的预测标签
        trainNoiseCnt = 0;  # 训练集有多少个噪声
        trainNoiseDetCnt = 0;  # 训练集被检测出了多少个噪声点
        trainNoiseToalCnt = 0;  # 训练集共检测出多少噪声
        for i in range(0, para_trainSetSize):
            if train_lable[i] == 1:
                trainNoiseCnt += 1;
                if (tra_label[i] == 1):
                    trainNoiseDetCnt += 1;
            if (tra_label[i] == 1):
                trainNoiseToalCnt += 1;
        logger.info("训练集中噪声数目：" + str(trainNoiseCnt));
        logger.info("训练集中噪声被检测出：" + str(trainNoiseDetCnt));
        logger.info("训练集中噪声共检测出：" + str(trainNoiseToalCnt));
        logger.info("训练集精度：" + str(classifier.score(train_data, train_lable)));

        logger.info("--------------------------------------------------------------------------------------");

        test_lable = test_label.ravel();
        accuracy = classifier.score(test_data, test_lable);
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
        logger.info("测试集中噪声共检测出：" + str(testNoiseToalCnt));
        logger.info("测试集精度：" + str(accuracy)+",损失函数："+str(1-accuracy));
        return 1 - accuracy;

    best_parameters, values, experiment, model = optimize(
        parameters=[
            {
                "name": "para_trainSetSize",
                "type": "range",
                "bounds": [1000, 5000],
                "value_type": "int"
            },
            {
                "name": "para_C",
                "type": "range",
                "bounds": [1, 300],
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
                "bounds": [1, 300],
                "value_type": "float"
            },
        ],
        evaluation_function=train_evaluate,
        objective_name='lossVal',
        total_trials=50,
        minimize=True,
    );

    print(best_parameters)
    means, covariances = values
    print(means)

import os
if __name__ == '__main__':
    envpath = '/home/p/anaconda3/lib/python3.8/site-packages/cv2/qt/plugins/platforms'
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
    # main();
    bayesOption();
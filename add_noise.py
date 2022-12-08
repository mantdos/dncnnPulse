import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
#**************************************************************************************
# 用于为图片添加各种噪声：
# 包括为dncnn的128*1*size*size或1*1*size*size的矩阵添加脉冲噪声、盐噪声、经过threashold处理后的盐噪声
# 同时包括为SVM添加size*size的图片矩阵添加脉冲噪声程序
#**************************************************************************************

import os

# 辅助添加脉冲噪声程序,灰度图像
import torch

#生成和脉冲噪声相关的矩阵
#out1:生成0的个数等于sigma的用于消除原始图片的矩阵
#out2:生成含有sigma个[0-1]随机值的填充矩阵
#用于DNCNN训练和测试使用
def creatPulseNoiseMat(size,sigma):
    # if sigma>0.9: # 说明是盲降噪，直接生成[0.1-0.6的随机噪声]
    #     # 根据大小生成对应尺寸的随机矩阵
    #     clearMatrix = np.random.rand(size[0], size[1], size[2], size[3]);
    #     # 对128个patch内所有图片，随机生成不同的噪声强度的clearMat
    #     for cnt in range(0, size[0]):
    #         # 生成随机的噪声强度
    #         sigma = random.uniform(0.1, 0.6);
    #         s = random.uniform(1, 1000);
    #         # 获得其中一张图片
    #         clearMat = clearMatrix[cnt, 0, :, :];
    #         clearMat = np.where(clearMat > sigma, 1, 0);
    #         z = np.sum(clearMat == 0);
    #         if s>997:
    #             print("盲噪声生成第{:d}张图片，噪声={:.2f},真实强度={:.2f}".format(cnt+1,sigma,z/(size[2]*size[3])));
    # else:
    # 根据大小生成对应尺寸的随机矩阵
    clearMatrix=np.random.rand(size[0],size[1],size[2],size[3]);
    # 根据sigma将大于sigma的置1，用于和原始矩阵相乘,消除原始矩阵对应位置的数目
    clearMatrix=np.where(clearMatrix>sigma,1,0);
    #对128个patch内所有图片，循环直到噪声强度精度一致
    for cnt in range(0,size[0]):
        #获得其中一张图片
        clearMat = clearMatrix[cnt,0,:,:];
        #保证噪声强度精度
        diff = np.sum(clearMat==0) - size[2]*size[3]*sigma;
        #精度阈值，允许0.05%的误差
        threshold = size[2]*size[3]*0.005;
        if diff < 0: #说明0的个数少了
            while diff< -threshold:#3个以内都算成功
                x = np.random.randint(0, size[2] - 1);
                y = np.random.randint(0, size[3] - 1);
                if(clearMat[x,y]==1):
                    clearMat[x,y] = 0;
                    diff += 1;
        else: #说明0的个数多了
            while diff> threshold:
                x = np.random.randint(0, size[2] - 1);
                y = np.random.randint(0, size[3] - 1);
                if(clearMat[x,y]==0):
                    clearMat[x,y] = 1;
                    diff -= 1;
        z=np.sum(clearMat==0)
        #print(z)
#生成用于和噪声相乘的矩阵,就是clearMat的去反
    mutMatrix = np.where(clearMatrix==0,1,0);
    noiseMatrix = np.random.rand(size[0],size[1],size[2],size[3]) * mutMatrix;
    return clearMatrix,noiseMatrix;

#生成和脉冲噪声相关的矩阵
#out1:生成0的个数等于sigma的用于消除原始图片的矩阵
#out2:生成含有sigma个[0-1]随机值的填充矩阵
#用于SVM训练和测试使用
def creatPulseNoiseMat_SVM(size,sigma):
    # 根据大小生成对应尺寸的随机矩阵
    clearMatrix=np.random.rand(size[0],size[1]);
    # 根据sigma将大于sigma的置1，用于和原始矩阵相乘,消除原始矩阵对应位置的数目
    clearMatrix=np.where(clearMatrix>sigma,1,0);
    #保证噪声强度精度
    diff = np.sum(clearMatrix==0) - size[0]*size[1]*sigma;
    #精度阈值，允许待分解次数0.05%的误差
    threshold = size[0]*size[1]*sigma*0.005;
    if diff < 0: #说明0的个数少了
        while diff< -threshold:#3个以内都算成功
            x = np.random.randint(0, size[0] - 1);
            y = np.random.randint(0, size[1] - 1);
            if(clearMatrix[x,y]==1):
                clearMatrix[x,y] = 0;
                diff += 1;
    else: #说明0的个数多了
        while diff> threshold:
            x = np.random.randint(0, size[0] - 1);
            y = np.random.randint(0, size[1] - 1);
            if(clearMatrix[x,y]==0):
                clearMatrix[x,y] = 1;
                diff -= 1;
    # z=np.sum(clearMatrix==0)/size[0]/size[1];
    # print("原始脉冲噪声强度:"+str(sigma)+"生成的脉冲噪声强度："+str(z));
    #生成用于和噪声相乘的矩阵,就是clearMat的去反
    mutMatrix = np.where(clearMatrix==0,1,0);
    noiseMatrix = np.random.rand(size[0],size[1]) * mutMatrix;
    return clearMatrix,noiseMatrix;

# 辅助添加脉冲噪声程序
def addPulseNoise_dncnn(image,sigma):
    size=image.size();#128,1,×，×
    clearMat,noiseMat = creatPulseNoiseMat(size, sigma);
    image = image.numpy();
    noiseMat = noiseMat + (clearMat*image);
    noiseMat = torch.tensor(noiseMat);
    noiseMat=noiseMat.type(torch.FloatTensor)
    return noiseMat;

# 辅助添加脉冲噪声程序,同时把对应的threasholdSalt噪声输出出去
def addPulseNoiseAndThreasholdSalt_SVM(image,sigma,threashold):
    # size=image.shape;#×，×
    # clearMat,noiseMat = creatPulseNoiseMat_SVM(size, sigma);
    # noiseMat = noiseMat + (clearMat*image);
    # return noiseMat;
    threashold = threashold / 255;
    size=image.shape;#×，×
    clearMat,noiseMat = creatPulseNoiseMat_SVM(size, sigma);
    realNoiseMat = noiseMat + (clearMat*image);#这个是真实的噪声
    # 先得到原始的image在clearMat这些位置的值
    noiseImage = np.where(clearMat==0,1,0)*image;
    # 和噪声做比较，如果差值<于threashold就置为0,否则当作噪声，变为1
    diffMat = abs(noiseMat-noiseImage);
    diffMat = np.where(diffMat<threashold,0,1);
    # 此时diffMat为1的位置代表真正需要使用噪声代替并且被替换为盐噪声的点
    clearMatNew = np.where(diffMat==1,0,1);
    #最后计算得到threashold噪声结果
    threasholdSaltNoiseMat = clearMatNew*(noiseMat + (clearMat*image));
    return realNoiseMat,threasholdSaltNoiseMat;

# 辅助添加盐噪声程序，其实盐噪声就是在clearMat的基础上直接和原始图片相乘就好了，把它变为0
def addSaltNoise_dncnn(image,sigma):
    size=image.size();#128,1,*,*
    clearMat,noiseMat = creatPulseNoiseMat(size, sigma);
    clearMat = torch.tensor(clearMat);
    clearMat = clearMat*image;
    clearMat=clearMat.type(torch.FloatTensor)
    return clearMat;

# 辅助添加带Threashold处理后的盐噪声程序
def addThreasholdSaltNoise_dncnn(image,sigma,threashold):
    threashold = threashold / 255;
    size=image.size();#128,1,×，×  or 1,1,×，×
    clearMat,noiseMat = creatPulseNoiseMat(size, sigma);
    # 先得到原始的image在clearMat这些位置的值
    image = image.numpy();
    noiseImage = np.where(clearMat==0,1,0)*image;
    # 和噪声做比较，如果差值<于threashold就置为0,否则当作噪声，变为1
    diffMat = abs(noiseMat-noiseImage);
    diffMat = np.where(diffMat<threashold,0,1);
    # 此时diffMat为1的位置代表真正需要使用噪声代替并且被替换为盐噪声的点
    clearMatNew = np.where(diffMat==1,0,1);
    #最后计算得到threashold噪声结果
    noiseMat = clearMatNew*(noiseMat + (clearMat*image));
    noiseMat = torch.tensor(noiseMat);
    noiseMat=noiseMat.type(torch.FloatTensor);
    return noiseMat;

# 辅助添加脉冲噪声程序
def addPulseNoise_SVM(image,sigma):
    size=image.shape;#×，×
    clearMat,noiseMat = creatPulseNoiseMat_SVM(size, sigma);
    noiseMat = noiseMat + (clearMat*image);
    return noiseMat;

if __name__ == '__main__':
    envpath = '/home/p/anaconda3/lib/python3.8/site-packages/cv2/qt/plugins/platforms'
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath

    # img = cv2.imread("./data/Set12/03.png")
    # size = img.shape();
    newImg = np.ones([128,1,40,40]);
    img = torch.Tensor(newImg);

    #添加脉冲噪声
    # imgnew = addSaltNoiseCor(img,0.2);

    # # 生成同样大小的空图
    # emptyImage = np.zeros(img.shape, np.uint8)
    # # 将灰度图片转换为彩色图片
    # emptyImage2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # cv2.imshow("Image", img)
    # cv2.imshow("EmptyImage", emptyImage)
    # cv2.imshow("EmptyImage2", emptyImage2)

    # 按一个按键后关闭所有图片
    # cv2.waitKey (0)
    # cv2.destroyAllWindows()

import numpy as np

#×××××××××××××××××××××××××××××××××××××××××××××
#生成和脉冲噪声相关的矩阵
#out1:生成0的个数等于sigma的用于消除原始图片的矩阵
#out2:生成含有sigma个[0-1]随机值的填充矩阵
#×××××××××××××××××××××××××××××××××××××××××××××
def creatPulseNoiseMat(size,sigma):
    # 根据大小生成对应尺寸的随机矩阵
    clearMat=np.random.rand(size[0],size[1]);
    # 根据sigma将大于sigma的置1，用于和原始矩阵相乘,消除原始矩阵对应位置的数目
    clearMat=np.where(clearMat>sigma,1,0);
    #循环直到噪声强度精度一致
    diff = np.sum(clearMat==0) - size[0]*size[1]*sigma;
    if diff < 0: #说明0的个数少了
        while diff<-3:#3个以内都算成功
            x = np.random.randint(0, size[0] - 1);
            y = np.random.randint(0, size[1] - 1);
            if(clearMat[x,y]==1):
                clearMat[x,y] = 0;
                diff += 1;
    else: #说明0的个数多了
        while diff>3:
            x = np.random.randint(0, size[0] - 1);
            y = np.random.randint(0, size[1] - 1);
            if(clearMat[x,y]==0):
                clearMat[x,y] = 1;
                diff -= 1;
    #生成用于和噪声相乘的矩阵,就是clearMat的去反
    mutMat = np.where(clearMat==0,1,0);
    noiseMat = np.random.rand(size[0],size[1]) * mutMat;
    z=np.sum(clearMat==1)
    print(z)
    return clearMat,noiseMat;

import os;
if __name__ == '__main__':
    envpath = '/home/p/anaconda3/lib/python3.8/site-packages/cv2/qt/plugins/platforms'
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
    size = [40, 40];
    sigma = 0.2;
    clearMat,noiseMat = creatPulseNoiseMat(size, sigma);

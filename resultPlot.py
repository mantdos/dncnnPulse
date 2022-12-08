import h5py
import numpy as np
import argparse
from shutil import copyfile
from shutil import copy
import os
from sys import exit
import logging
from utils1 import utils_logger
from utils1 import utils_image as util
from utils1 import utils_option as option

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="DnCNN")
# parser.add_argument("--h5FileHasOpen", type=bool, default=False, help='mark the .hr file is Opened or not')
opt = parser.parse_args()

def main():
    # 确定h5文件保存路径
    destPsnrDir = './result/dncnnPulseNoiseResult_spec';
    # 获得文件读句柄
    f = h5py.File(destPsnrDir + '/psnr.h5', 'r');
    # 分别绘制噪声强度为25、50、75下posnr随着epoach的变化曲线
    depthList = [17];
    noiseList = [0.3,1.3,2.3];
    cnt = 0;
    for depth in depthList:
        #创建一张图，并命名
        cnt+=1;
        plt.figure(num=cnt);
        plt.title('the line while noiseS = {:s}'.format(str(depth)),fontsize=15);
        for noiseS in noiseList:
            # 根据深度和噪声强度获取h5文件中的key值并读取出数据
            key = "/validPsnr_depth_"+ str(depth) + "&noise_" + str(noiseS);
            psnrList = f[(key)];
            # 根据psnrList生成X轴
            epoachCnt = len(psnrList);
            x = np.linspace(1, epoachCnt, epoachCnt);
            # 获得曲线的lable名称
            lableName = "depth_"+ str(depth) + "&noise_" + str(noiseS);
            plt.plot(x, psnrList,label=lableName, linewidth=1.0,);
            plt.xlabel("epoach",fontsize=12);
            plt.ylabel("Valid psnr",fontsize=12);
        plt.legend(loc='best')
    plt.show();
    f.close;

import os
if __name__ == '__main__':
    envpath = '/home/p/anaconda3/lib/python3.8/site-packages/cv2/qt/plugins/platforms'
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
    main()

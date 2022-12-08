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

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--useNotebook", type=bool, default=False, help='use my notebook to excute')
# parser.add_argument("--h5FileHasOpen", type=bool, default=False, help='mark the .hr file is Opened or not')
opt = parser.parse_args()

#**************************************************************************************
# 测试h5文件的读写
# 后来修改为了利用该文件修改psnr.h5的程序
#**************************************************************************************

def write():
    data = [13.92702035,15.55521765,14.37119118,14.9299175,18.67679851,16.70501993
,17.05559675,16.52280662,17.17584456,17.80297154,15.51446081,15.56251035
,17.13282659,17.04845699,17.10907716,16.37258538,17.26829982,18.13847166
,18.96342661,17.43422898,16.39703807,17.73618203,17.22603582,17.84005959
,16.41160183,18.92592003,17.35009304,17.59752294,17.90737251,17.12386932
,17.3901392,17.83878625,17.53675206,17.62243015,17.44626161,17.54825215
,17.59851105,17.546734,17.65914587,17.437905,17.46121364,17.27023754
,17.49141063,17.59211935,17.41088744,17.49008791,17.35134423,17.37879745
,17.56686265,17.57556652];
    destPsnrDir = './result/dncnnPulseNoiseResult_spec/';
    # 读取h5.psnr文件数据
    destPsnrFileDir = destPsnrDir + "psnr.h5";

    f = h5py.File(destPsnrFileDir,'a');
    # print(f['validPsnr_depth_17&noise_2.8'].value)  # 数据名称
    a = f.keys();
    if 'validPsnr_depth_17&noise_2.8' in a:
        del f['validPsnr_depth_17&noise_2.8'];
    f['validPsnr_depth_17&noise_2.8'] = data;
    print(f['validPsnr_depth_17&noise_2.8'][()])  # 数据名称
    del f['validPsnr_depth_17&noise_2.8'];
    f.close();

def read():
    destPsnrDir = './result/dncnnPulseNoiseResult_spec/';
    # 读取h5.psnr文件数据
    destPsnrFileDir = destPsnrDir + "psnr.h5";
    f = h5py.File(destPsnrFileDir, 'r');
    for key in f.keys():
        # logger.info('[psnr.h5 data] name:{:s}, data:{:s}'.format(f[key].name, str(f[key].value)));
        print(f[key].name)  # 数据名称
        # print(f[key].shape)  # 数据的大小
        str1 = str(f[key][()]);
        str1 = str1.replace('  ',',');
        str1 = str1.replace(' ',',');
        print(str1)  # 数据值

def main():
    write();
    # read();

if __name__ == "__main__":
    main()

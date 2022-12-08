import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import h5py
from sklearn.datasets import make_blobs

parser = argparse.ArgumentParser()
parser.add_argument("--c", type=float, default=0.01)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--batchsize", type=int, default=5)
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
args = parser.parse_args()
args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(args)

# svm的参数，第一个是权重n*1，第二个是偏移1
svmPara = [[],[]];

def train(x1, y1, logger, epoch=10, batchsize=10000):
    # 定义epoch和batchsize参数
    args.epoch = epoch;
    args.batchsize = batchsize;
    # 转换X，Y的格式
    X = torch.FloatTensor(x1)
    Y = torch.FloatTensor(y1)
    N = len(Y)
    # 定义模型
    featureDim = x1.shape[1];
    model = nn.Linear(featureDim, 1)
    model.to(args.device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(args.epoch):
        perm = torch.randperm(N)
        sum_loss = 0
        for i in range(0, N, args.batchsize):
            x = X[perm[i : i + args.batchsize]].to(args.device)
            y = Y[perm[i : i + args.batchsize]].to(args.device)
            # 用于测试直接通过ndarray的参数是否可以得到相同的结果
            # x2 = x1[perm[i : i + args.batchsize]];
            # y2 = y1[perm[i : i + args.batchsize]];
            # W = model.weight.squeeze().t().detach().cpu().numpy();
            # b = model.bias.squeeze().detach().cpu().numpy();
            # output1 = np.dot(x2,W)+b;

            optimizer.zero_grad()
            output = model(x).squeeze()
            weight = model.weight.squeeze()

            loss = torch.mean(torch.clamp(1 - y * output, min=0))
            loss += args.c * (weight.t() @ weight) / 2.0
            loss.backward()
            optimizer.step()
            sum_loss += float(loss)
        logger.info("Epoch: {:4d}\tloss: {}".format(epoch, sum_loss / N));
    # 训练完成后将参数同步到全局变量中
    logger.info("将svm参数同步到全局变量中");
    w = model.weight.squeeze().t().detach().cpu().numpy();
    b = model.bias.squeeze().detach().cpu().numpy();
    svmPara[0] = w;
    svmPara[1] = b;
    logger.info("weight:"+str(svmPara[0]));
    logger.info("bias:"+str(svmPara[1]));

def trainPrint(x1, y1, epoch=50, batchsize=1000):
    # 定义epoch和batchsize参数
    args.epoch = epoch;
    args.batchsize = batchsize;
    # 转换X，Y的格式
    X = torch.FloatTensor(x1)
    Y = torch.FloatTensor(y1)
    N = len(Y)
    # 定义模型
    featureDim = x1.shape[1];
    model = nn.Linear(featureDim, 1)
    model.to(args.device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(args.epoch):
        perm = torch.randperm(N)
        sum_loss = 0
        cnt = 0;
        for i in range(0, N, args.batchsize):
            cnt += 1;
            x = X[perm[i : i + args.batchsize]].to(args.device)
            y = Y[perm[i : i + args.batchsize]].to(args.device)
            # 用于测试直接通过ndarray的参数是否可以得到相同的结果
            # x2 = x1[perm[i : i + args.batchsize]];
            # y2 = y1[perm[i : i + args.batchsize]];
            # W = model.weight.squeeze().t().detach().cpu().numpy();
            # b = model.bias.squeeze().detach().cpu().numpy();
            # output1 = np.dot(x2,W)+b;

            optimizer.zero_grad()
            output = model(x).squeeze()
            # weight = model.weight.squeeze()

            loss = torch.mean(torch.clamp(1 - y * output, min=0))
            # loss += args.c * (weight.t() @ weight) / 2.0
            loss.backward()
            optimizer.step()
            print("Epoch:{:3d},cnt:{:4d},loss: {:4f}".format(epoch, cnt,loss));
    # 训练完成后将参数同步到全局变量中
    print("将svm参数同步到全局变量中...");
    w = model.weight.squeeze().t().detach().cpu().numpy();
    b = model.bias.squeeze().detach().cpu().numpy();
    svmPara[0] = w;
    svmPara[1] = b;
    print("weight:"+str(svmPara[0]));
    print("bias:"+str(svmPara[1]));


# 将全局参数变量保存到.h5文件中，h5文件的名称是svmModel.h5
def saveModelPara(dir,logger):
    logger.info("保存svm模型参数...");
    if not os.path.exists(dir):
        os.mkdir(dir);
    # 加载h5文件，直接采用复写而不是添加模式*********************************
    destH5FilePath = dir + "svmModel.h5";
    f = h5py.File(destH5FilePath, 'w');
    f["svmPara_weight"] = svmPara[0];
    f["svmPara_bias"] = svmPara[1];
    f.close();
    logger.info("保存成功");
    logger.info("weight:"+str(svmPara[0]));
    logger.info("bias:"+str(svmPara[1]));

def saveModelParaPrint(dir):
    print("保存svm模型参数...");
    if not os.path.exists(dir):
        os.mkdir(dir);
    # 加载h5文件，直接采用复写而不是添加模式*********************************
    destH5FilePath = dir + "svmModel.h5";
    f = h5py.File(destH5FilePath, 'w');
    f["svmPara_weight"] = svmPara[0];
    f["svmPara_bias"] = svmPara[1];
    f.close();
    print("保存成功");
    print("weight:"+str(svmPara[0]));
    print("bias:"+str(svmPara[1]));

def loadSvmPara(dir,logger):
    destFileDir = dir + "svmModel.h5";
    logger.info("加载svm模型参数...："+destFileDir);
    f = h5py.File(destFileDir, 'r');
    svmPara[0] = f["svmPara_weight"].value;
    svmPara[1] = f["svmPara_bias"].value;
    logger.info("weight:"+str(svmPara[0]));
    logger.info("bias:"+str(svmPara[1]));
    f.close();
    logger.info("加载完成");

def sign(x):
    x[x>=0] = 1;
    x[x<0] = -1;
    return x;

# 对一维的特征数据进行预测
def predict(data):
    output = np.dot(data, svmPara[0]) + svmPara[1];
    return sign(output);

# 对图像形式的二维特征数据进行预测 size*size×4
def predictImg(data):
    output = np.dot(data, svmPara[0]) + svmPara[1];
    return sign(output);

# 对dncnn中的batch格式特征数据进行预测 128*1*size*size*4  1*1*size*size*4
def predictImgDncnn(data):
    output = np.dot(data, svmPara[0]) + svmPara[1];
    return sign(output);
# def visualize(X, Y, model):
#     W = model.weight.squeeze().detach().cpu().numpy()
#     b = model.bias.squeeze().detach().cpu().numpy()
#
#     delta = 0.001
#     x = np.arange(X[:, 0].min(), X[:, 0].max(), delta)
#     y = np.arange(X[:, 1].min(), X[:, 1].max(), delta)
#     x, y = np.meshgrid(x, y)
#     xy = list(map(np.ravel, [x, y]))
#
#     z = (W.dot(xy) + b).reshape(x.shape)
#     z[np.where(z > 1.0)] = 4
#     z[np.where((z > 0.0) & (z <= 1.0))] = 3
#     z[np.where((z > -1.0) & (z <= 0.0))] = 2
#     z[np.where(z <= -1.0)] = 1
#
#     plt.figure(figsize=(10, 10))
#     plt.xlim([X[:, 0].min() + delta, X[:, 0].max() - delta])
#     plt.ylim([X[:, 1].min() + delta, X[:, 1].max() - delta])
#     plt.contourf(x, y, z, alpha=0.8, cmap="Greys")
#     plt.scatter(x=X[:, 0], y=X[:, 1], c="black", s=10)
#     plt.tight_layout()
#     plt.show()



# if __name__ == "__main__":
#
#     X, Y = make_blobs(n_samples=500, centers=2, random_state=0, cluster_std=0.4)
#     X = (X - X.mean()) / X.std()
#     Y[np.where(Y == 0)] = -1
#
#     model = nn.Linear(2, 1)
#     model.to(args.device)
#
#     train(X, Y, model, args)
#     visualize(X, Y, model)

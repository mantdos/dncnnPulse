import kornia as K
import torch
import cv2
from feature import GD_blur,LTP_blur
img=cv2.imread('/home/p/PycharmProjects/dncnnPulse/data/Set12/01.png')
img=img[:,:,0]
img_tem=img/255.
img_t:torch.Tensor=K.utils.image_to_tensor(img)
img_t=img_t.unsqueeze(0).float()/255.
######################中值
img_mid=K.median_blur(img_t,(3,3))
img_mid_ndarray=img_mid.squeeze().numpy()
########################平均值
img_av=K.box_blur(img_t,(3,3))
img_ABD = abs(img_av - img_t);
# img_av_ndarray=img_av.squeeze().numpy()
################GD
img_GD=abs(GD_blur(img_t))
###############LTP
img_LTP=LTP_blur(img_t,float(5/255))
a=1




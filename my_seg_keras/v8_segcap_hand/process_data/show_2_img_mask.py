#coding=utf-8
import sys
sys.path.append('../')
from v1 import config as cfg
import glob
import numpy as np
import cv2
import random
import os
##########################要改的东西#######################################
img_path = cfg.train_images
label_path =  cfg.train_label
n_classes = cfg.n_classes
reshape_size =(32,32,1)
vstack_pics = False
hstack_pics = True
###########################################################################

def show_dataset( images_path , segs_path ,  n_classes ):
    font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体

    images = glob.glob( os.path.join(images_path ,"*.jpg" ) ) \
    + glob.glob( os.path.join(images_path , "*.png" ) ) +  glob.glob( os.path.join(images_path, "*.jpeg" ) )
    images.sort()
    segmentations  =  glob.glob( os.path.join(segs_path ,"*.jpg" ) )\
    + glob.glob( os.path.join(segs_path , "*.png" ) ) +  glob.glob( os.path.join(segs_path, "*.jpeg" ) )
    segmentations.sort()

    colors = [( random.randint(0,255),random.randint(0,255),random.randint(0,255)) for _ in range(n_classes)]

    assert len( images ) == len(segmentations)

    for im_fn , seg_fn in zip(images,segmentations):
        show_pos = (10, 10)

        #1、读原图
        img = cv2.imread( im_fn )
        # 添加文字，1.2表示字体大小，（0,40）是初始的位置，(255,255,255)表示颜色，2表示粗细
        img = cv2.putText(img, 'img', show_pos, font, 0.5, (255, 255,255),2)

        #2、读label
        seg = cv2.imread( seg_fn )
        print(np.unique( seg ))
        seg_img = np.zeros_like( seg )
        for c in range(n_classes):
            seg_img[:,:,0] += ( (seg[:,:,0] == c )*( colors[c][0] )).astype('uint8')
            seg_img[:,:,1] += ((seg[:,:,0] == c )*( colors[c][1] )).astype('uint8')
            seg_img[:,:,2] += ((seg[:,:,0] == c )*( colors[c][2] )).astype('uint8')
        seg_img = cv2.putText(seg_img, 'label', show_pos, font, 0.5, (255, 255,255),2)

        if vstack_pics:
            vmerge = np.vstack((img, seg_img))  # 垂直拼接
            cv2.imshow("1: img  2: label ", vmerge)
        elif hstack_pics:
            hstack = np.hstack((img, seg_img))  # 垂直拼接
            cv2.imshow("1: img  2: label", hstack)
        else:
            cv2.imshow("img", img)
            cv2.imshow("seg_img", seg_img)

        cv2.waitKey(2000)







show_dataset(img_path , label_path ,  n_classes)
# show_dataset(cfg.val_images , cfg.val_label ,  cfg.n_classes)

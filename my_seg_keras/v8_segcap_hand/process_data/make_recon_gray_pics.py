# -*- coding: utf-8 -*-
import os
import cv2
import sys
sys.path.append('../')
from v1 import config as cfg
import glob
import numpy as np
##########################要改的东西#######################################
img_path = cfg.train_images
label_path =  cfg.train_label
save_gray_route = cfg.gray_label
###########################################################################



def get_files_list(path):
    images_list = glob.glob( os.path.join(path ,"*.jpg" ) ) \
    + glob.glob( os.path.join(path , "*.png" ) ) +  glob.glob( os.path.join(path, "*.jpeg" ) )
    images_list.sort()
    return images_list

#获取所有图片文件
img_list = get_files_list(img_path)
label_list = get_files_list(label_path)
assert len(img_list) == len(label_list)
os.makedirs(save_gray_route,exist_ok=True)
#一对对图片读取、合并、转成灰度图、保存
for img_file,label_file in zip(img_list,label_list):

    # 1、读原图、label
    img = cv2.imread(img_file)
    label = cv2.imread(label_file)

    # 2、合并
    merge = np.multiply(img,label)

    #3、 转成灰度图
    merge = cv2.cvtColor(merge, cv2.COLOR_BGR2GRAY)
    merge = np.array(merge)
    #4、显示
    # cv2.imshow('cc',merge)
    # cv2.waitKey(500)

    #5、保存(cv2无法直接保存到绝对路径，只能cmd命令转移了)
    save_name = img_file.split('/')[-1]
    save_name = save_name.split('.')[0] + '.jpg'
    save_dir = os.path.join(save_gray_route,save_name)
    cv2.imwrite(save_name,merge)
    cmd = 'mv '+ save_name + ' ' + save_dir
    os.system(cmd)
    print(cmd)
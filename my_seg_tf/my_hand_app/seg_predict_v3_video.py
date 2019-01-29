import numpy as np
import tensorflow as tf
import sys
sys.path.append('../')

import config as cfg
import time
#用cv2显示不正常，fuck主要原因是，Opencv是BGR,而我原来训练的是RGB
import cv2
import os
# from PIL import Image
# import matplotlib.image as mpimg # mpimg 用于读取图片
# import matplotlib.pyplot as plt
##########################   要改的东西   #######################################
from models.unet import Unet
num_epochs = cfg.num_epochs
is_train=True #True使用训练集，#False使用测试集
test_data_number = cfg.test_data_number
predict_pics_save = cfg.predict_pics_save #
batch_size = cfg.batch_size
model_restore_name = None
model_restore_name = "model_1999.ckpt"
##########################   end   ##########################################
#1、视频流


os.makedirs(predict_pics_save,exist_ok=True)

if  __name__== '__main__':

    # 1、读图，opencv视频流
    cap = cv2.VideoCapture("/home/mo/work/seg_caps/my_hand_app/data/ddd.mp4")

    # 2、GPU设置
    session_config = tf.ConfigProto(
        device_count={'GPU': 0},
        gpu_options={'allow_growth': 1,
                     # 'per_process_gpu_memory_fraction': 0.1,
                     'visible_device_list': '0'},
        allow_soft_placement=True)  ##这个设置必须有，否则无论如何都会报cudnn不匹配的错误,BUG十分隐蔽


    with tf.Session(config=session_config) as sess:
        # 1、定义model
        model = Unet(sess, cfg, is_train=is_train)

        # 2、恢复模型
        model.restore(model_restore_name)

        # 3、视频流预测
        while True:
            since = time.time()
            #1、读取一帧
            ret,frame = cap.read()
            if ret == False:
                break

            #2、resize成网络需要的大小尺寸
            image_RGB = cv2.resize(frame, (cfg.input_shape[0], cfg.input_shape[1]))

            #3、图片旋转
            image_RGB = np.rot90(image_RGB)
            image_RGB = np.rot90(image_RGB)
            image_RGB = np.rot90(image_RGB)
            # cv2.imshow('image_RGB', image_RGB)

            #4、RGB2BGR，训练的时候是用BGR格式训练的，因此要还原成BGR格式
            image = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2BGR)
            # cv2.imshow('image_BGR', image)

            #5、图片归一化和拓张维度，net要四维的
            image = image / 255.0
            image = np.expand_dims(image, axis=0)

            #6、如果退出
            if cv2.waitKey(10) ==27:
                break

            #7、网络预测
            pre= model.predict(image)

            #8、把四维图片还原成三维图片，并把单通道转成三通道
            pre_list = np.split(pre[0],batch_size,axis=0)
            pres = np.squeeze(pre_list,axis=0)
            pres = cv2.cvtColor(pres, cv2.COLOR_GRAY2BGR)
            image_RGB = image_RGB/255

            #9、合并
            merge= np.hstack((image_RGB,pres))

            #10、显示
            cv2.imshow('merge',merge)
            seconds = time.time() - since
            print('seconds: ',seconds)


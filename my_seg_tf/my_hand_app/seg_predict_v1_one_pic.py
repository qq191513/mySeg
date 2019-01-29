import numpy as np
import tensorflow as tf
import sys
sys.path.append('../')

import config as cfg
import time
#用cv2显示不正常，fuck主要原因是，Opencv是BGR,而我原来训练的是RGB
import cv2
import os
import daiquiri
from tool.visual_tool import *
logger = daiquiri.getLogger(__name__)

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
#1、切割一张图

os.makedirs(predict_pics_save,exist_ok=True)

if  __name__== '__main__':

    # 1、读图
    im_file = os.path.join('data', '170.png')
    # （1）用opencv方式读图
    image = cv2.cvtColor(cv2.imread(im_file),cv2.COLOR_RGB2BGR)

    # （2）用PIL方式读图
    # image =  np.array(Image.open(im_file))

    # 2、图片归一化、resize
    image = image / 255.0


    image = cv2.resize(image,(cfg.input_shape[0],cfg.input_shape[1]))
    image = np.expand_dims(image, axis=0)
    # 3、GPU设置
    session_config = tf.ConfigProto(
        device_count={'GPU': 0},
        gpu_options={'allow_growth': 1,
                     # 'per_process_gpu_memory_fraction': 0.1,
                     'visible_device_list': '0'},
        allow_soft_placement=True)  ##这个设置必须有，否则无论如何都会报cudnn不匹配的错误


    with tf.Session(config=session_config) as sess:
        # 1、定义model
        model = Unet(sess, cfg, is_train=is_train)


        # 2、恢复模型
        model.restore(model_restore_name)

        # 3、预测
        since = time.time()
        pre= model.predict(image)
        seconds = time.time() - since

        # 4、显示预测结果
        pre_list = np.split(pre[0],batch_size,axis=0)
        image = np.squeeze(image,axis=0)
        pres = np.squeeze(pre_list,axis=0)
        plt_imshow_2_pics(image, pres)
        time.sleep(2)





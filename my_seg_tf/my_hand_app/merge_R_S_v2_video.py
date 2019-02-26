import tensorflow as tf
import sys
sys.path.append('../')

import config as cfg
import time
#用cv2显示不正常，fuck主要原因是，Opencv是BGR,而我原来训练的是RGB
import cv2
import os
from tool.visual_tool import *
from tool.cut_black import get_box
##########################  seg 要改的东西   #######################################
from models.unet import Unet
num_epochs = cfg.num_epochs
is_train=True #True使用训练集，#False使用测试集
test_data_number = cfg.test_data_number
predict_pics_save = cfg.predict_pics_save #
batch_size = cfg.batch_size
model_restore_name = None
model_restore_name = "model_1999.ckpt"
##########################   end   ##########################################

from models.config_em import get_coord_add
from models.config_em import search_keyword_files
from models.config_em import read_label_txt_to_dict

##########################   recognize 要改的东西   #######################################
import models.capsnet_em as net
recognize_data_dir = 'data'
recognize_labels_txt_keywords = 'asl_labels.txt'
recognize_latest_model_ckpt = os.path.join('recognize_caps_logdir/asl/')
recognize_num_classes=36
recognize_dataset_name='asl'
recognize_input_shape=[28,28,3]
plot_pics = False
##########################   end   ##########################################
font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体

if  __name__== '__main__':

    # 1、读取mp4
    cap = cv2.VideoCapture("/home/mo/work/seg_caps/my_hand_app/data/ddd.mp4")
    im_file = os.path.join('data/asl_dataset_gray_32x32/5', 'hand1_5_bot_seg_1_cropped.jpg')

    # 2、GPU设置
    session_config = tf.ConfigProto(
        device_count={'GPU': 0},
        gpu_options={'allow_growth': 1,
                     # 'per_process_gpu_memory_fraction': 0.1,
                     'visible_device_list': '0'},
        allow_soft_placement=True)  ##这个设置必须有，否则无论如何都会报cudnn不匹配的错误,BUG十分隐蔽，真是智障

    # 3、定义sess1、sess2
    g1 = tf.Graph()
    g2 = tf.Graph()
    sess1 = tf.Session(graph=g1,config=session_config)  # Session1
    sess2 = tf.Session(graph=g2,config=session_config)  # Session2

    # 4、加载第一个模型
    with sess1.as_default():
        with g1.as_default():
                model = Unet(sess1, cfg, is_train=is_train)
                sess1.run(tf.global_variables_initializer())
                model.restore(model_restore_name)

    # 5、加载第二个模型
    with sess2.as_default():  # 1
        with g2.as_default():
                coord_add = get_coord_add(recognize_dataset_name)
                input = tf.placeholder(tf.float32, [cfg.batch_size, recognize_input_shape[0], recognize_input_shape[1],
                                                    recognize_input_shape[2]])
                # batch_x_norm = slim.batch_norm(input, center=False, is_training=False, trainable=False)

                output = net.build_arch(input, coord_add, is_train=False, num_classes=recognize_num_classes)
                sess2.run(tf.global_variables_initializer())
                var_list = tf.global_variables()
                model_file = tf.train.latest_checkpoint(recognize_latest_model_ckpt)
                saver = tf.train.Saver(var_list=var_list)
                saver.restore(sess2, model_file)

    # 6、视频流预测
    while True:

        # 1、读取一帧
        since = time.time()
        ret, frame = cap.read()
        if ret == False:
            break

        # 2、resize成网络需要的大小尺寸
        image_RGB = cv2.resize(frame, (cfg.input_shape[0], cfg.input_shape[1]))

        # 3、图片旋转
        # image_90 = np.rot90(image_RGB)  # 转90°
        # image_180 = np.rot90(image_RGB, 2)  # 转180°
        image_270 = np.rot90(image_RGB, 3)  # 转270°

        # 4、RGB2BGR，训练的时候是用BGR格式训练的，因此要还原成BGR格式
        image = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2BGR)

        # 5、图片归一化和拓张维度，net要四维的
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        # 6、使用第一个模型进行分割
        with sess1.as_default():
            with sess1.graph.as_default():  # 2
                pre = model.predict(image)

        # 7、调整维度
        pre_list = np.split(pre[0], batch_size, axis=0)
        image = np.squeeze(image, axis=0)
        pres = np.squeeze(pre_list, axis=0)
        pres = np.expand_dims(pres, axis=-1)

        # 8、定位，得出包围框
        (x1, y1, height, width) = get_box((image,pres * 255,plot_pics))

        # 9、裁剪、得出最终分割效果
        result = np.multiply(pres / 255, image_RGB)
        crop_result = result[y1:y1 + height, x1:x1 + width]
        crop_result = cv2.resize(crop_result, (cfg.input_shape[0], cfg.input_shape[1]))
        if plot_pics:
            crop_result = cv2.cvtColor(crop_result, cv2.COLOR_BGR2RGB)
            plt_imshow_1_pics(crop_result)
            time.sleep(5)  # 休息5秒防止死机
        # 10、合并、显示
        pres = cv2.cvtColor(pres, cv2.COLOR_GRAY2BGR)
        image_RGB = np.float32(image_RGB / 255)
        merge3 = np.hstack((image_RGB, pres, crop_result))

        # 11、显示3幅度图片
        # cv2.imshow('merge3', merge3)

        # 12、使用上面的测试结果
        # crop_result =crop_result*255
        crop_result = np.float32(crop_result)
        # crop_result = cv2.cvtColor(crop_result,cv2.COLOR_RGB2GRAY)  #当测试灰度图
        crop_result = cv2.resize(crop_result, (28, 28))
        # cv2.imshow('crop_result',crop_result)  #显示裁剪结果
        # cv2.waitKey(25)
        crop_result = np.expand_dims(crop_result, axis=0)
        # crop_result = np.expand_dims(crop_result, axis=-1) #当测试灰度图时要扩充维度

        # 12、不使用上面的测试结果，只用一张图片
        # crop_result = cv2.imread(im_file)
        # crop_result = cv2.cvtColor(crop_result,cv2.COLOR_RGB2GRAY)
        # crop_result = cv2.resize(crop_result, (28, 28))
        # cv2.imshow('dd', crop_result)
        # cv2.waitKey(5000)
        # crop_result = np.expand_dims(crop_result, axis=0)
        # crop_result = np.expand_dims(crop_result, axis=-1)

        # 13、使用第二个模型预测手势类别
        with sess2.as_default():
            with sess2.graph.as_default():
                result = sess2.run(output, {input: crop_result})
                # print('已经恢复的预测结果 ', result)

        # 14、映射标签
        result = np.argmax(result, axis=-1)
        print('已经恢复的预测结果 ', result)
        result = str(result[0])
        labels_txt = search_keyword_files(recognize_data_dir, recognize_labels_txt_keywords)
        labels_maps = read_label_txt_to_dict(labels_txt[0])
        label_name = labels_maps[result]
        print('label_name: ', label_name)

        # 15、显示全部时间
        seconds = time.time() - since
        print('seconds: ',seconds)

        # 16、显示4幅度图片
        predict_text = np.zeros((image_RGB.shape[0],image_RGB.shape[1]))
        predict_text = cv2.putText(predict_text, label_name, (40,80), font, 2, (1,1,1),2)
        predict_text= np.uint8(predict_text)
        predict_text = cv2.cvtColor(predict_text,cv2.COLOR_GRAY2RGB)
        merge4 = np.hstack((merge3,predict_text))
        cv2.imshow('merge4', merge4)
        key = cv2.waitKey(25)
        # 16、按键功能
        plot_pics = False
        if key& 0xFF == ord('q'):
            plot_pics =True
        if key & 0xFF == ord('n'):
            plot_pics =False
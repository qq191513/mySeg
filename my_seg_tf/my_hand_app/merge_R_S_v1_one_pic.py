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
from tool.visual_tool import plt_imshow_1_pics
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
from tool.visual_tool import print_variable

##########################   recognize 要改的东西   #######################################
import models.capsnet_em as net
recognize_data_dir = 'data'
recognize_labels_txt_keywords = 'asl_labels.txt'
recognize_latest_model_ckpt = os.path.join('recognize_caps_logdir/asl/')
recognize_num_classes=36
# recognize_num_classes=22
recognize_dataset_name='asl'
recognize_input_shape=[28,28,3]
##########################   end   ##########################################

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
        allow_soft_placement=True)  ##这个设置必须有，否则无论如何都会报cudnn不匹配的错误,BUG十分隐蔽，真是智障

    # 1、定义seg model并恢复seg模型
    g1 = tf.Graph()
    # isess = tf.Session(graph=g1)
    with g1.as_default():
        with tf.Session(config=session_config) as sess:
            model = Unet(sess, cfg, is_train=is_train)
            model.restore(model_restore_name)

    # with tf.Session(config=session_config) as sess:
            # 3、预测
            since = time.time()
            pre= model.predict(image)
            seconds = time.time() - since

            # 4、调整维度
            pre_list = np.split(pre[0],batch_size,axis=0)
            image = np.squeeze(image,axis=0)
            pres = np.squeeze(pre_list,axis=0)
            pres =np.expand_dims(pres,axis=-1)
            result = np.multiply(pres,image)

            # 5、得出包围框
            pres = pres * 255
            (x1,y1,height,width) = get_box(pres)

            # 6、裁剪
            crop_result = result[y1:y1 + height, x1:x1 + width]
            crop_result = cv2.resize(crop_result, (cfg.input_shape[0], cfg.input_shape[1]))

            # 7、显示结果
            plt_imshow_3_pics(image,pres,crop_result)

    #使用上面的测试结果
    crop_result = np.float32(crop_result)
    crop_result =crop_result
    # crop_result = cv2.cvtColor(crop_result,cv2.COLOR_BGR2GRAY)  #当测试灰度图
    crop_result = cv2.resize(crop_result, (28, 28))
    plt_imshow_1_pics(crop_result)
    cv2.imshow('crop_result',crop_result)
    cv2.waitKey(5000)
    crop_result = np.expand_dims(crop_result, axis=0)
    # crop_result = np.expand_dims(crop_result, axis=-1) #当测试灰度图时要扩充维度


    #不使用上面的测试结果，只用一张图片
    # im_file = os.path.join('data/asl_dataset_32x32/p', 'hand1_p_bot_seg_1_cropped.jpg')
    # crop_result = cv2.imread(im_file)
    # # crop_result = cv2.cvtColor(crop_result,cv2.COLOR_RGB2GRAY)
    # # crop_result = cv2.cvtColor(crop_result,cv2.COLOR_RGB2BGR)
    # crop_result = cv2.resize(crop_result, (28, 28))
    # cv2.imshow('dd', crop_result)
    # cv2.waitKey(5000)
    # crop_result = np.expand_dims(crop_result, axis=0)


    # 2、定义recognize model并恢复recognize模型
    g2 = tf.Graph()
    with g2.as_default():
        with tf.Session(config=session_config) as sess:
            coord_add = get_coord_add(recognize_dataset_name)
            input = tf.placeholder(tf.float32, [cfg.batch_size, recognize_input_shape[0], recognize_input_shape[1],
                                                recognize_input_shape[2]])
            output = net.build_arch(input, coord_add, is_train=False, num_classes=recognize_num_classes)
            sess.run(tf.global_variables_initializer())

            ####################  恢复模型  ########################################
            var_to_save = [v for v in tf.global_variables(
            ) if 'Adam' not in v.name]  # Don't save redundant Adam beta/gamma
            saver = tf.train.Saver(var_list=var_to_save, max_to_keep=5)
            model_file = tf.train.latest_checkpoint(recognize_latest_model_ckpt)
            saver.restore(sess, model_file)
            ####################    end   ########################################

            # 6、测试恢复model的预测结果
            result = sess.run(output, {input: crop_result})
            print('已经恢复的预测结果 ', result)
            result = np.argmax(result, axis=-1)
            print('已经恢复的预测结果 ', result)

            # 7、映射标签
            result = str(result[0])
            labels_txt = search_keyword_files(recognize_data_dir, recognize_labels_txt_keywords)
            labels_maps = read_label_txt_to_dict(labels_txt[0])
            label_name = labels_maps[result]
            print('label_name: ', label_name)



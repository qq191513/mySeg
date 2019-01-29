#coding=utf-8

import numpy as np
import tensorflow as tf
import sys
sys.path.append('../')
import time
import tools.development_kit as dk
import cv2
import os
from data_process.use_seg_tfrecord import create_inputs_seg_hand as create_inputs
###############################   使用unet 改这里    #############################
# import tools.config.config_unet as cfg
# from model.unet import my_unet as model
##############################      end    #######################################

###############################   使用res-unet 改这里    #########################
import config.config_res_unet as cfg
from models.res_unet import my_residual_unet as model
##############################      end    #######################################

##########################   一般设置   #######################################
is_train=False #True使用训练集，#False使用测试集
test_data_number = cfg.test_data_number
batch_size = cfg.batch_size
save_list_csv = cfg.save_list_csv
save_mean_csv = cfg.save_mean_csv
save_plot_curve = cfg.save_plot_curve
input_shape = cfg.input_shape
labels_shape= cfg.labels_shape
ckpt =cfg.ckpt
predict_pics_save = cfg.predict_pics_save
restore_model  = True
train_data_number = cfg.train_data_number
test_data_number = cfg.test_data_number
test_opoch = 2
##########################   end   ##########################################

session_config = dk.set_gpu()
n_batch_train = int(train_data_number //batch_size)
n_batch_test = int(test_data_number //batch_size)

if  __name__== '__main__':
    with tf.Session(config=session_config) as sess:
        # 入口
        train_x, train_y = create_inputs(is_train)
        # train_y = tf.reshape(train_y, labels_shape)
        x = tf.placeholder(tf.float32, shape=input_shape)
        y = tf.placeholder(tf.float32, shape=labels_shape)
        # 构建网络
        prediction = model(images=x, is_train=is_train, size=input_shape, l2_reg=0.0001)
        # prediction = tf.reshape(prediction, labels_shape)
        # 初始化变量
        coord, threads = dk.init_variables_and_start_thread(sess)
        # 恢复model
        saver = dk.restore_model(sess, ckpt, restore_model=restore_model)
        # 显示参数量
        dk.show_parament_numbers()
        # 测试loop
        start_epoch= 0
        index = 0
        for epoch_n in range(start_epoch, test_opoch):
            for n_batch in range(n_batch_test):
                batch_x, batch_y = sess.run([train_x, train_y])  # 取出一个batchsize的图片。
                batch_x = batch_x / 255.0
                # 3、预测输出
                since = time.time()
                predict = sess.run(prediction, feed_dict={x: batch_x})
                seconds = time.time() - since
                # 4、预测图转二值化图（非0即1）
                predict[predict>=0.5] =1
                predict[predict< 0.5] =0
                # 5、把batch图片分割一张张图片
                batch_pre_list = np.split(predict, batch_size, axis=0)
                pre_pics_list = np.squeeze(batch_pre_list, axis=0)
                for img, label, predict in zip(batch_x, batch_y, pre_pics_list):
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    label = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
                    predict = cv2.cvtColor(predict, cv2.COLOR_GRAY2RGB)
                    hstack = np.hstack((img, label,predict))
                    hstack = cv2.resize(hstack,(512,512))

                    save_name = '{}.jpg'.format(index)
                    save_name = os.path.join(predict_pics_save, save_name)
                    cv2.imshow('hstack',hstack)
                    cv2.waitKey(500)
                    cv2.imwrite(save_name, hstack)
                    index += 1
                    print(save_name)

    coord.request_stop()
    coord.join(threads)

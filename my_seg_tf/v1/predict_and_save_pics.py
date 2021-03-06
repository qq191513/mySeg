import numpy as np
import tensorflow as tf
from model.SegCaps import SegCaps
import config as cfg
import time
import cv2
import os

##########################   要改的东西   #######################################
num_epochs = cfg.num_epochs
is_train=True #校验阶段
test_data_number = 100
predict_pics_save = cfg.predict_pics_save #
batch_size = cfg.batch_size
##########################   end   ##########################################

from v1.use_seg_tfrecord import create_inputs_seg_hand
os.makedirs(predict_pics_save,exist_ok=True)

if  __name__== '__main__':
    images,labels = create_inputs_seg_hand(is_train = is_train)

    session_config = tf.ConfigProto(
        device_count={'GPU': 0},
        gpu_options={'allow_growth': 1,
                     # 'per_process_gpu_memory_fraction': 0.1,
                     'visible_device_list': '0'},
        allow_soft_placement=True)  ##这个设置必须有，否则无论如何都会报cudnn不匹配的错误,BUG十分隐蔽，真是智障
    with tf.Session(config=session_config) as sess:
        # 1、先定义model才能执行第二步的初始化
        model = SegCaps(sess, cfg, is_train=is_train)

        # 2、初始化和启动线程
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        model.restore()

        #3、测试图片
        index = 0
        for i in range(test_data_number//batch_size):
            pics,pics_masks = sess.run([images,labels])  # 取出一个batchsize的图片
            # 3、计算耗时
            since = time.time()
            pre= model.predict(pics)
            seconds = time.time() - since

            pre_list = np.split(pre,batch_size,axis=0)
            pres = np.squeeze(pre_list,axis=0)

            for img, label, pre in zip(pics, pics_masks, pres):
                label=label*255
                label = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)
                pre = pre*255
                pre = cv2.cvtColor(pre, cv2.COLOR_GRAY2BGR)
                hstack = np.hstack((img, label, pre))
                save_name = '{}.jpg'.format(index)
                save_name = os.path.join(predict_pics_save, save_name)
                cv2.imwrite(save_name, hstack)
                index += 1
                print(save_name)

    coord.request_stop()
    coord.join(threads)

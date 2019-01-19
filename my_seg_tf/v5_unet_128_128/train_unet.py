import tensorflow as tf

import config as cfg
from data_process.use_seg_tfrecord import create_inputs_seg_hand
import time
import os
from data_process.preprocess import augmentImages

##########################   要改的东西   #######################################
from model.unet import Unet
num_epochs = cfg.num_epochs
train_print_log = os.path.join(cfg.train_print_log,'train_log.txt')
model_restore_name = "model_unet_999.ckpt"
model_restore_name = None
start_epoch = 2000
start_epoch = 0

##########################   end   ##########################################
def print_and_save_txt(str=None,filename=r'log.txt'):
    with open(filename, "a+") as log_writter:
        print(str)
        log_writter.write(str)

if  __name__== '__main__':
    images,labels = create_inputs_seg_hand(is_train = True)

    session_config = tf.ConfigProto(
        device_count={'GPU': 0},
        gpu_options={'allow_growth': 1,
                     # 'per_process_gpu_memory_fraction': 0.1,
                     'visible_device_list': '0'},
        allow_soft_placement=True)  ##这个设置必须有，否则无论如何都会报cudnn不匹配的错误,BUG十分隐蔽，真是智障
    with tf.Session(config=session_config) as sess:
        # 1、先定义model才能执行第二步的初始化
        model = Unet(sess, cfg, is_train=True,size=(128, 128), l2_reg=0.0001)
        # 2、初始化和启动线程
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        if model_restore_name:
            model.restore(model_restore_name)

        # 3、训练模型
        # num_epochs=10000
        for i in range(start_epoch, num_epochs):
            since = time.time()
            #1、读图
            pics,pics_masks = sess.run([images,labels])  # 取出一个batchsize的图片
            ##########################   数据增强   ###################################
            pics = pics / 255  # 归一化，加了这句话loss值小了几十倍
            pics, pics_masks = augmentImages(pics,pics_masks)
            ##########################   end   #######################################
            # 2、训练
            loss_value= model.fit(pics, pics_masks,summary_step=i)

            # 3、计算耗时
            interval_time = time.time() - since
            # 4、打印结果
            # print('{}/{} acc_value: {:.3f} loss_value: {:.3f} ,time used: {:.3f}s'.format(i,num_epochs,acc_value,loss_value,interval_time))
            # print('{}/{} loss_value: {:.3f} ,time used: {:.3f}s'.format(i,num_epochs,loss_value,interval_time))
            message ='{}/{} loss_value: {:.3f} ,time used: {:.3f}s'.format(i,num_epochs,loss_value,interval_time)
            print_and_save_txt(str=message,filename=train_print_log)

            # 5、保存model
            if (i+1)%1000 ==0:
                model.save(i)
        # model.save()
        coord.request_stop()
        coord.join(threads)

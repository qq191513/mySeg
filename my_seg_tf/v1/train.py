import tensorflow as tf
from model.SegCaps import SegCaps
import config as cfg

##########################   要改的东西   #######################################
num_epochs = cfg.num_epochs
##########################   end   ##########################################

from v1.use_seg_tfrecord import create_inputs_seg_hand
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
        model = SegCaps(sess, cfg, is_train=True)

        # 2、初始化和启动线程
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)


        # 3、训练模型
        for i in range(num_epochs):
            pics,pics_masks = sess.run([images,labels])  # 取出一个batchsize的图片
            loss_value= model.fit(pics, pics_masks,summary_step=i)
            print('loss_value: ',loss_value)
            model.save()
        coord.request_stop()
        coord.join(threads)

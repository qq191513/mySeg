import tensorflow as tf
import sys
sys.path.append('../')
import os
import tools.development_kit as dk
from tools.loss import get_loss
from data_process.preprocess import augmentImages
###############################   改这里    ################################
from data_process.use_seg_tfrecord import create_inputs_seg_hand as create_inputs
import tools.config_unet_v1 as cfg
from model.unet_v1 import my_unet as model
is_train = True
restore_model  = True
##############################      end    #######################################

###############################     cfg    ####################################
ckpt =cfg.ckpt
batch_size = cfg.batch_size
input_shape = cfg.input_shape
labels_shape = cfg.labels_shape
labels_shape_vec  = cfg.labels_shape_vec
epoch = cfg.epoch
train_data_number = cfg.train_data_number
test_data_number = cfg.test_data_number
save_epoch_n = cfg.save_epoch_n  #每多少epoch保存一次
logdir = cfg.logdir
##############################      end    ########################################

n_batch_train = int(train_data_number //batch_size)
os.makedirs(ckpt,exist_ok=True)
session_config = dk.set_gpu()

if  __name__== '__main__':
    with tf.Session(config = session_config) as sess:
        # 入口
        train_x, train_y = create_inputs(is_train)
        # train_y = tf.reshape(train_y,labels_shape)
        x = tf.placeholder(tf.float32, shape=input_shape)
        y = tf.placeholder(tf.float32, shape=labels_shape)
        # 构建网络和预测
        prediction = model(images= x, is_train =is_train,size= input_shape,l2_reg =0.0001 )
        # 求loss
        # loss = dk.cross_entropy_loss(prediction, y)
        the_loss = get_loss('bce_dice')
        loss = the_loss(y, prediction,labels_shape_vec)
        # 设置优化器
        global_step, train_step = dk.set_optimizer(num_batches_per_epoch=n_batch_train, loss=loss)
        # 求dice_hard，不合适用acc
        dice_hard = dk.dice_hard(y, prediction, threshold=0.5, axis=[1, 2, 3], smooth=1e-5)
        # accuracy = dk.get_acc(prediction, y)
        # 初始化变量
        coord, threads = dk.init_variables_and_start_thread(sess)
        # 设置训练日志
        summary_dict = {'loss':loss,'dice_hard':dice_hard}
        summary_writer, summary_op = dk.set_summary(sess,logdir,summary_dict)
        # 恢复model
        saver = dk.restore_model(sess,ckpt,restore_model =restore_model)
        # 显示参数量
        dk.show_parament_numbers()
        # 若恢复model，则重新计算start_epoch继续
        start_epoch = 0
        if restore_model:
            step = sess.run(global_step)
            start_epoch = int(step/n_batch_train/save_epoch_n)*save_epoch_n
        # 训练loop
        total_step = n_batch_train * epoch
        for epoch_n in range(start_epoch,epoch):
            for n_batch in range(n_batch_train):
                batch_x, batch_y = sess.run([train_x, train_y])
                ##########################   数据增强   ###################################
                batch_x = batch_x / 255  # 归一化，加了这句话loss值小了几十倍
                batch_x, batch_y = augmentImages(batch_x, batch_y)
                ##########################   end   #######################################
                # 训练一个step
                _, loss_value,dice_hard_value, summary_str ,step= sess.run(
                    [train_step, loss,dice_hard, summary_op,global_step],
                    feed_dict={x: batch_x, y: batch_y})
                # 显示结果batch_size
                dk.print_message(epoch_n,step,total_step,loss_value,dice_hard_value)
                # 保存summary
                if (step + 1) % 20 == 0:
                    summary_writer.add_summary(summary_str, step)

            # 保存model
            if (((epoch_n + 1) % save_epoch_n)) == 0:
                print('epoch_n :{} saving movdel.......'.format(epoch_n))
                saver.save(sess,os.path.join(ckpt,'model_{}.ckpt'.format(epoch_n)), global_step=global_step)

        dk.stop_threads(coord,threads)
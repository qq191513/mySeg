import tensorflow as tf
import sys
sys.path.append('../')
import os
import tools.development_kit as dk
from tools.loss import get_loss
from data_process.preprocess import augmentImages
import time
from data_process.use_seg_tfrecord import create_inputs_seg_hand as create_inputs
###############################   segcap 改这里    ###############################################
# import config.config_segcap as cfg
# from model.segcap import my_segcap as model
# is_train = True
# restore_model  = False
##############################      end    ######################################################

###############################  res_segcap 改这里    ##########################################
# import config.config_res_segcap as cfg
# from model.res_segcap import my_segcap as model
# is_train = True
# restore_model  = False
##############################      end    ######################################################

###############################  res_segcap_mini 改这里    ######################################
# import config.config_res_segcap_mini as cfg
# from model.res_segcap_mini import my_segcap as model
# is_train = True
# restore_model  = False
##############################      end    ######################################################

###############################  res_segcap_mini_v1 改这里    ######################################
# import config.config_res_segcap_mini_v1 as cfg
# from model.res_segcap_mini_v1 import my_segcap as model
# is_train = True
# restore_model  = False
##############################      end    ######################################################


###############################  res_segcap_my_final 改这里    ######################################
# import config.config_res_segcap_my_final as cfg
# from model.res_segcap_my_final import my_segcap as model
from choice import choice_cfg
from choice import choice_model
is_train = True
restore_model  = True
##############################      end    ######################################################


###############################     cfg    ####################################
model=choice_model()
cfg =choice_cfg()
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
lr_range = cfg.lr_range
##############################      end    ########################################

n_batch_train = int(train_data_number //batch_size)
os.makedirs(ckpt,exist_ok=True)
session_config = dk.set_gpu()

if  __name__== '__main__':
    with tf.Session(config = session_config) as sess:
        # 入口
        train_x, train_y = create_inputs(is_train)
        x = tf.placeholder(tf.float32, shape=input_shape)
        y = tf.placeholder(tf.float32, shape=labels_shape)
        # 构建网络和预测
        prediction = model(images= x, is_train =is_train,size= input_shape,l2_reg =0.0001 )
        # 求loss
        # the_loss = get_loss('bce_dice')
        # the_loss = get_loss('bce_dice_focus')
        # the_loss = get_loss('bce_dice_margin')
        the_loss = get_loss('dice_margin_focus')
        # the_loss = get_loss('bce_dice_margin_focus')
        loss = the_loss(y, prediction,labels_shape_vec)
        # 设置优化器
        global_step, train_step = dk.set_optimizer(lr_range=lr_range,num_batches_per_epoch=n_batch_train, loss=loss)
        # 求dice_hard，不合适用acc
        dice_hard = dk.dice_hard(y, prediction, threshold=0.5, axis=[1, 2, 3], smooth=1e-5)
        # dice_hard = dk.iou_metric(prediction, y)

        # 初始化变量
        coord, threads = dk.init_variables_and_start_thread(sess)
        # 设置训练日志
        summary_dict = {'loss':loss,'dice_hard':dice_hard}
        summary_writer, summary_op = dk.set_summary(sess,logdir,summary_dict)
        # 恢复model
        saver,start_epoch = dk.restore_model(sess,ckpt,restore_model =restore_model)
        # 显示参数量
        dk.show_parament_numbers()
        # 训练loop
        total_step = n_batch_train * epoch
        for epoch_n in range(start_epoch,epoch):
            since = time.time()
            for n_batch in range(n_batch_train):
                batch_x, batch_y = sess.run([train_x, train_y])
                ##########################   数据增强   ###################################
                batch_x = batch_x / 255.0  # 归一化，加了这句话loss值小了几十倍
                batch_x, batch_y = augmentImages(batch_x, batch_y)
                ##########################   end   #######################################
                # 训练一个step
                _, loss_value,dice_hard_value, summary_str ,step= sess.run(
                    [train_step, loss,dice_hard, summary_op,global_step],
                    feed_dict={x: batch_x, y: batch_y})
                # 显示结果batch_size
                dk.print_effect_message(epoch_n,n_batch,n_batch_train,loss_value,dice_hard_value)
                # 保存summary
                if (step + 1) % 20 == 0:
                    summary_writer.add_summary(summary_str, step)

            # 显示进度和耗时
            seconds_mean = (time.time() - since) / n_batch_train
            dk.print_progress_and_time_massge(seconds_mean,step,total_step)

            # 保存model
            if (((epoch_n + 1) % save_epoch_n)) == 0:
                print('epoch_n :{} saving movdel.......'.format(epoch_n))
                saver.save(sess,os.path.join(ckpt,'model_{}.ckpt'.format(epoch_n)), global_step=global_step)

        dk.stop_threads(coord,threads)
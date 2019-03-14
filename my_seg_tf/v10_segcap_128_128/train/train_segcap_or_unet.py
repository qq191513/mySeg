import tensorflow as tf
# import sys
# sys.path.append('../')
import os
import tools.development_kit as dk
from tools.loss import get_loss
from data_process.preprocess import augmentImages
import time
from data_process.use_seg_tfrecord import create_inputs_seg_hand as create_inputs
from tensorflow.python import debug as tfdbg
from choice import model
from choice import restore_model
from choice import cfg
###############################     cfg    ####################################
ckpt = cfg.ckpt
batch_size = cfg.batch_size
input_shape = cfg.input_shape
labels_shape = cfg.labels_shape
labels_shape_vec = cfg.labels_shape_vec
epoch = cfg.epoch
train_data_number = cfg.train_data_number
test_data_number = cfg.test_data_number
save_epoch_n = cfg.save_epoch_n  # 每多少epoch保存一次
logdir = cfg.logdir
lr_range = cfg.lr_range
choose_loss = cfg.choose_loss
is_train =True
use_tensoflow_debug=False
##############################      end    ########################################

def train_model():

    # 代码初始化
    n_batch_train = int(train_data_number // batch_size)
    print('n_batch_train: ', n_batch_train)
    os.makedirs(ckpt, exist_ok=True)
    session_config = dk.set_gpu()
    latest_train_data = {}
    latest_train_data['latest_min'] = [0,0,0,0,0,0]
    latest_train_data['latest_max'] = [0,0,0,0,0,0]
    latest_train_data['latest_mean'] = [0,0,0,0,0,0]

    with tf.Session(config = session_config) as sess:
        #如果使用tensorlfow1的debug神器（主要用于查出哪里有inf或nan，不能在pycharm运行调试程序，只能在xshell里面运行）
        if use_tensoflow_debug:
            sess = tfdbg.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tfdbg.has_inf_or_nan)
            #然后在xshell里面运行run -f has_inf_or_nan
            # 一旦inf / nan出现，界面现实所有包含此类病态数值的张量，按照时间排序。所以第一个就最有可能是最先出现inf / nan的节点。
            # 可以用node_info, list_inputs等命令进一步查看节点的类型和输入，来发现问题的缘由。
            #教程https://blog.csdn.net/tanmx219/article/details/82318133
        # 入口
        train_x, train_y = create_inputs(is_train)
        x = tf.placeholder(tf.float32, shape=input_shape)
        y = tf.placeholder(tf.float32, shape=labels_shape)
        # 构建网络和预测
        prediction,endpoint = model(images= x, is_train =is_train,size= input_shape,l2_reg =0.0001 )
        # 打印模型结构
        dk.print_model_struct(endpoint)
        # 求loss
        the_loss = get_loss(choose_loss)
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
            dice_hard_value_list = []#清空
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
                # 保存结果
                dice_hard_value_list.append(dice_hard_value)

            # 显示进度、耗时、最小最大平均值
            seconds_mean = (time.time() - since) / n_batch_train
            latest_train_data = dk.print_progress_and_time_massge(
                seconds_mean,step,total_step,dice_hard_value_list,latest_train_data)

            # 保存model
            if (((epoch_n + 1) % save_epoch_n)) == 0:
                print('epoch_n :{} saving movdel.......'.format(epoch_n))
                saver.save(sess,os.path.join(ckpt,'model_{}.ckpt'.format(epoch_n)), global_step=global_step)

        dk.stop_threads(coord,threads)
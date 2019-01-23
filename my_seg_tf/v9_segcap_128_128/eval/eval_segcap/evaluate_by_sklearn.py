#coding=utf-8

import numpy as np
import tensorflow as tf
import sys
sys.path.append('../')
import time
import tools.development_kit as dk
import tools.eval_seg_kit as esk
from data_process.use_seg_tfrecord import create_inputs_seg_hand as create_inputs
import os
###############################  res_segcap 改这里    ######################################
# import config.config_res_segcap as cfg
# from model.res_segcap import my_segcap as model
##############################      end    ######################################################

###############################  res_segcap_mini 改这里    ######################################
# import tools.config.config_res_segcap_mini as cfg
# from model.res_segcap_mini import my_segcap as model
##############################      end    ######################################################

###############################  res_segcap_mini_v1 改这里    ######################################
# import config.config_res_segcap_mini_v1 as cfg
# from model.res_segcap_mini_v1 import my_segcap as model
##############################      end    ######################################################


###############################  res_segcap_my_final 改这里    ######################################
import config.config_res_segcap_my_final as cfg
from model.res_segcap_my_final import my_segcap as model
##############################      end    ######################################################


##########################   一般设置   #######################################
is_train=False #True使用训练集，#False使用测试集
batch_size = cfg.batch_size
save_list_csv = cfg.save_list_csv
save_mean_csv = cfg.save_mean_csv
save_plot_curve_dir = cfg.save_plot_curve_dir
input_shape = cfg.input_shape
labels_shape= cfg.labels_shape
ckpt =cfg.ckpt
restore_model  = True
train_data_number = cfg.train_data_number
test_data_number = cfg.test_data_number
test_opoch = 2
##########################   end   ##########################################
#代码初始化
session_config = dk.set_gpu()
n_batch_train = int(train_data_number //batch_size)
n_batch_test = int(test_data_number //batch_size)
os.makedirs(save_plot_curve_dir,exist_ok=True)

if  __name__== '__main__':
    with tf.Session(config=session_config) as sess:
        # 入口
        train_x, train_y = create_inputs(is_train)
        train_y = tf.reshape(train_y, labels_shape)
        x = tf.placeholder(tf.float32, shape=input_shape)
        y = tf.placeholder(tf.float32, shape=labels_shape)
        # 构建网络
        prediction = model(images=x, is_train=True, size=input_shape, l2_reg=0.0001)
        prediction = tf.reshape(prediction, labels_shape)
        # 初始化变量
        coord, threads = dk.init_variables_and_start_thread(sess)
        # 恢复model
        saver = dk.restore_model(sess, ckpt, restore_model=restore_model)
        # 显示参数量
        dk.show_parament_numbers()
        # 列表初始化
        AUC_ROC_list,AUC_prec_rec_list,accuracy_list,specificity_list,sensitivity_list,\
        precision_list,jaccard_index_list,F1_score_list,all_list,mean_list\
            =[],[],[],[],[],[],[],[],[],[]
        # 测试loop
        start_epoch= 0
        index = 0
        for epoch_n in range(start_epoch, test_opoch):
            for n_batch in range(n_batch_test):
                batch_x,batch_y = sess.run([train_x,train_y])  # 取出一个batchsize的图片。
                batch_x = batch_x / 255.0
                # 3、预测输出
                since = time.time()
                predict = sess.run(prediction,feed_dict={x: batch_x})
                seconds = time.time() - since
                # 4、预测图转二值化图（非0即1）,加了这步性能指标会下降2%左右
                predict[predict>=0.5] =1
                predict[predict< 0.5] =0
                # 5、把batch图片分割一张张图片
                batch_pre_list = np.split(predict,batch_size,axis=0)
                pre_pics_list = np.squeeze(batch_pre_list,axis=0)

                for label, pre in zip( batch_y, pre_pics_list):
                    index +=1
                    y_scores = pre.reshape(-1, 1)
                    y_true = label.reshape(-1, 1)
                    print('#########################   start   ####################################')

                    # 1、画ROC曲线
                    AUC_ROC = esk.plot_roc_curve(y_true,y_scores,save_plot_curve_dir,curve_name=str(index))
                    AUC_ROC_list.append(AUC_ROC)

                    #2、画P_R-curve曲线
                    AUC_prec_rec = esk.plot_precision_recall_curve(y_true,y_scores,save_plot_curve_dir,curve_name=str(index))

                    #3、Confusion matrix
                    y_pred_binary = esk.convert_to_binary(shape = y_scores.shape[0], y_scores = y_scores)
                    accuracy, specificity, sensitivity, precision \
                        = esk.plot_confusion_matrix(y_true, y_pred_binary)

                    #4、Jaccard similarity index
                    jaccard_index = esk.get_jaccard_index(y_true, y_pred_binary)

                    #5、F1 score
                    F1_score = esk.get_F1_score(y_true, y_pred_binary)

                    print('#########################   end   ####################################')
                    # 保存结果
                    AUC_prec_rec_list.append(AUC_prec_rec)
                    accuracy_list.append(accuracy)
                    specificity_list.append(specificity)
                    sensitivity_list.append(sensitivity)
                    precision_list.append(precision)
                    jaccard_index_list.append(jaccard_index)
                    F1_score_list.append(F1_score)


        #1、评估数据存进列表中
        all_list = [AUC_ROC_list, AUC_prec_rec_list, accuracy_list, specificity_list \
            , sensitivity_list, precision_list, jaccard_index_list, F1_score_list]
        name_list = ['AUC_ROC', 'AUC_prec_rec', 'accuracy', 'specificity',
                     'sensitivity', 'precision', 'jaccard_index', 'F1_score']

        # 2、panda保存所有图片的评估值到CSV文件
        esk.save_all_pics_value(name_list, all_list,save_list_csv)

        #3、panda保存平均值到CSV文件
        esk.save_mean_value(name_list, all_list,save_mean_csv)

        #4、结束
        dk.stop_threads(coord,threads)




#coding=utf-8
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score

import numpy as np
import tensorflow as tf
import sys
sys.path.append('../')

import config as cfg
import time
from matplotlib import pyplot as plt
import pandas as pd
import os
import sys
sys.path.append('../')
##########################   要改的东西   #######################################
from model.unet import Unet
num_epochs = cfg.num_epochs
is_train=True #True使用训练集，#False使用测试集
test_data_number = cfg.test_data_number
batch_size = cfg.batch_size
save_list_csv = cfg.save_list_csv
save_mean_csv = cfg.save_mean_csv
save_plot_curve = cfg.save_plot_curve
model_restore_name = "model_1999.ckpt"

##########################   end   ##########################################

from data_process.use_seg_tfrecord import create_inputs_seg_hand



def plot_roc_curve(y_true,y_scores):
    fpr, tpr, thresholds = roc_curve((y_true), y_scores)
    AUC_ROC = roc_auc_score(y_true, y_scores)
    print("Area under the ROC curve: " + str(AUC_ROC))
    roc_curve_figure = plt.figure()
    plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_plot_curve,"ROC.png"))
    plt.cla()
    plt.close("all")
    return AUC_ROC

def plot_precision_recall_curve(y_true, y_scores):
    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
    AUC_prec_rec = np.trapz(precision, recall)
    print("Area under Precision-Recall curve: " + str(AUC_prec_rec))
    prec_rec_curve = plt.figure()
    plt.plot(recall, precision, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
    plt.title('Precision - Recall curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_plot_curve,"Precision_recall.png"))

    plt.cla()
    plt.close("all")
    return AUC_prec_rec


def convert_to_binary(shape,y_scores):
    threshold_confusion = 0.5
    print("Confusion matrix:  Custom threshold (for positive) of " + str(threshold_confusion))
    y_pred = np.empty((shape))
    for i in range(shape):
        if y_scores[i] >= threshold_confusion:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    return y_pred

def plot_confusion_matrix(y_true, y_pred):
    # Confusion matrix
    confusion = confusion_matrix(y_true, y_pred)
    print(confusion)
    accuracy = 0
    if float(np.sum(confusion)) != 0:
        accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
    print("Global Accuracy: " + str(accuracy))
    specificity = 0
    if float(confusion[0, 0] + confusion[0, 1]) != 0:
        specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    print("Specificity: " + str(specificity))
    sensitivity = 0
    if float(confusion[1, 1] + confusion[1, 0]) != 0:
        sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
    print("Sensitivity: " + str(sensitivity))
    precision = 0
    if float(confusion[1, 1] + confusion[0, 1]) != 0:
        precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
    print("Precision: " + str(precision))
    return accuracy, specificity, sensitivity, precision

def get_F1_score(y_true, y_pred):
    # F1 score
    F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
    print("F1 score (F-measure): " + str(F1_score))
    return F1_score




def get_jaccard_index(y_true, y_pred):
    # Jaccard similarity index
    jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
    print("Jaccard similarity score: " + str(jaccard_index))
    return jaccard_index

def save_all_pics_value(name_list,all_list):

    data = {}
    # 1、保存所有图片的评估值
    for name, value in zip(name_list, all_list):
        data.update({name: value})
        print(data)
    result = pd.DataFrame(data=data)
    result.to_csv(save_list_csv, encoding='gbk')
    print('save to {}'.format(save_list_csv))

def save_mean_value(name_list,all_list):

    AUC_ROC_mean = np.mean(all_list[0])
    AUC_prec_rec_mean = np.mean(all_list[1])
    accuracy_mean = np.mean(all_list[2])
    specificity_mean = np.mean(all_list[3])
    sensitivity_mean = np.mean(all_list[4])
    precision_mean = np.mean(all_list[5])
    jaccard_index_mean = np.mean(all_list[6])
    F1_score_mean = np.mean(all_list[7])

    mean_list = [AUC_ROC_mean, AUC_prec_rec_mean, accuracy_mean, specificity_mean,
                 sensitivity_mean, precision_mean, jaccard_index_mean, F1_score_mean]
    data = {}
    index = 1  # 只有一行 (为何不加这个index就会报错)
    for name, mean in zip(name_list, mean_list):
        data.update({name: mean})
    mean_result = pd.DataFrame(data, index=[index])
    mean_result.to_csv(save_mean_csv, encoding='gbk')
    print('save to {}'.format(save_mean_csv))

if  __name__== '__main__':
    images, labels = create_inputs_seg_hand(is_train=is_train)

    session_config = tf.ConfigProto(
        device_count={'GPU': 0},
        gpu_options={'allow_growth': 1,
                     # 'per_process_gpu_memory_fraction': 0.1,
                     'visible_device_list': '0'},
        allow_soft_placement=True)  ##这个设置必须有，否则无论如何都会报cudnn不匹配的错误,BUG十分隐蔽，真是智障
    with tf.Session(config=session_config) as sess:
        # 1、先定义model才能执行第二步的初始化
        model = Unet(sess, cfg, is_train=is_train)

        # 2、初始化和启动线程
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        model.restore(model_restore_name)


        #列表初始化
        AUC_ROC_list,AUC_prec_rec_list,accuracy_list,specificity_list,sensitivity_list,\
        precision_list,jaccard_index_list,F1_score_list,all_list,mean_list\
            =[],[],[],[],[],[],[],[],[],[]
        #3、测试图片
        index = 0
        for i in range(test_data_number//batch_size):
            pics,pics_masks = sess.run([images,labels])  # 取出一个batchsize的图片。
            pics = pics / 255
            # 3、计算耗时
            since = time.time()
            pre= model.predict(pics)
            seconds = time.time() - since

            pre_list = np.split(pre[0],batch_size,axis=0)
            pres = np.squeeze(pre_list,axis=0)

            for label, pre in zip( pics_masks, pres):
                print('#########################   start   ####################################')

                y_scores = pre.reshape(-1, 1)
                y_true = label.reshape(-1, 1)

                # 1、画ROC曲线
                AUC_ROC = plot_roc_curve(y_true,y_scores)
                AUC_ROC_list.append(AUC_ROC)

                #2、画P_R-curve曲线
                AUC_prec_rec = plot_precision_recall_curve(y_true,y_scores)
                AUC_prec_rec_list.append(AUC_prec_rec)

                #3、Confusion matrix
                y_pred_binary = convert_to_binary(shape = y_scores.shape[0], y_scores = y_scores)
                accuracy, specificity, sensitivity, precision \
                    = plot_confusion_matrix(y_true, y_pred_binary)

                accuracy_list.append(accuracy)
                specificity_list.append(specificity)
                sensitivity_list.append(sensitivity)
                precision_list.append(precision)

                #4、Jaccard similarity index
                jaccard_index = get_jaccard_index(y_true, y_pred_binary)
                jaccard_index_list.append(jaccard_index)

                #5、F1 score
                F1_score = get_F1_score(y_true, y_pred_binary)
                F1_score_list.append(F1_score)

                print('#########################   end   ####################################')


        #1、评估数据存进列表中
        all_list = [AUC_ROC_list, AUC_prec_rec_list, accuracy_list, specificity_list \
            , sensitivity_list, precision_list, jaccard_index_list, F1_score_list]
        name_list = ['AUC_ROC', 'AUC_prec_rec', 'accuracy', 'specificity',
                     'sensitivity', 'precision', 'jaccard_index', 'F1_score']

        # 2、panda保存所有图片的评估值到CSV文件
        save_all_pics_value(name_list, all_list)

        #3、panda保存平均值到CSV文件
        save_mean_value(name_list, all_list)

        #4、结束
        coord.request_stop()
        coord.join(threads)




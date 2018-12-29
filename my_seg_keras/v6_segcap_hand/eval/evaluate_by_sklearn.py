#coding=utf-8
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score

import cv2
import numpy as np
import random
import sys
sys.path.append('../')
from v1 import config as cfg
from set_gpu import *
import time
from data_process.use_generator import X_Y_Z_iter
from matplotlib import pyplot as plt
import os
import pandas as pd
from keras import backend as K
from models.capsnet import CapsNetR3

##########################   改这里   #######################################
#已经训练好的模型
model_name =os.path.join(cfg.save_weights_path,'model.19')
model = CapsNetR3(cfg.input_shape)
# model = UNet(cfg.input_shape)
#预测的图片集
# images_path = cfg.val_images
# label_path = cfg.val_annotations
images_path = cfg.train_images
label_path = cfg.train_label
#输出路径
output_path =cfg.pre_path
save_list_csv = "../../output_predict_result/hand_128_128/test_csv/test.csv"
save_mean_csv = "../../output_predict_result/hand_128_128/test_csv/test_mean.csv"

########################        end      ########################################


if not os.path.exists(output_path):
    os.makedirs(output_path)

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
    plt.savefig("ROC.png")
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
    plt.savefig("Precision_recall.png")
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

def save_pre_pic(pr,colors,save_name):

    pr = pr.reshape((cfg.output_shape[0], cfg.output_shape[1], cfg.n_classes))
    pr = pr.argmax(axis=2)
    seg_img = np.zeros((cfg.output_shape[0], cfg.output_shape[1], 3))
    for c in range(cfg.n_classes):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')
    seg_img = cv2.resize(seg_img, (cfg.input_shape[1], cfg.input_shape[0]))
    cv2.imwrite(save_name, seg_img)

    
def get_jaccard_index(y_true, y_pred):
    # Jaccard similarity index
    jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
    print("Jaccard similarity score: " + str(jaccard_index))
    return jaccard_index

def save_all_pics_value(name_list,all_list):
    if not os.path.exists(os.path.dirname(save_list_csv)):
        os.makedirs(os.path.dirname(save_list_csv))
    data = {}
    # 1、保存所有图片的评估值
    for name, value in zip(name_list, all_list):
        data.update({name: value})
        print(data)
    result = pd.DataFrame(data=data)
    result.to_csv(save_list_csv, encoding='gbk')
    print('save to {}'.format(save_list_csv))

def save_mean_value(name_list,all_list):
    if not os.path.exists(os.path.dirname(save_list_csv)):
        os.makedirs(os.path.dirname(save_list_csv))
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

def start_eval():
    # 随机生成颜色
    colors = [( random.randint(0,255),random.randint(0,255),random.randint(0,255)) for _ in range(cfg.n_classes)  ]
    model.load_weights(os.path.join(model_name))
    model.compile(optimizer=cfg.optimizer, loss=cfg.loss,metrics=cfg.metrics)

    #列表初始化
    AUC_ROC_list,AUC_prec_rec_list,accuracy_list,specificity_list,sensitivity_list,\
    precision_list,jaccard_index_list,F1_score_list,all_list,mean_list\
        =[],[],[],[],[],[],[],[],[],[]

    #G2迭代器，X:图片、Y:标签、Z:图片名字
    G2 = X_Y_Z_iter(images_path,label_path , 1,
            cfg.n_classes, cfg.input_shape[0], cfg.input_shape[1], cfg.output_shape[0],
                     cfg.output_shape[1])
    for X,Y,Z in G2:
        print('#########################   start   ####################################')
        X = np.squeeze(X,axis=0)
        Y = np.squeeze(Y,axis=0)

        # 1、开始计时
        since = time.time()
        # 2、Net开始预测
        pr = model.predict( np.array([X]))[0]
        # 3、计算耗时
        seconds = time.time() - since
        y_scores= pr.reshape(-1,1)
        y_true =Y.reshape(-1,1)

        #1、画ROC曲线
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

        #6、save pre_pic
        save_name = output_path + str(Z[0])+'.png'
        save_pre_pic(pr, colors, save_name)

        print('predict time use {:.3f}s,save to {}'.format(seconds, save_name))
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

    # 4、程序结束退出，不加这个会报错,因为config文件设置了GPU模式，但是这里又不去使用它
    K.clear_session()

start_eval()
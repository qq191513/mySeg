#coding=utf-8
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score

from matplotlib import pyplot as plt
import pandas as pd
import os
import numpy as np
from collections import OrderedDict
def plot_roc_curve(y_true,y_scores,save_plot_curve_dir,curve_name):
    fpr, tpr, thresholds = roc_curve((y_true), y_scores)
    AUC_ROC = roc_auc_score(y_true, y_scores)
    print("Area under the ROC curve: " + str(AUC_ROC))
    roc_curve_figure = plt.figure()
    plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_plot_curve_dir,'ROC'+curve_name+".png"))
    plt.cla()
    plt.close("all")
    return AUC_ROC

def plot_precision_recall_curve(y_true, y_scores,save_plot_curve,curve_name):
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
    plt.savefig(os.path.join(save_plot_curve,'Precision_recall'+curve_name+'.png'))

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

def save_all_pics_value(name_list,all_list,save_list_csv):

    data = OrderedDict()
    # 1、保存所有图片的评估值
    for name, value in zip(name_list, all_list):
        data.update({name: value})
        # print(data)
    result = pd.DataFrame(data=data)
    result.to_csv(save_list_csv, encoding='gbk')
    print('save to {}'.format(save_list_csv))

def save_mean_value(name_list,all_list,save_mean_csv):

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
    data = OrderedDict()
    index = 1  # 只有一行 (为何不加这个index就会报错)
    for name, mean in zip(name_list, mean_list):
        data.update({name: mean})
    print('mean: ',data)
    mean_result = pd.DataFrame(data, index=[index])
    mean_result.to_csv(save_mean_csv, encoding='gbk')
    print('save to {}'.format(save_mean_csv))

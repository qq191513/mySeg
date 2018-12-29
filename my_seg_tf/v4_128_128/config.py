import os
##########################   训练集   #######################################
project_root ='/home/mo/work/seg_caps/my_seg_tf/v3'
mask=True
train_data_number = 1015
num_classes = 1
input_shape =[64,64,3]
labels_shape =[64,64,1]
num_epochs = 10000
########################        end      ########################################

##########################   训练配置   ########################################
batch_size = 3
########################        end      ########################################

##########################   输出   #######################################
ckpt_dir = os.path.join(project_root,'../output_result/segcap_hand_64_64/ckpt_dir')
predict_pics_save = os.path.join(project_root,'../output_result/segcap_hand_64_64/predict_pics')
save_mean_csv = os.path.join(project_root,'../output_result/segcap_hand_64_64/eval_result_mean.csv')
save_list_csv = os.path.join(project_root,'../output_result/segcap_hand_64_64/eval_result.csv')
save_plot_curve = os.path.join(project_root,'../output_result/segcap_hand_64_64/')
train_print_log =  os.path.join(project_root,'../output_result/segcap_hand_64_64/')
########################        end      ########################################

##########################   校验集   #######################################
test_data_number =100
########################        end      ########################################







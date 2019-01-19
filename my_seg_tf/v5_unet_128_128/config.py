import os
##########################   训练集   #######################################
project_root ='/home/mo/work/seg_caps/my_seg_tf/v5_unet_128_128'
mask=True
train_data_number = 57
num_classes = 1
input_shape =[128,128,3]
labels_shape =[128,128,1]
num_epochs = 2001
########################        end      ########################################

##########################   训练配置   ########################################
batch_size = 2
lr_init=0.001
########################        end      ########################################

##########################   输出   #######################################
ckpt_dir = os.path.join(project_root,'../output_result/unet_my_hand_128/ckpt_dir')
predict_pics_save = os.path.join(project_root,'../output_result/unet_my_hand_128/predict_pics')
save_mean_csv = os.path.join(project_root,'../output_result/unet_my_hand_128/eval_result_mean.csv')
save_list_csv = os.path.join(project_root,'../output_result/unet_my_hand_128/eval_result.csv')
save_plot_curve = os.path.join(project_root,'../output_result/unet_my_hand_128/')
train_print_log =  os.path.join(project_root,'../output_result/unet_my_hand_128/')
########################        end      ########################################

##########################   校验集   #######################################
test_data_number =9
########################        end      ########################################







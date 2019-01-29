import os
##########################   训练集   #######################################
mask=True
train_data_number = 57
num_classes = 1
batch_size = 2
input_shape =[batch_size,128,128,3]
labels_shape =[batch_size,128,128,1]
labels_shape_vec =[batch_size,128*128*1]

epoch = 100
save_epoch_n = 10  #每多少epoch保存一次
lr_range=(1e-3,1e-6,0.96)
test_data_number =9
# choose_loss = 'bce'
# choose_loss = 'dice'
# choose_loss = 'margin'
# choose_loss = 'focal'
# choose_loss = 'bce_dice'
# choose_loss = 'bce_dice_margin'
choose_loss = 'bce_dice_margin_focus'
# choose_loss = 'dice_margin_focus'
# choose_loss = 'bce_margin_focus'
# choose_loss = 'margin_focus'
########################        end      ########################################

##########################   输出路径   #######################################
project_root ='/home/mo/work/seg_caps/my_seg_tf/v10_segcap_128_128'
output_path = '/home/mo/work/output'
branch_name = 'v10_segcap_128_128'
model_name = 'res_segcap_mini_v2'
dataset_name = 'my_128'
########################        end      ########################################

#固定写法
ckpt =os.path.join(output_path,branch_name,model_name + '_' + dataset_name)
save_mean_csv = os.path.join(ckpt,'eval_result_mean.csv')
save_list_csv = os.path.join(ckpt,'eval_result.csv')
save_plot_curve_dir = os.path.join(ckpt,'plot_curve')
train_print_log =  os.path.join(ckpt)
logdir = os.path.join(ckpt,'logdir')
predict_pics_save =  os.path.join(ckpt,'predict_pics')
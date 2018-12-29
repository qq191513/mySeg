import os

mask=True
project_root ='/home/mo/work/seg_caps/my_seg_tf/v1'

ckpt_dir = os.path.join(project_root,'output_result/ckpt_dir')
num_classes = 1
input_shape =[64,64,3]
labels_shape =[64,64,1]
num_epochs = 100
batch_size = 3
train_data_number = 1015
test_data_number =100
predict_pics_save = os.path.join(project_root,'output_result/predict_pics_save')
save_mean_csv = os.path.join(project_root,'output_result/eval_result_mean.csv')
save_list_csv = os.path.join(project_root,'output_result/eval_result.csv')
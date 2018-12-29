#coding=utf-8

from keras.utils import print_summary

from data_process.use_generator import segcap_data_Generator
import config as cfg
from set_gpu import *

##########################   改这里   #######################################
#下面是生成器填写的东西
images_path =cfg.train_images
label_path =  cfg.train_label
gray_path = cfg.gray_label
batch_size = cfg.train_batch_size
n_classes =cfg.n_classes
input_height = cfg.input_shape[0]
input_width = cfg.input_shape[1]
output_height = cfg.output_shape[0]
output_width = cfg.output_shape[1]
save_history_plt = cfg.save_history_plt
train_weights_recover_path = cfg.train_weights_recover_path
steps_per_epoch = cfg.train_steps_per_epoch
##########################   end   #######################################
if cfg.debug:
    steps_per_epoch = 100
def train():
    os.makedirs(cfg.save_weights_path, exist_ok=True)

    #选择网络
    train_model, eval_model = cfg.select_model

    #编译和打印模型
    train_model.compile(optimizer=cfg.optimizer, loss=cfg.loss,metrics=cfg.metrics)
    print_summary(model=train_model)

    #恢复训练
    if train_weights_recover_path:
        train_model.load_weights(train_weights_recover_path)

    #训练数据生成器G1
    G1 = segcap_data_Generator(images_path , label_path ,gray_path,batch_size,
    n_classes , input_height ,input_width, output_height,output_width)
    #测试数据生成器G2
    if cfg.validate:
        G2 = segcap_data_Generator(images_path , label_path ,gray_path,batch_size,
    n_classes , input_height ,input_width, output_height,output_width)

    #循环训练
    for ep in range(cfg.epochs):
        #1、训练两种方式
        if not cfg.validate: #只有G1
            hisroy = train_model.fit_generator(
                G1,steps_per_epoch=steps_per_epoch,
                workers=cfg.workers,epochs=1,verbose=1,use_multiprocessing=cfg.use_multiprocessing)
        else: #有G1和G2
            hisroy = train_model.fit_generator(
                G1,steps_per_epoch=steps_per_epoch,
                workers=cfg.workers, epochs=1, verbose=1,use_multiprocessing=cfg.use_multiprocessing,
                validation_data = G2,validation_steps=cfg.validate_steps_per_epoch
              )

        # 2、画线
        # plot_training(hisroy, cfg.save_history_plt)

        # 3、保存模型
        if (ep % cfg.epochs_save) == (cfg.epochs_save-1):
            print('saving model.{}.......'.format(ep))
            save_weights_name = 'model.{}'.format(ep)
            save_weights_path = os.path.join(cfg.save_weights_path,save_weights_name)
            train_model.save_weights(save_weights_path)



train()
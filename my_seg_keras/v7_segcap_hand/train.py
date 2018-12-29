#coding=utf-8

from keras.utils import print_summary
from data_process.use_generator import imageSegmentationGenerator
import config as cfg
from set_gpu import *
if not os.path.exists(cfg.save_weights_path):
    os.makedirs(cfg.save_weights_path)

def train():

    model = cfg.load_model

    #编译和打印模型
    model.compile(optimizer=cfg.optimizer, loss=cfg.loss,metrics=cfg.metrics)
    print_summary(model=model)

    #训练数据生成器G1
    G1 = imageSegmentationGenerator(cfg.train_images, cfg.train_label, cfg.train_batch_size,
                               cfg.n_classes, cfg.input_shape[0], cfg.input_shape[1], cfg.output_shape[0],
                                    cfg.output_shape[1])
    #测试数据生成器G2
    if cfg.validate:
        G2 = imageSegmentationGenerator(cfg.val_images, cfg.train_label, cfg.val_batch_size,
                                        cfg.n_classes, cfg.input_shape[0], cfg.input_shape[1], cfg.output_shape[0],
                                        cfg.output_shape[1])
    #循环训练
    for ep in range(cfg.epochs):
        #1、训练两种方式
        if not cfg.validate: #只有G1
            hisroy = model.fit_generator(
                G1,steps_per_epoch=cfg.train_steps_per_epoch,
                workers=cfg.workers,epochs=1,verbose=1,use_multiprocessing=cfg.use_multiprocessing,
                class_weight='auto')
        else: #有G1和G2
            hisroy = model.fit_generator(
                G1,steps_per_epoch=cfg.train_steps_per_epoch,
                workers=cfg.workers, epochs=1, verbose=1,use_multiprocessing=cfg.use_multiprocessing,
                validation_data = G2,validation_steps=cfg.validate_steps_per_epoch
              )

        # 2、保存模型
        if (ep % cfg.epochs_save) == (cfg.epochs_save-1):
            print('saving model.{}.......'.format(ep))
            save_weights_name = 'model.{}'.format(ep)
            save_weights_path = os.path.join(cfg.save_weights_path,save_weights_name)
            model.save_weights(save_weights_path)



train()
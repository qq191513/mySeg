from models.unet import UNet
from keras.utils import print_summary
from use_generator import imageSegmentationGenerator
import config as cfg

#########################   使用GPU  动态申请显存占用 ####################
# 1、使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放内存，所以会导致碎片
# 2、visible_device_list指定使用的GPU设备号；
# 3、allow_soft_placement如果指定的设备不存在，允许TF自动分配设备（这个设置必须有，否则无论如何都会报cudnn不匹配的错误）
# 4、per_process_gpu_memory_fraction  指定每个可用GPU上的显存分配比
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
session_config = tf.ConfigProto(
            device_count={'GPU': 0},  #不能写成小写的gpu，否则无效
            gpu_options={'allow_growth': 1,
                # 'per_process_gpu_memory_fraction': 0.1,
                'visible_device_list': '0'},
                allow_soft_placement=True) #这个设置必须有，否则无论如何都会报cudnn不匹配的错误

sess = tf.Session(config=session_config)
KTF.set_session(sess)

#########################   END   ####################################

if not os.path.exists(cfg.save_weights_path):
    os.makedirs(cfg.save_weights_path)


def train():
    model = UNet(cfg.input_shape)

    #编译和打印模型
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    print_summary(model=model)

    #训练数据生成器G1
    G1 = imageSegmentationGenerator(cfg.train_images, cfg.train_annotations, cfg.train_batch_size,
                               cfg.n_classes, cfg.input_shape[0], cfg.input_shape[1], cfg.output_shape[0],
                                    cfg.output_shape[1])
    #测试数据生成器G2
    if cfg.validate:
        G2 = imageSegmentationGenerator(cfg.val_images, cfg.val_annotations, cfg.val_batch_size,
                                        cfg.n_classes, cfg.input_shape[0], cfg.input_shape[1], cfg.output_shape[0],
                                        cfg.output_shape[1])
    #循环训练
    for ep in range(1,cfg.epochs+1):
        #1、训练两种方式
        if not cfg.validate: #只有G1
            hisroy = model.fit_generator(
                G1,steps_per_epoch=cfg.train_steps_per_epoch,
                workers=cfg.workers,epochs=1,verbose=1,use_multiprocessing=cfg.use_multiprocessing)
        else: #有G1和G2
            hisroy = model.fit_generator(
                G1,steps_per_epoch=cfg.train_steps_per_epoch,
                workers=cfg.workers, epochs=1, verbose=1,use_multiprocessing=cfg.use_multiprocessing,
                validation_data = G2,validation_steps=cfg.validate_steps_per_epoch
              )

        # 2、保存模型
        if cfg.epochs % cfg.epochs_save ==0:
            save_weights_name = 'model.{}'.format(ep)
            save_weights_path = os.path.join(cfg.save_weights_path,save_weights_name)
            model.save_weights(save_weights_path)


train()
'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file contains the network definitions for the various capsule network architectures.
'''

from keras import layers
from keras import backend as K

K.set_image_data_format('channels_last')
import sys
sys.path.append('../')
from models.capsule_layers import ConvCapsuleLayer, Mask, Length
from keras.models import Model


def CapsNetBasic(input_shape, n_class=2):
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1',kernel_initializer='he_normal')(x)
    # conv1_1 = layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu', name='conv1_1')(x)

    # Reshape layer to be 1 capsule x [filters] atoms
    _, H, W, C = conv1.get_shape()
    conv1_reshaped = layers.Reshape((H.value, W.value, 1, C.value))(conv1)

    # Layer 1: Primary Capsule: Conv cap with routing 1
    primary_caps = ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=8, strides=1, padding='same',
                                    routings=1, name='primarycaps')(conv1_reshaped)

    # primary_caps = conv1_reshaped_2 +primary_caps
    # Layer 4: Convolutional Capsule: 1x1
    Convcap1 = ConvCapsuleLayer(kernel_size=1, num_capsule=1, num_atoms=2, strides=1, padding='same',
                                routings=3, name='Convcap1')(primary_caps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    out_seg = Length(num_classes=2, seg=True, name='out_seg')(Convcap1)

    # out_seg_1 = layers.Conv2D(filters=2, kernel_size=3, strides=1, padding='same', activation='relu', name='out_seg_1')(out_seg)

    # Decoder network.
    _, H, W, C, A = Convcap1.get_shape()
    x2 = layers.Input(shape=input_shape[:-1]+(n_class,))
    masked_by_y = Mask()([Convcap1,x2])
    masked = Mask()(Convcap1)

    def shared_decoder(mask_layer):
        recon_remove_dim = layers.Reshape((H.value, W.value, A.value))(mask_layer)

        recon_1 = layers.Conv2D(filters=64, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                activation='relu', name='recon_1')(recon_remove_dim)

        recon_2 = layers.Conv2D(filters=128, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                activation='relu', name='recon_2')(recon_1)

        out_recon = layers.Conv2D(filters=1, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                  activation='sigmoid', name='out_recon')(recon_2)

        return out_recon

    # Models for training and evaluation (prediction)
    train_model = Model(inputs=[x, x2], outputs=[out_seg, shared_decoder(masked_by_y)])
    eval_model = Model(inputs=x, outputs=[out_seg, shared_decoder(masked)])


    return train_model, eval_model

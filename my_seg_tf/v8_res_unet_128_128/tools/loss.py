#coding=utf-8
from keras.losses import binary_crossentropy
import tensorflow as tf

def get_loss(choice):
    if choice == 'bce':
        loss = 'binary_crossentropy'
    # elif choice == 'w_bce':
    #     pos_class_weight = load_class_weights(root=root, split=split)
    #     loss = weighted_binary_crossentropy_loss(pos_class_weight)

    elif choice == 'dice':
        loss = dice_loss
    # elif choice == 'w_mar':
    #     pos_class_weight = load_class_weights(root=root, split=split)
    #     loss = margin_loss(margin=0.4, downweight=0.5, pos_weight=pos_class_weight)
    # elif choice == 'mar':
    #     loss = margin_loss(margin=0.4, downweight=0.5, pos_weight=1.0)
    elif choice == 'bce_dice':
        loss = bce_dice_loss


    # if net.find('caps') != -1:
    #     return {'out_seg': loss, 'out_recon': 'mse'}, {'out_seg': 1., 'out_recon': recon_wei}
    # else:
    return loss


#dice_hard用于评估
def dice_hard(y_true, y_pred, threshold=0.5, axis=[1,2,3], smooth=1e-5):

    y_pred = tf.cast(y_pred > threshold, dtype=tf.float32)
    y_true = tf.cast(y_true > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(y_pred, y_true), axis=axis)
    l = tf.reduce_sum(y_pred, axis=axis)
    r = tf.reduce_sum(y_true, axis=axis)
    ## old axis=[0,1,2,3]
    # hard_dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # hard_dice = tf.clip_by_value(hard_dice, 0, 1.0-epsilon)
    ## new haodong
    hard_dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    hard_dice = tf.reduce_mean(hard_dice)
    return hard_dice

#dice_soft用于loss
def dice_soft(y_true, y_pred, loss_type='jaccard', axis=[1], smooth=1e-5, from_logits=False):

    if not from_logits:
        # transform back to logits
        _epsilon = tf.convert_to_tensor(1e-7, y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
        y_pred = tf.log(y_pred / (1 - y_pred))

    inse = tf.reduce_sum(y_pred * y_true, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(y_pred * y_pred, axis=axis)
        r = tf.reduce_sum(y_true * y_true, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(y_pred, axis=axis)
        r = tf.reduce_sum(y_true, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    ## old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    ## new haodong
    dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    dice = tf.reduce_mean(dice)
    return dice


def dice_loss(y_true, y_pred, from_logits=False):
    return 1-dice_soft(y_true, y_pred,axis=[1], from_logits=False)

def bce_dice_loss(y_true, y_pred,shape):
    y_true = tf.reshape(y_true,shape)
    y_pred = tf.reshape(y_pred,shape)
    batch_loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    mean_loss = tf.reduce_mean(tf.cast(batch_loss, tf.float32))
    return mean_loss






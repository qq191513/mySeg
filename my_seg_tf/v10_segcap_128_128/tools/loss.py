#coding=utf-8
from keras.losses import binary_crossentropy
import tensorflow as tf

def get_loss(choice):
    if choice == 'bce':
        loss = bce_loss
    elif choice == 'dice':
        loss = dice_loss
    elif choice == 'bce_dice':
        loss = bce_dice_loss
    elif choice == 'bce_dice_margin':
        loss = bce_dice_margin_loss
    elif choice == 'bce_dice_margin_focus':
        loss = bce_dice_margin_focus_loss
    elif choice =='dice_margin_focus':
        loss = dice_margin_focus_loss
    elif choice == 'bce_margin_focus':
        loss = bce_margin_focus_loss
    elif choice =='margin_focus':
        loss = margin_focus_loss
    elif choice =='margin':
        loss = margin_loss
    elif choice == 'focal':
        loss = focal_loss
    else:
        loss = None
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

def bce_loss(y_true, y_pred, shape):
    y_true = tf.reshape(y_true,shape)
    y_pred = tf.reshape(y_pred,shape)
    batch_loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    mean_loss = tf.reduce_mean(tf.cast(batch_loss, tf.float32))
    return mean_loss

def dice_loss(y_true, y_pred, from_logits=False):
    return 1-dice_soft(y_true, y_pred,axis=[1], from_logits=True)

def bce_dice_loss(y_true, y_pred,shape):
    y_true = tf.reshape(y_true,shape)
    y_pred = tf.reshape(y_pred,shape)
    batch_loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    mean_loss = tf.reduce_mean(tf.cast(batch_loss, tf.float32))
    return mean_loss

def use_margin_loss(v_lens,labels):
    class_loss = tf.reduce_mean(
        labels * tf.square(tf.maximum(0., 0.9 - v_lens)) +
        0.5 * (1. - labels) * tf.square(tf.maximum(0., v_lens - 0.1)))
    return class_loss

def bce_dice_margin_loss(y_true, y_pred,shape):
    y_true = tf.reshape(y_true,shape)
    y_pred = tf.reshape(y_pred,shape)
    batch_loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred) + use_margin_loss(v_lens=y_pred,labels=y_true)
    mean_loss = tf.reduce_mean(tf.cast(batch_loss, tf.float32))
    return mean_loss

def bce_dice_margin_focus_loss(y_true, y_pred,shape):
    y_true = tf.reshape(y_true,shape)
    y_pred = tf.reshape(y_pred,shape)
    weight_focus = 0.001
    batch_loss = \
    binary_crossentropy(y_true, y_pred) + \
    dice_loss(y_true, y_pred) + \
    use_margin_loss(v_lens=y_pred,labels=y_true) + \
    weight_focus*use_focal_loss(y_true, y_pred, axis=[1])
    mean_loss = tf.reduce_mean(tf.cast(batch_loss, tf.float32))
    return mean_loss

def bce_margin_focus_loss(y_true, y_pred,shape):
    y_true = tf.reshape(y_true,shape)
    y_pred = tf.reshape(y_pred,shape)
    weight_focus = 0.001
    batch_loss = \
    binary_crossentropy(y_true, y_pred) + \
    use_margin_loss(v_lens=y_pred,labels=y_true) + \
    weight_focus*use_focal_loss(y_true, y_pred, axis=[1])
    mean_loss = tf.reduce_mean(tf.cast(batch_loss, tf.float32))
    return mean_loss

def dice_margin_focus_loss(y_true, y_pred,shape):
    y_true = tf.reshape(y_true,shape)
    y_pred = tf.reshape(y_pred,shape)
    weight_focus = 0.001
    batch_loss = \
    dice_loss(y_true, y_pred) + \
    use_margin_loss(v_lens=y_pred,labels=y_true) + \
    weight_focus*use_focal_loss(y_true, y_pred, axis=[1])
    mean_loss = tf.reduce_mean(tf.cast(batch_loss, tf.float32))
    return mean_loss

def margin_focus_loss(y_true, y_pred,shape):
    y_true = tf.reshape(y_true,shape)
    y_pred = tf.reshape(y_pred,shape)
    weight_focus = 0.001
    batch_loss = \
    use_margin_loss(v_lens=y_pred,labels=y_true) + \
    weight_focus*use_focal_loss(y_true, y_pred, axis=[1])
    mean_loss = tf.reduce_mean(tf.cast(batch_loss, tf.float32))
    return mean_loss

def margin_loss(y_true, y_pred,shape):
    y_true = tf.reshape(y_true,shape)
    y_pred = tf.reshape(y_pred,shape)
    batch_loss = \
    use_margin_loss(v_lens=y_pred,labels=y_true)
    mean_loss = tf.reduce_mean(tf.cast(batch_loss, tf.float32))
    return mean_loss

def focal_loss(y_true, y_pred,shape):
    y_true = tf.reshape(y_true,shape)
    y_pred = tf.reshape(y_pred,shape)
    # y_pred = tf.clip_by_value(tf.sigmoid(y_pred), 1e-3, 1.0)
    batch_loss = \
    use_focal_loss(y_true, y_pred, axis=[1])
    mean_loss = tf.reduce_mean(tf.cast(batch_loss, tf.float32))
    return mean_loss

def use_focal_loss(y_true, y_pred,axis=[1]):
    # gamma = 0.75
    gamma = 2
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    pt_1 = tf.clip_by_value(pt_1, 1e-3, .999)
    pt_0 = tf.clip_by_value(pt_0, 1e-3, .999)

    return -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.log(pt_1),axis=axis) - tf.reduce_sum(
        (1 - alpha) * tf.pow(pt_0, gamma) * tf.log(1. - pt_0),axis=axis)


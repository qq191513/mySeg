import tensorflow as tf
import cv2
import os
def print_tensor(tensor,message=None):
	if message is None:
		message = 'Debug '
	return tf.Print(tensor, [tensor], message=message+': ', summarize=150)

def save_tensor_to_pics(sess,tensor,feed_dict,show,save_path=None,save_name=None):
	np_array = sess.run([tensor],feed_dict)
	if show:
		cv2.imshow('show',np_array)
		cv2.waitKey(1000)
	if save_path is not None:
		cv2.imwrite(os.path.join(save_path,save_name), np_array)


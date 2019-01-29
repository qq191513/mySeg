from train.train_segcap_or_unet import train_model
from eval.eval_segcap_or_unet.evaluate_by_sklearn import evaluate_by_sklearn_model
from eval.eval_segcap_or_unet.predict_and_save_pics import predict_and_save_pics_model

if __name__ == '__main__':
	option = '1'

	if option == '1':
		train_model()
	if option == '2':
		predict_and_save_pics_model()
	if option == '3':
		evaluate_by_sklearn_model()










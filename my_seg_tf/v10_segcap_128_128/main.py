if __name__ == '__main__':
	option = '2'

	if option == '1':
		from train.train_segcap_or_unet import train_model
		train_model()
	if option == '2':
		from eval.eval_segcap_or_unet.predict_and_save_pics import predict_and_save_pics_model
		predict_and_save_pics_model()
	if option == '3':
		from eval.eval_segcap_or_unet.evaluate_by_sklearn import evaluate_by_sklearn_model
		evaluate_by_sklearn_model()
	if option == '4':
		from eval.eval_segcap_or_unet.predict_and_save_one_tensor_feature_map import predict_and_save_tensor_feature_map_model
		predict_and_save_tensor_feature_map_model()








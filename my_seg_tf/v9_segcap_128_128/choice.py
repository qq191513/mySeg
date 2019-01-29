#选择模型配置器
def choice_cfg():
	# import config.config_res_segcap as cfg
	# import config.config_res_segcap_mini as cfg
	# import config.config_res_segcap_mini_v1 as cfg
	import config.config_res_segcap_my_final as cfg
	# import config.config_res_unet as cfg
	# import config.config_segcap as cfg
	# import config.config_unet as cfg
	return cfg

#选择模型
def choice_model():
	# from models.res_segcap import my_segcap as model
	# from models.res_segcap_mini import my_segcap as model
	from models.res_segcap_my_final import my_segcap as model
	# from models.res_segcap_mini_v1 import my_segcap as model
	# from models.res_unet import my_residual_unet as model
	# from models.segcap import my_segcap as model
	# from models.unet import my_unet as model
	return model




cfg = choice_cfg()
model = choice_model()







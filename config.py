class FLAGS():
# cmu model parameter
	model = 'cmu'
	resize = '368x256'
	w = 368
	h = 256
	resize_out_ratio = 4.0
# Path
	train_path = "./train/"
	test_path = "./test/"

	images = "./images/"
	model_path = "./models"
	lstm_model = "lstm"
# LSTM parameter
	D_LABEL = {
		0:"duck",
		1:"lion",
		2:"chicken",
		3:"frog",
		4:"kingkong",
		5:"bird",
		6:"seal",
		7:"tiger",
		8:"dog",
		9:"crab",
		10:"snake",

		11:"swim",
		12:"basketball",
		13:"baseball",
		14:"soccer"
	}
	'''
	LSTM input shape = (None, n_frames, n_input)
	'''
	n_input = 36  		# 관절 개수
	n_steps = 20
	n_frames = 20		# 연속 프레임 개수
	n_features = 36
	n_hiddens = 34
	n_outputs = 14		# 분류 class 개수

	threshold = 0.95
#updated for learning-rate decay
# calculated as: decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
	lr = 0.0025 #used if decaying_learning_rate set to False
	init_lr = 0.005
#	epochs = 1200
	batch_size = 80
	display_iter = batch_size*8
	decaying_learning_rate = True
	decay_rate = 0.96 #the base of the exponential in the decay
	decay_steps = 100000 #used in decay every 60000 steps with a base of 0.96
	lambda_loss_amount = 0.0015

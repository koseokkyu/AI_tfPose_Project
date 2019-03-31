class FLAGS():
	'''
	default
	'''
	model = 'cmu'
	resize = '368x256'
	w = 368
	h = 256
	resize_out_ratio = 4.0
#Path
	train_path = "train/"
	test_path = "test/"

	d_path = "./dynamic/"
	s_path = "./static/"

	images = "./images/"

	model_path = "./models/"
	static_lstm = model_path + "static_lstm/"
	dynamic_lstm = model_path + "dynamic_lstm/"

# LSTM variable
	D_LABEL = {
		"bird":0,
		"kingkong":1,
		"seal":2
	}
	S_LABEL = {
		"chicken":0,
		"duck":1,
		"frog":2,
		"lion":3
	}

	n_input = 36
	n_steps = 15
	n_frames = 15
	n_features = 36
	n_hiddens = 34
	n_outputs = 5

	threshold = 0.8
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

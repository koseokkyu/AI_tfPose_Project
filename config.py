class FLAGS():
	'''
	default
	'''
	model = 'cmu'
	resize = '256x224'
	resize_out_ratio = 4.0
#Path
	train_path = "train_data/"
	test_path = "test_data/"
# LSTM variable
	LABELS = ["1","2","3","4","5"] 
	n_frames = 1
	n_features = 36
	n_hiddens = 34
	n_outputs = 5
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

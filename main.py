from preprocess import *
from pose_estimation import *

e = create_estimator()
image = image_read('./new train/1/image 0.jpg')
humans = img_read_joint(image, e, 368, 256)
X = lstm_input_convert(humans)
for i in X:
    print(i)

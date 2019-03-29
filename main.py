from preprocess import *
from pose_estimation import *
import tensorflow as tf
from collections import deque
import itertools


img_q = deque()     # 원본이미지 담은 queue
joint_q = deque()   # 관절 좌표 담은 queue
motion_q = deque()  # 20frame씩 담은 queue

init_flag = True
Temp_joint = None
Cur_joint = None
def model(image, e) :
    global init_flag, Temp_joint, Cur_joint, joint_q
    Cur_joint = get_xy(img_read_joint(np.array(image), e, 368, 256)[0])
    if init_flag:
        if len(Cur_joint)!=36:
            print("your joints are not fully captured!")
            return
        Temp_joint = Cur_joint
        init_flag = False
        return

    Cur_joint = fill_na(Temp_joint, Cur_joint)  #결측치 채운뒤,
    Temp_joint = Cur_joint
    Cur_joint = dict_to_list(Cur_joint)
    Cur_joint = img_scaling(Cur_joint)

    joint_q.append(Cur_joint)   # 관절 queue에 추가

    if len(joint_q) >= FLAGS.n_frames:  # 특정 frame개수만큼 채워지면
        motion = list(itertools.islice(joint_q, FLAGS.n_frames))
        joint_q.popleft()

        '''
        motion을 3차원 numpy array로 변환 -> lstm으로 input
        '''

e = create_estimator()
while True:
    i = input(':')
    image = image_read('./new train/1/image300.jpg')
    model(image, e)

#humans = img_read_joint(image, e, 368, 256)
#dic = get_xy(humans)
#temp = humans[0]





#X = lstm_input_convert(humans)
#print(get_prediction(X))
'''
sess = tf.Session()
saver = tf.train.import_meta_graph("./models/lstm/lstm.meta")
saver.restore(sess, "./models/lstm/lstm")
graph = tf.get_default_graph()

pred = graph.get_tensor_by_name('softmax:0')
x = graph.get_tensor_by_name('Placeholder:0')

print(sess.run(pred, feed_dict={x:X}))
'''
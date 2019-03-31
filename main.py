from preprocess import *
from pose_estimation import *
import tensorflow as tf
from collections import deque
import itertools
import time

img_q = deque()     # 원본이미지 담은 queue
joint_q = deque()   # 관절 좌표 담은 queue
motion_q = deque()  # 20frame씩 담은 queue

init_flag = True
Temp_joint = None
Cur_joint = None
test_datas = None

sess = tf.Session()
saver = tf.train.import_meta_graph("./models/lstm/lstm.meta")
saver.restore(sess, "./models/lstm/")
graph = tf.get_default_graph()
g1 = tf.Graph()
g2 = tf.Graph()

static_pred = graph.get_tensor_by_name('softmax:0')
static_x = graph.get_tensor_by_name('Placeholder:0')

def model(image, e):
    t = time.time()
    global init_flag, Temp_joint, Cur_joint, joint_q, test_datas
    Cur_joint = get_xy(img_read_joint(np.array(image), e, 368, 256)[0])
    # human = img_read_joint(np.array(image), e, 368, 256)
    # draw_joint(image, human)
    # Cur_joint = get_xy(human)

    if init_flag:
        if len(Cur_joint) != 36:
            print("your joints are not fully captured!")
            return
        Temp_joint = Cur_joint
        init_flag = False
        return

    Cur_joint = fill_na(Temp_joint, Cur_joint)  # 결측치 채운뒤,
    Temp_joint = Cur_joint
    Cur_joint = dict_to_list(Cur_joint)
    Cur_joint = img_scaling(Cur_joint)

    joint_q.append(Cur_joint)   # 관절 queue에 추가

    if len(joint_q) >= FLAGS.n_frames:  # 특정 frame개수만큼 채워지면
        motion = np.array(list(itertools.islice(joint_q, FLAGS.n_frames)))
        motion = np.expand_dims(motion, axis=0)
        # test_datas = np.append(motion, axis=0)
        if test_datas is None:
            test_datas = motion
            print('none')
        else:
            print('not noe')
            test_datas = np.concatenate((test_datas, motion))
        joint_q.popleft()
        print(t-time.time())
        # print(motion.shape)
        print(test_datas.shape)
        
       # motion을 3차원 numpy array로 변환 -> lstm으로 input


e = create_estimator()
i = 10
while True:
    i += 1
    if i == 3000:
        break
    image = image_read('./train/drum/image%s.jpg' % i)
    model(image, e)

# humans = img_read_joint(image, e, 368, 256)
# dic = get_xy(humans)
# temp = humans[0]





# X = lstm_input_convert(humans)
# print(get_prediction(X))
'''
sess = tf.Session()
saver = tf.train.import_meta_graph("./models/lstm/lstm.meta")
saver.restore(sess, "./models/lstm/lstm")
graph = tf.get_default_graph()

pred = graph.get_tensor_by_name('softmax:0')
x = graph.get_tensor_by_name('Placeholder:0')

print(sess.run(pred, feed_dict={x:X}))
'''
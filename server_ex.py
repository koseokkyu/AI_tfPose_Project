import io
import socket
import struct
import os
import tensorflow as tf
import time

import numpy as np
from preprocess import *
from pose_estimation import *
from collections import deque
from PIL import Image
from get_model import ImportGraph
import itertools


#static_model = ImportGraph("./models/static_lstm/lstm")
#lstm_model = ImportGraph(os.path.join(FLAGS.model, FLAGS.lstm_model))
lstm_model = ImportGraph("./models/lstm_20/lstm_20")

joint_q = deque()   # 관절 좌표 담은 queue

init_flag = True
Temp_joint = None
Cur_joint = None

def model(image, e):
    global init_flag, Temp_joint, Cur_joint, joint_q

    humans = img_read_joint(np.array(image), e, 368, 256)
    if len(humans) > 0:
        Cur_joint = get_xy(humans[0])
    else:
        return
    if init_flag:
        if len(Cur_joint)!=36:
            print("your joints are not fully captured!")
            return
        Temp_joint = Cur_joint
        init_flag = False
        return

    '''
    이전프레임으로 결측치 채운 뒤, list로 변환. 
    '''
    Cur_joint = fill_na(Temp_joint, Cur_joint)
    Temp_joint = Cur_joint
    Cur_joint = dict_to_list(Cur_joint)
    Cur_joint = img_scaling(Cur_joint)

    X = np.array(Cur_joint)
    joint_q.append(X)   # 관절 queue에 추가

    if len(joint_q) >= FLAGS.n_frames:  # 특정 frame개수만큼 채워지면
        motion = list(itertools.islice(joint_q, FLAGS.n_frames))
        motion = np.expand_dims(motion, axis=0)
        joint_q.popleft()
        try:
            prob = lstm_model.run(motion)
            if np.max(prob) > FLAGS.threshold:
                print("dynamic: ", FLAGS.D_LABEL[np.argmax(prob)])
                return np.argmax(prob)
        except ValueError:
            None


def main() :
    e = create_estimator()
    IP = ""
    CAPTURE_PORT = 8100
    LABEL_PORT = 8000

    #def send_label(label, socket) :
    #    socket.send(label.encode())


    # Start a socket listening for connections on 0.0.0.0:8000 (0.0.0.0 means
    # all interfaces)
    s_capture_socket = socket.socket()
    print("capture socket create")

    s_capture_socket.bind(("", CAPTURE_PORT))
    print("capture socket bind")

    s_capture_socket.listen(5)
    print("capture socket listen")


    s_label_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("label socket create")

    s_label_socket.bind(("", LABEL_PORT))
    print("label socket bind")

    s_label_socket.listen(5)
    print("label socket listen")


    num = 0

    while True :
        # Accept a single connection and make a file-like object out of it
        capture_connection = s_capture_socket.accept()[0].makefile('rb')
        label_connection = s_label_socket.accept()[0]

        try:
             while True:
                # Read the length of the image as a 32-bit unsigned int. If the
                # length is zero, quit the loop
                image_recv = capture_connection.read(struct.calcsize('<L'))
                if image_recv == 0 :
                    return

                image_len = struct.unpack('<L', image_recv)[0]

                if not image_len:
                    return

                # Construct a stream to hold the image data and read the image
                # data from the connection
                image_stream = io.BytesIO()
                image_stream.write(capture_connection.read(image_len))

                # Rewind the stream, open it as an image with PIL and do some
                # processing on it
                image_stream.seek(0)

                image = Image.open(image_stream)
                #model(image, e)
                #dataStr = "image%2s.jpg receive" % num
                # image.save("./images/dynamic/soccer/3/image%02s.jpg" % num)
                num += 1

                dataStr = str(model(image, e))

                #label_connection.send(dataStr.encode())

                #print('Image is verified')

        except KeyboardInterrupt :
            capture_connection.close()
            s_capture_socket.close()
            s_label_socket.close()

            return 0


if __name__ == "__main__" :
    main()

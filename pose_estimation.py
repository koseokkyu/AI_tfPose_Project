from config import FLAGS
import cv2
import numpy as np
from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from preprocess import get_xy, img_scaling

def create_estimator():
    '''

    :return: TfPoseEstimator 생성
    '''
    w, h = model_wh(FLAGS.resize)
    e = TfPoseEstimator(get_graph_path(FLAGS.model), target_size=(w, h))
    return e

def draw_joint(image, humans):
    '''
    img_read_joint를 이용해 읽은 관절을 원본 이미지에 덧씌움

    :param image: 원본 이미지
    :param humans: 관절 정보 담긴 human 객체
    :return:
    '''
    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    cv2.imshow(" ",image)
    cv2.waitKey(1)

def image_read(img_path):
    return common.read_imgfile(img_path, None, None)

def img_read_joint(image, e, w, h):
    '''
    이미지를 불러와 사람의 관절 검출

    :param image: 처리할 이미지
    :param e: create_estimator로 생성한 TfPoseEstimator
    :param w: width
    :param h: height
    :return: image, humans
    '''

    #image = common.read_imgfile(img_path, None, None)
    #image = cv2.imread(img_path)
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=FLAGS.resize_out_ratio)
    return humans

def lstm_input_convert(humans):
    '''

    :param humans:
    :return:
    '''
    human = get_xy(humans)
    lstm_input=[]
    for h in human:
        data = img_scaling(h)
        lstm_input.append(' '.join(str(e) for e in data) +'\n')
    text = string_to_3d_array(lstm_input)
    return text

def string_to_3d_array(lstm_input):
    temp=[]
    for e in lstm_input:
        n_features=len(e.split())
        temp+=[list(map(float,e.split()))]
    return np.array(temp).reshape(-1,1,n_features)
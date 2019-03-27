from sklearn.preprocessing import minmax_scale

import numpy as np


def img_scaling(point_list):
    '''

    :param point_list: (36 x 1) 관절 포인트 위치
    :return: 표준 규격에 맞춘 point_list
    '''

    '''
    X : point_list에서 x좌표 
    Y : point_list에서 y좌표
    '''
    X = []
    Y = []
    result = []
    point_list = point_list.split(',')
    for i in range(len(point_list)):
        if i % 2 == 0:
            X.append(float(point_list[i]))
        else:
            Y.append(float(point_list[i]))

    y_max = np.max(Y)
    y_min = np.min(Y)
    y_scale = minmax_scale(Y)  # minmax로 사람의 키를 통일
    x_mean = np.mean(X)
    x_scale = (X - x_mean) / (y_max - y_min) + 0.5  # x 비율을 키(y)와 동일하게 scale

    y_scale = y_scale * 255
    x_scale = x_scale * 255

    for i in range(len(x_scale)):
        result.append(x_scale[i])
        result.append(y_scale[i])
    return result


def get_xy(humans):
    '''

    :param humans: 이미지에 잡힌 모든 사람의 정제되지 않은 관절 포인트
    :return: (0_x, 0_y, 1_x, 1_y ....) 전처리를 거친 포인트 (lstm 모델 input에 맞춤)
    '''
    h_list = list(map(str,humans))
    result = []
    for hm in h_list:  #모든 사람에 대해
        #print(hm)
        temp = []
        hm2 = hm.split('BodyPart:')
        for hm3 in hm2:
            if hm3 == '':
                continue
            hm4 = hm3.split('-')[1].split('score=')[0].replace('(','').replace(')','').split(',')
            temp.append(str(float(hm4[0])*255) + ','+ str(float(hm4[1])*255))
        temp2 = str(','.join(temp))
        result.append(temp2)
    return result
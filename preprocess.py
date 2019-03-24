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
    for i in range(len(point_list)):
        if i % 2 == 0:
            X.append(float(point_list[i]))
        else:
            Y.append(float(point_list[i]))

    y_max = np.max(Y)
    y_min = np.min(Y)
    y_scale = minmax_scale(Y)    #minmax로 사람의 키를 통일
    x_mean = np.mean(X)
    x_scale = (X - x_mean) / (y_max - y_min)  #x 비율을 키(y)와 동일하게 scale

    for i in range(len(x_scale)):
        result.append(x_scale[i])
        result.append(y_scale[i])
    return result
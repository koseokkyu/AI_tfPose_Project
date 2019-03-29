from sklearn.preprocessing import minmax_scale

import numpy as np
import pandas as pd
from collections import OrderedDict

def get_xy(human):
    '''

    :param humans: Human 객체
    :return: x,y 좌표만 뽑은 dictionary
    '''
    dic= {}
    for i in human.body_parts:
        dic[str(i)+'_x'] = human.body_parts[i].x
        dic[str(i)+'_y'] = human.body_parts[i].y
    return dic

def fill_na(temp_dic,cur_dic):
    '''

    :param temp_hum: 이전 frame human joint
    :param cur_hum: 현재 frame human joint
    :return: 결측치를 채운 현재 관절
    '''
    #temp_dic=get_xy(temp_hum)
    #cur_dic = get_xy(cur_hum)
    for k in temp_dic.keys():
        if k not in cur_dic.keys():
            cur_dic[k]=temp_dic[k]
    return cur_dic

def dict_to_list(dict):
    '''

    :param dict: 결측치 채워진 human joint
    :return: dictionary를 list로 반환
    '''
    return list(OrderedDict(sorted(dict.items())).values())

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
    #point_list = point_list.split(',')
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


def train_get_bodyparts_xy(humans):
    result = pd.DataFrame()
    for h in humans:
        dic= {}
        for i in h.body_parts:
            dic[str(i)+'_x'] = h.body_parts[i].x
            dic[str(i)+'_y'] = h.body_parts[i].y
        result.append(dic)
    return result
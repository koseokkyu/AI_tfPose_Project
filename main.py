import cv2
import numpy as np
import time
import pandas as pd

from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import preprocess

train_data = pd.read_csv('./train_data/pose_person_train.csv')
test_data = pd.read_csv('./train_data/pose_person_test.csv')

print(train_data.columns)
print(train_data.shape)
print(test_data.shape)


train_data.drop(['Unnamed: 0', 'person'], axis = 1, inplace = True)
test_data.drop(['Unnamed: 0', 'person'], axis = 1, inplace = True)

train_data = train_data.dropna(axis=0)
test_data = test_data.dropna(axis=0)

print(train_data.shape)
print(test_data.shape)


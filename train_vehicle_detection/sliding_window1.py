import cv2
from ultils import *
from setting import win_size
params= load_classifier('ver1.p', path="/Users/datle/Desktop/plate_v4/train_vehicle_detection")
print(params)
test(params=params ,win_size=win_size)

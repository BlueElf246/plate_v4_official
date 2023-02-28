import cv2
from skimage.feature import hog
import skimage
import numpy as np
print(skimage.__version__)
img= cv2.imread("/Users/datle/Desktop/Official_license_plate/Training_vehicle_detection/result/middle_close.jpeg", cv2.IMREAD_COLOR)
img=cv2.resize(img, (494, 988))
h=[]
temp_img= img[:65,:65,:]
for x in range(3):
    feat= hog(temp_img[:,:,x], orientations=9, pixels_per_cell=(4,4), cells_per_block=(4,4), feature_vector=True)
    f1=hog(temp_img[:,:,x], orientations=9, pixels_per_cell=(4,4), cells_per_block=(4,4), feature_vector=False)
    print(f1.shape)
    print(len(feat))
    h.append(feat)

h= np.concatenate(h)
print(len(h))
total_ch= []
for x in range(3):
    img_total= hog(img[:,:,x], orientations=9, pixels_per_cell=(4,4), cells_per_block=(4,4), feature_vector=False)
    print(img_total.shape)
    total_ch.append(img_total)
ch1=total_ch[0]
ch2=total_ch[1]
ch3=total_ch[2]

hog_feat1=ch1[:13,:13].ravel()
hog_feat2=ch2[:13,:13].ravel()
hog_feat3=ch3[:13,:13].ravel()
fog_feat= np.hstack((hog_feat1,hog_feat2,hog_feat3))
print(fog_feat.shape)
# for x in range(58,70+2):
#     print(x)
#     img1=img[:x,:x,:]
#     print(img1.shape)
#     print(np.array(feat).shape)


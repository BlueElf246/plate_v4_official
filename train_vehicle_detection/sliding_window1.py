import cv2
from ultils import *
from setting import win_size
# from plate_detect.run1 import detect_plate
from plate_detect.run2 import detect_plate1
from plate_segmentation.run import model, character_extract
def pipeline(model_use, img):
    params= load_classifier(model_use, path="/Users/datle/Desktop/plate_v4/train_vehicle_detection")
    print(params)
    result=[]
    if type(img)!= str:
        result.append(detect(img, params, win_size))
    else:
        img_test= glob.glob(img)
        for x,i in enumerate(img_test):
            img= cv2.imread(i, cv2.IMREAD_COLOR)
            result.append(detect(img, params, win_size,x))
    return result
def detect(img_crop, params, win_size, x=0):
    result, bbox, img_crop = run(img_crop, params, win_size, debug=True)
    img_plate=[]
    for y, img_c in enumerate(img_crop):
        cv2.imwrite(f"/Users/datle/Desktop/plate_v4/img_car/{x}_{y}.png", img_c)
        plate=detect_plate1(img_c, show=False)
        if len(plate)==0:
            continue
        re, license_plate= character_extract(plate[0], show=False)
        cv2.imwrite("/Users/datle/Desktop/plate_v4/plate_segment_image/result.png", img=license_plate*255)
    return re


img= cv2.imread("/Users/datle/Desktop/plate_v4/test_pic/Cars108.png")
r=pipeline(model_use='ver1.p',img=img)
print(r)
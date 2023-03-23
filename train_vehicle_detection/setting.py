import numpy as np

params = {}
params['model_name']='svc_nu'
params['color_space'] = 'yuv'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
params['orient'] = 8  # HOG orientations
params['pix_per_cell'] = 4 # HOG pixels per cell
params['cell_per_block'] = 4  # HOG cells per block
params['size_of_window']=(64,64,3)
params['test_size']=0.2

#2.06 =10
win_size={}
s=[0.45, 0.611, 0.772, 0.933, 1.094, 1.255, 1.416, 1.577, 1.738, 1.899, 2.06, 2.2, 2.5, 2.75,3]
inx= np.argmax(s)
for x,y in enumerate(s):
    win_size[f'scale_{x}']=(0,1000,y)
win_size['thresh']=20
#10
win_size['overlap_thresh']= 0.1
win_size['length']= len(s)
# 0.5, 1.3
win_size['use_scale']=(len(s)-1,)
win_size['y_start_stop']= [None,None]
# 9,10
# close: use scale 2, 3, 4, 6, 7(good), 8, 9(good), 10 => 7,8,9
#  when  scale 5: can not detect upper part of car (close)
# far: scale 1,2,3, (6,7,8,9)
# 5,6,7,8,9,10
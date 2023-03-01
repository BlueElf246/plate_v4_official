import cv2
import matplotlib.pyplot as plt
import easyocr
import imutils
import numpy as np
def plate(img, mask, locations, show):

    cv2.drawContours(mask, [locations], 0, 255, -1)
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_img = img[x1:x2 + 1, y1:y2 + 1]
    plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    plt.show()
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_img)
    if len(result)==0:
        return None
    text= result[0][-2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text=text, org=(locations[0][0][0], locations[1][0][1] + 60), fontFace=font, fontScale=1,
                      color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    res = cv2.rectangle(img, tuple(locations[0][0]), tuple(locations[2][0]), (0, 255, 0), 3)
    img_result=res
    cv2.imwrite(f'/Users/datle/Desktop/plate_v4/result_plate/result_for_pic.png', img=img_result)
    if show==True:
        plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
        plt.show()
    return text,img_result
def detect_plate(img, show=False):
    # img = cv2.imread("Screen Shot 2023-02-28 at 23.49.10.png", cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # noise reduction
    edged = cv2.Canny(bfilter, 170, 200)
    plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
    plt.show()
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    locations = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            locations.append(approx)
    mask = np.zeros(gray.shape, np.uint8)
    for loc in locations:
        if plate(img,mask, loc, show) is None:
            continue
        else:
            text, img= plate(img,mask, loc, show)
            return text, img


# img= cv2.imread("/Users/datle/Desktop/plate_v4/img_car/0_0.png", cv2.IMREAD_COLOR)
# detect_plate(img, True)
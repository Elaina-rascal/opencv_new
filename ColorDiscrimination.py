import cv2 as cv
import numpy as np
#import matplotlib.pyplot as plt
#from imutils import contours


def color_discrimination(frame):        #颜色识别函数
    frame_gsblur=cv.GaussianBlur(frame,(5,5),0)
    frame_hsv=cv.cvtColor(frame_gsblur,cv.COLOR_BGR2HSV)

    red_mask_0=cv.inRange(frame_hsv,lower_red_0,upper_red_0)
    red_mask_1=cv.inRange(frame_hsv,lower_red_1,upper_red_1)
    blue_mask_0=cv.inRange(frame_hsv,lower_blue_0,upper_blue_0)
    yellow_mask_0=cv.inRange(frame_hsv,lower_yellow_0,upper_yellow_0)
    green_mask_0=cv.inRange(frame_hsv,lower_green_0,upper_green_0)
    purple_mask_0=cv.inRange(frame_hsv,lower_purple_0,upper_purple_0)

    kernel = np.ones((3,3), np.uint8)
    red_mask_0 = cv.morphologyEx(red_mask_0,cv.MORPH_OPEN, kernel)
    red_mask_1 = cv.morphologyEx(red_mask_1,cv.MORPH_OPEN, kernel)
    blue_mask_0 = cv.morphologyEx(blue_mask_0,cv.MORPH_OPEN, kernel)
    green_mask_0 = cv.morphologyEx(green_mask_0,cv.MORPH_OPEN, kernel)
    yellow_mask_0 = cv.morphologyEx(yellow_mask_0,cv.MORPH_OPEN,kernel)
    purple_mask_0 = cv.morphologyEx(purple_mask_0,cv.MORPH_OPEN,kernel)

    contours, _=cv.findContours(red_mask_0+red_mask_1+blue_mask_0+green_mask_0+yellow_mask_0+purple_mask_0,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x,y,w,h=cv.boundingRect(contour)
        color=""
        if cv.contourArea(contour)>2000:
            if np.any(red_mask_0[y:y + h, x:x + w]):
                color = "red"
                print(color)
                print('center_x',x+w/2)
                print('center_y',y+h/2)
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            elif np.any(red_mask_1[y:y + h, x:x + w]):
                color = "red"
                print(color)
                print('center_x', x + w / 2)
                print('center_y', y + h / 2)
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            elif np.any(blue_mask_0[y:y + h, x:x + w]):
                color = "blue"
                print(color)
                print('center_x', x + w / 2)
                print('center_y', y + h / 2)
                cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            elif np.any(green_mask_0[y:y + h, x:x + w]):
                color = "green"
                print(color)
                print('center_x', x + w / 2)
                print('center_y', y + h / 2)
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            elif np.any(yellow_mask_0[y:y + h, x:x + w]):
                color = "yellow"
                print(color)
                print('center_x', x + w / 2)
                print('center_y', y + h / 2)
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            elif np.any(purple_mask_0[y:y + h, x:x + w]):
                color = "purple"
                print(color)
                print('center_x', x + w / 2)
                print('center_y', y + h / 2)
                cv.rectangle(frame, (x, y), (x + w, y + h), (128, 0, 128), 2)
            cv.putText(frame, color, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return frame

#主函数

lower_red_0 = np.array([0, 100, 100])
upper_red_0 = np.array([6, 255, 255])
lower_red_1 = np.array([160, 100, 100])
upper_red_1 = np.array([170, 255, 255])
lower_blue_0 = np.array([110, 60, 60])
upper_blue_0 = np.array([120, 255, 255])
lower_yellow_0 = np.array([25, 100, 100])
upper_yellow_0 = np.array([35, 255, 255])
lower_green_0 = np.array([45, 150, 150])
upper_green_0 = np.array([85, 255, 255])
lower_purple_0 = np.array([130, 100, 100])
upper_purple_0 = np.array([150, 255, 255])

cap=cv.VideoCapture("image/VID_20240922_171544.mp4")
cap.set(cv.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,720)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame=cv.resize(frame,(1280,720))
        result = color_discrimination(frame)
        cv.imshow("Color Discrimination", result)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()
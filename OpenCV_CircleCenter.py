import cv2
import matplotlib.pyplot as plt
import numpy as np

cv2.namedWindow("result",cv2.WINDOW_FREERATIO)
cv2.resizeWindow("result",1920,1080)
vc = cv2.VideoCapture("OpenCVTry.mp4")
if vc.isOpened():
    ifopen,frame = vc.read()
else:
    ifopen = False
    print("error")

pre = []

while ifopen:
    ret,frame = vc.read()
    if frame is None:
        break
    if ret == True:
        frame_copy = frame.copy()

        #图像处理
        b,g,r = cv2.split(frame)
        b[:] = 0
        frame = cv2.merge((b,g,r))
        cv2.imshow('result',frame)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame = cv2.medianBlur(frame,5)                     #中值滤波
        frame = cv2.GaussianBlur(frame,(3,3),sigmaX = 1)    #高斯滤波
        frame = cv2.Canny(frame,2,250)                      #Cannny边缘检测
        tmp,frame = cv2.threshold(frame,10,255,cv2.THRESH_BINARY)
        #膨胀
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        frame = cv2.dilate(frame,kernel,iterations = 5)
        contours,hierarchy = cv2.findContours(frame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        if len(contours) > 0:
            circle = [0,0,0]
            Num = 0
            contoursNew = np.empty(0,dtype=np.int32)

            #筛选有用点
            for contour in range(len(contours)):
                for pointNum in range(len(contours[contour])):
                    if len(pre) == 0:
                        if len(contoursNew) == 0:
                            contoursNew = np.append(contoursNew,contours[contour][pointNum])
                            contoursNew = contoursNew[np.newaxis,np.newaxis,:]
                        else:
                            contoursNew = np.vstack((contoursNew,contours[contour][pointNum][np.newaxis,:]))
                    else:
                        des = abs(((contours[contour][pointNum][0][0]-pre[0])**2+(contours[contour][pointNum][0][1]-pre[1])**2)**0.5-pre[2])
                        if des > 100:
                            pass
                        else:
                            if len(contoursNew) == 0:
                                contoursNew = np.append(contoursNew,contours[contour][pointNum])
                                contoursNew = contoursNew[np.newaxis,np.newaxis,:]
                            else:
                                contoursNew = np.vstack((contoursNew,contours[contour][pointNum][np.newaxis,:]))

            #找最小圆
            if len(contoursNew) != 0:
                ((x,y),radius) = cv2.minEnclosingCircle(contoursNew)
                circle = [int(x),int(y),int(radius)]
                cv2.circle(frame_copy,(circle[0],circle[1]),circle[2],(0,0,255),2)
                cv2.circle(frame_copy,(circle[0],circle[1]),10,(0,0,255),-1)
                cv2.putText(frame_copy,f"{int(x)},{int(y)}",(circle[0]-20,circle[1]-20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
                pre = circle
        cv2.imshow('result',frame_copy)
        if cv2.waitKey(10) == 27:
            break
vc.release()
cv2.destroyAllWindows()

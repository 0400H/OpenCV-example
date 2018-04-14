#!/usr/bin/env python
# coding=utf-8

import cv2

cap = cv2.VideoCapture("./Apink - Remember.mkv")               #读取视频文件
fps = cap.get(cv2.CAP_PROP_FPS)         #fps获取视频的帧数
#size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))   #size获取视频尺寸
#fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')            #fourcc是标识视频数据流格式的四字符代码
#video = cv2.VideoWriter(r"./Face.mkv", fourcc, fps, size)      #视频保存路径
print (cap.isOpened())

classifier = cv2.CascadeClassifier(r"./haarcascade_frontalface_alt.xml")

count = 0
while count > -1:
    ret, img = cap.read()
    faceRects = classifier.detectMultiScale(img, 1.2, 2, cv2.CASCADE_SCALE_IMAGE, (20, 20))

    for x, y, w, h in faceRects:
        cv2.rectangle(img, (int(x), int(y)), (int(x) + int(w), int(y) + int(h)), (0, 255, 0), 2, 0)

    cv2.putText(img, "FPS:" + str(fps), (50, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
    cv2.namedWindow('Video Face Recognition')
    cv2.imshow('Video Face Recognition', img)
    # video.write(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#video.release()
cap.release()
cv2.destroyAllWindows()



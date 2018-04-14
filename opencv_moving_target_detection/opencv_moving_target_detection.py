#!usr/bin/env python
# coding=utf-8

import cv2

camera = cv2.VideoCapture("./Apink - Remember.mkv")
width = int(camera.get(1))
height = int(camera.get(1))

firstFrame = None

while True:
    (grabbed, frame) = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if firstFrame is None:
        firstFrame = gray
        continue

    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    # thresh = cv2.adaptiveThreshold(frameDelta,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    # cv2.THRESH_BINARY,11,2)
    # thresh = cv2.adaptiveThreshold(frameDelta,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    #           cv2.THRESH_BINARY,11,2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) < 10000:
            continue
        (x, y, w, h) = cv2.boundingRect(c)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Security Feed", frame)
    firstFrame = gray.copy()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

import cv2
from time import sleep

cascPath = "./haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)               #打开视频捕获设备
fps = video_capture.get(cv2.CAP_PROP_FPS)         #fps获取视频的帧数
size = (                                          #size获取视频尺寸
    int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')          #fourcc是标识视频数据流格式的四字符代码
video = cv2.VideoWriter(r"./opencv_camera_face_recognition.mkv", fourcc, 5, size)


index = 0
flag = 0
while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass


    ret, frame = video_capture.read()       # 读视频帧

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)     # 转为灰度图像

    faces = faceCascade.detectMultiScale(    # 调用分类器进行检测
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # 画矩形框  
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(frame, "FPS:"+str(fps), (50, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
    cv2.namedWindow('Camera Face Recognition')
    cv2.imshow('Camera Face Recognition', frame)              # 显示视频

    if cv2.waitKey(1) & 0xFF == 27:  #ESC exit
        break
    if cv2.waitKey(1) & 0xFF == ord('p'):  # 保存截图
        cv2.imwrite(str(index)+".jpg", frame)
        index = index + 1
    if cv2.waitKey(1) & 0xFF == ord('v'):  # 视频录制
        video.write(frame)


#video.release()           # 关闭摄像
video_capture.release()   # 关闭摄像头设备
cv2.destroyAllWindows()   # 关闭所有窗口
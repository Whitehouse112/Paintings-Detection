import cv2
import numpy as np


video_name =  ''
video = cv2.VideoCapture(video_name)
while video.isOpened():
    ret, frame = video.read()
    if ret is False:
        break
        
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(im, (5, 5), 0)
    thr = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 3)
    kernel = np.ones((2,2), np.uint8)
    clean = cv2.erode(thr, kernel, iterations=2)
    thr = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, (4, 4), iterations=1)
    #ret, thr = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #thr = cv2.Canny(thr, 10, 2000, apertureSize=5, L2gradient=True)
    
    #contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(im, contours, -1, (255,0,0), 1)
    
    result = np.concatenate((im, thr), axis=0)
    
    cv2.imshow('Frame', cv2.resize(result, (720,960)))

    if cv2.waitKey(1) == ord('q'):
            break
video.release()
cv2.destroyAllWindows()
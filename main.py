import numpy as np
import cv2
from painting_detection import detect_paintings, HW3, init
from crop import cropROI


video_name = 'VIRB0399.MP4'
video = cv2.VideoCapture('videos/' + video_name)
if not video.isOpened():
    print("File not found.")
init()

while video.grab():
    _, frame = video.retrieve()
    frame = np.swapaxes(np.swapaxes(frame, 0, 2), 1, 2)  # (3, H, W)
    print("Frame", int(video.get(cv2.CAP_PROP_POS_FRAMES)) - 1)

    roi_list, roi_frame = detect_paintings(frame)
    images = cropROI(frame, roi_list)

    for img in images:
        cv2.imshow('First', img)
    
    # Show results
    print("ROI list:", roi_list, '\n')
    cv2.imshow('ROI', cv2.resize(HW3(roi_frame), (1280, 720)))

    # Delay & escape-key
    video.set(cv2.CAP_PROP_POS_FRAMES, int(video.get(cv2.CAP_PROP_POS_FRAMES)) + int(video.get(cv2.CAP_PROP_FPS) / 2))
    if cv2.waitKey(50) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

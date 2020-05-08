import numpy as np
import cv2
from painting_detection import detect_paintings, init_histogram
from painting_rectification import rectify_paintings
from utility import draw, load_video, HW3
# from crop import cropROI


video_name = 'VIRB0395.MP4'
video = load_video(video_name)

init_histogram()

while video.grab():
    _, frame = video.retrieve()
    frame = np.swapaxes(np.swapaxes(frame, 0, 2), 1, 2)  # (3, H, W)
    print("Frame", int(video.get(cv2.CAP_PROP_POS_FRAMES)) - 1)

    roi_list, cont_list = detect_paintings(np.array(frame))
    poly_list = rectify_paintings(cont_list, np.array(frame))
    # poly_list = rectify_paintings(cont_list)
    # paintings = cropROI(HW3(frame), roi_list)
    
    # Show results
    print("ROI list:", roi_list, '\n')
    # draw(cont_list, roi_list, poly_list, np.array(frame))
    # for painting in paintings:
    #     cv2.imshow('Segmentation', painting)

    # Delay & escape-key
    video.set(cv2.CAP_PROP_POS_FRAMES, int(video.get(cv2.CAP_PROP_POS_FRAMES)) + int(video.get(cv2.CAP_PROP_FPS)))
    if cv2.waitKey(50) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

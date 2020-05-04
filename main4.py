import numpy as np
import cv2
from painting_detection4 import detect_paintings, HW3


def load_video(video_name):
    video = cv2.VideoCapture('videos/' + video_name)
    if not video.isOpened():
        print("File not found.")
    return video


def get_frame(video):
    ret, frame = video.read()
    if ret is False:
        return None
    frame = np.swapaxes(np.swapaxes(frame, 0, 2), 1, 2)  # (3, H, W)
    video.set(cv2.CAP_PROP_POS_FRAMES, int(video.get(cv2.CAP_PROP_POS_FRAMES)) + int(video.get(cv2.CAP_PROP_FPS)) / 2)
    return frame


video1 = load_video('VIRB0391.MP4')
video2 = load_video('VIRB0392.MP4')
video3 = load_video('VIRB0393.MP4')
video4 = load_video('VIRB0395.MP4')

frame_entr = 15
frame_entr1 = frame_entr
frame_entr2 = frame_entr
frame_entr3 = frame_entr
frame_entr4 = frame_entr

while True:
    frame1 = get_frame(video1)
    frame2 = get_frame(video2)
    frame3 = get_frame(video3)
    frame4 = get_frame(video4)
    if frame1 is None or frame2 is None or frame3 is None or frame4 is None:
        break
    print("Frame", int(video1.get(cv2.CAP_PROP_POS_FRAMES)) - 1)

    roi_list1, roi_frame1, frame_entr1 = detect_paintings(frame1, frame_entr1)
    roi_list2, roi_frame2, frame_entr2 = detect_paintings(frame2, frame_entr2)
    roi_list3, roi_frame3, frame_entr3 = detect_paintings(frame3, frame_entr3)
    roi_list4, roi_frame4, frame_entr4 = detect_paintings(frame4, frame_entr4)

    # Show results
    print("ROI list1:", roi_list1)
    print("ROI list2:", roi_list2)
    print("ROI list3:", roi_list3)
    print("ROI list4:", roi_list4, '\n')
    roi_frame1 = cv2.resize(HW3(roi_frame1), (1280, 720))
    roi_frame2 = cv2.resize(HW3(roi_frame2), (1280, 720))
    roi_frame3 = cv2.resize(HW3(roi_frame3), (1280, 720))
    roi_frame4 = cv2.resize(HW3(roi_frame4), (1280, 720))
    horizontal_concat_1 = np.concatenate((roi_frame1, roi_frame2), axis=1)
    horizontal_concat_2 = np.concatenate((roi_frame3, roi_frame4), axis=1)
    vertical_concat = np.concatenate((horizontal_concat_1, horizontal_concat_2), axis=0)
    cv2.imshow('ROI', cv2.resize(vertical_concat, (1280, 720)))

    # Delay & escape-key
    if cv2.waitKey(1) == ord('q'):
        break

video1.release()
video2.release()
video3.release()
video4.release()
cv2.destroyAllWindows()

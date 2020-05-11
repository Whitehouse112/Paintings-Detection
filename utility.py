import numpy as np
import cv2


def hw3(image):
    return np.swapaxes(np.swapaxes(image, 0, 1), 1, 2)


def load_video(video_name):
    path = 'videos/'
    video = cv2.VideoCapture(path + video_name)
    if not video.isOpened():
        print("File", path + video_name, "not found.")
        exit(1)
    return video


def draw(cont_list, roi_list, poly_list, frame):
    img_contours = np.zeros_like(frame)
    cv2.drawContours(hw3(img_contours), cont_list, -1, (0, 255, 0), thickness=2)

    roi_frame = np.array(frame)
    for rect in roi_list:
        cv2.rectangle(hw3(roi_frame), (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0),
                      thickness=2)

    img_poly = np.zeros_like(frame)
    cv2.drawContours(hw3(img_poly), poly_list, -1, (0, 255, 0), thickness=2)

    poly_frame = np.array(frame)
    cv2.drawContours(hw3(poly_frame), poly_list, -1, (0, 255, 0), thickness=2)

    horizontal_concat_1 = np.concatenate((img_contours, roi_frame), axis=2)
    horizontal_concat_2 = np.concatenate((img_poly, poly_frame), axis=2)
    vertical_concat = np.concatenate((horizontal_concat_1, horizontal_concat_2), axis=1)

    cv2.imshow('Results', cv2.resize(hw3(vertical_concat), (1280, 720)))

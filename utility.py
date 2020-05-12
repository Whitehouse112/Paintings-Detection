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


def draw(roi_list, paintings, frame):
    roi_frame = np.array(frame)
    for rect in roi_list:
        cv2.rectangle(hw3(roi_frame), (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0),
                      thickness=2)

    # horizontal_concat_1 = np.concatenate((img_contours, roi_frame), axis=2)
    # horizontal_concat_2 = np.concatenate((img_poly, poly_frame), axis=2)
    # vertical_concat = np.concatenate((horizontal_concat_1, horizontal_concat_2), axis=1)
    # cv2.imshow('Results', cv2.resize(hw3(vertical_concat), (1280, 720)))

    cv2.imshow('Results', cv2.resize(hw3(roi_frame), (1280, 720)))

    for painting in paintings:
        cv2.imshow("Rectified", painting)


# def segmentation(frame, roi_list):
#     paintings = []
#
#     for roi in roi_list:
#         x, y, w, h = roi
#         painting = np.array(frame[:, y:y + h, x:x + w])
#         gray = cv2.cvtColor(hw3(painting), cv2.COLOR_RGB2GRAY)
#         _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#         # thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3)
#
#         # thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, (3, 3), iterations=1)
#
#         contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         hull = cv2.convexHull(max(contours, key=len))
#         # cv2.drawContours(HW3(painting), [hull], -1, (0, 255, 0), thickness=2)
#
#         img_hull = np.zeros_like(painting)
#         cv2.drawContours(hw3(img_hull), [hull], -1, (0, 255, 0), thickness=cv2.FILLED)
#         img_hull = cv2.cvtColor(hw3(img_hull), cv2.COLOR_RGB2GRAY)
#         painting *= np.uint8(img_hull > 0)
#
#         paintings.append(painting)
#
#     return paintings

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

    cv2.imshow('Painting Detection', cv2.resize(hw3(roi_frame), (1280, 720)))

    small_paintings = []
    for painting in paintings:
        h, w = painting.shape[0:2]
        if h > w:
            wide = h
            border = int((h - w) / 2)
            vert = True
        else:
            wide = w
            border = int((w - h) / 2)
            vert = False
        small = np.zeros((wide, wide, 3), dtype=np.uint8)
        if vert:
            if border * 2 != np.abs(h - w):
                small[:, border:wide - border - 1] = painting
            else:
                small[:, border:wide - border] = painting
        else:
            if border * 2 != np.abs(h - w):
                small[border:wide - border - 1] = painting
            else:
                small[border:wide - border] = painting
        size = int(1280 / 4)
        small = cv2.resize(small, (size, size))
        small_paintings.append(small)

    cv2.imshow("Painting Rectification", np.concatenate(small_paintings, axis=1))


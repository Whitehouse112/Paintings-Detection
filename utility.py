import numpy as np
import cv2
from matplotlib import pyplot as plt


def load_video(video_name):
    path = 'videos/'
    video = cv2.VideoCapture(path + video_name)
    if not video.isOpened():
        print("File", path + video_name, "not found.")
        exit(1)
    return video


def resize_images(paintings):
    small_paintings = []
    for painting in paintings:
        if len(painting.shape) == 2:
            painting = cv2.cvtColor(painting, cv2.COLOR_GRAY2RGB)
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
    return small_paintings


def draw(roi_list, cont_list, paintings, retrieved, frame):
    roi_frame = np.array(frame)
    for rect in roi_list:
        x, y, w, h = rect
        cv2.rectangle(roi_frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

    mask = np.zeros_like(frame)
    for cont in cont_list:
        cv2.drawContours(mask, [cont], -1, (255, 255, 255), thickness=cv2.FILLED)
    segm_frame = np.uint8(mask == 255) * frame

    vertical_concat = np.concatenate((roi_frame, segm_frame), axis=0)
    cv2.namedWindow("Painting Detection", flags=cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    cv2.imshow("Painting Detection", cv2.resize(vertical_concat, (int(1600 / 2), 900)))

    small_paintings = resize_images(paintings)
    if len(small_paintings) > 0:
        cv2.namedWindow("Painting Rectification",
                        flags=cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        cv2.imshow("Painting Rectification", np.concatenate(small_paintings, axis=1))

    # small_retrieved = resize_images(retrieved)
    # if len(small_retrieved) > 0:
    #     cv2.namedWindow("Retrieved paintings", flags=cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    #     cv2.imshow("Retrieved paintings", np.concatenate(small_retrieved, axis=1))


def plot_f_histogram(f_list):
    plt.hist(f_list, 20, [0, 2000])
    plt.show()


def skip_frames(video, fps=1):
    video.set(cv2.CAP_PROP_POS_FRAMES, int(video.get(cv2.CAP_PROP_POS_FRAMES)) + int(video.get(cv2.CAP_PROP_FPS)) / fps)
    return video

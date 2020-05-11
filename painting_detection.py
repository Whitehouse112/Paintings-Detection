import numpy as np
import cv2
import os
from utility import hw3


base_hist = None


def compute_histogram(img, stripes=10):
    mask = np.zeros((img.shape[1], img.shape[2]), dtype=np.uint8)

    every = img.shape[1] // (stripes + 1)
    for i in range(1, stripes + 1):
        mask[i * every, :] = 255

    hist = cv2.calcHist([hw3(img)], [0, 1, 2], mask, [32, 32, 32], [0, 255, 0, 255, 0, 255])
    return hist


def init_histogram():
    global base_hist

    n = len([name for name in os.listdir('photos/')])
    for i in range(1, n + 1):
        img = cv2.imread("photos/" + str(i) + ".png")
        img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)  # (3, H, W)

        hist = compute_histogram(img)

        if i == 1:
            base_hist = hist
        else:
            base_hist += hist
    base_hist /= n


def edge_detection(img):
    """Edge detection with Sobel"""

    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    edges = np.uint8(mag > 15) * 255
    return edges


def check_dims(_w, _h):
    if _w < 100 or _h < 100:
        return False
    if _w >= 1500 or _h > 1080:
        return False
    if _w / _h > 3.5 or _h / _w > 3.5:
        return False
    return True


def histogram_distance(roi):
    global base_hist

    hist = compute_histogram(roi)
    retval = cv2.compareHist(base_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
    similarity = 1 - retval
    # print(similarity)
    return similarity


def discard_inner_rectangles(box, roi_list, cont_list):
    x, y, w, h = box
    find = False
    restart = True
    while restart:
        restart = False
        for idx, rect in enumerate(roi_list):
            # nuovo è contenuto in uno già esistente
            if x >= rect[0] and y >= rect[1] and x + w < rect[0] + rect[2] and y + h < rect[1] + rect[3]:
                find = True
                break
            # uno già esistente è più piccolo di uno nuovo
            if rect[0] >= x and rect[1] >= y and rect[0] + rect[2] <= x + w and rect[1] + rect[3] <= y + h:
                roi_list.pop(idx)
                cont_list.pop(idx)
                restart = True
            if restart is True:
                break
    return find, roi_list, cont_list


def discard_false_positives(frame, contours):
    roi_list = []
    cont_list = []
    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)  # Bounding boxes
        box = (x, y, w, h)
        if not check_dims(w, h):
            continue

        # Histogram distance
        roi = frame[:, y:y + h, x:x + w]
        similarity = histogram_distance(roi)
        if similarity < 0.3:
            continue
        # cv2.imwrite('/home/lorenzo/cvcs/image.png', HW3(roi))

        # Discard inner rectangles
        inner, roi_list, cont_list = discard_inner_rectangles(box, roi_list, cont_list)
        if inner is True:
            continue

        roi_list.append(box)
        cont_list.append(cont)

    return roi_list, cont_list


def detect_paintings(frame):
    # Blurring
    gray = cv2.cvtColor(hw3(frame), cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = edge_detection(blur)

    # Significant contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = np.zeros_like(frame)
    cv2.drawContours(hw3(img_contours), contours, -1, (0, 255, 0), thickness=2)
    img_contours = cv2.cvtColor(hw3(img_contours), cv2.COLOR_RGB2GRAY)

    # Morphology transformations
    img_contours = cv2.morphologyEx(img_contours, cv2.MORPH_ERODE, (3, 3), iterations=2)
    img_contours = cv2.morphologyEx(img_contours, cv2.MORPH_CLOSE, (3, 3), iterations=5)

    # Significant contours
    contours, _ = cv2.findContours(img_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Discard false positives
    roi_list, cont_list = discard_false_positives(frame, contours)

    return roi_list, cont_list

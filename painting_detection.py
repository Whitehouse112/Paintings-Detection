import numpy as np
import cv2


base_hist = None
num_ex = 0
sum_hist = 0


def compute_histogram(img, stripes=10):
    """https://www.researchgate.net/publication/310953361_Comparative_study_of_histogram_distance_measures_for_re
    -identification """

    h = img.shape[0]
    w = img.shape[1]
    mask = np.zeros((h, w), dtype=np.uint8)

    every = h // (stripes + 1)
    for i in range(1, stripes + 1):
        mask[i * every, :] = 255

    hist = cv2.calcHist([img], [0, 1, 2], mask, [32, 32, 32], [0, 255, 0, 255, 0, 255])
    return hist


def init_histogram():
    import os
    global base_hist, num_ex, sum_hist

    n = len([name for name in os.listdir('photos/')])
    for i in range(1, n + 1):
        img = cv2.imread("photos/" + str(i) + ".png")

        hist = compute_histogram(img)

        if i == 1:
            base_hist = hist
        else:
            base_hist += hist
    sum_hist = base_hist
    num_ex = n
    base_hist /= n


def edge_detection(img):
    """Edge detection with Sobel"""

    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    return mag


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
    x_box, y_box, w_box, h_box = box
    find = False
    restart = True
    while restart:
        restart = False
        for idx, rect in enumerate(roi_list):
            x_rect, y_rect, w_rect, h_rect = rect
            # nuovo è contenuto in uno già esistente
            if x_box >= x_rect and y_box >= y_rect and \
               x_box + w_box < x_rect + w_rect and y_box + h_box < y_rect + h_rect:
                find = True
                break
            # uno già esistente è più piccolo di uno nuovo
            if x_rect >= x_box and y_rect >= y_box and \
               x_rect + w_rect <= x_box + w_box and y_rect + h_rect <= y_box + h_box:
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
        roi = frame[y:y + h, x:x + w]
        similarity = histogram_distance(roi)
        if similarity < 0.38:
            continue
        # cv2.imwrite('photos/image.png', HW3(roi))

        # Discard inner rectangles
        inner, roi_list, cont_list = discard_inner_rectangles(box, roi_list, cont_list)
        if inner is True:
            continue

        roi_list.append(box)
        cont_list.append(cont)

    return roi_list, cont_list


def update_histogram(roi_list, frame):
    global base_hist, num_ex, sum_hist

    for roi in roi_list:
        x, y, w, h = roi
        example = frame[y:y + h, x:x + w]
        ex_hist = compute_histogram(example)
        sum_hist += ex_hist
        num_ex += 1
        base_hist = sum_hist / num_ex


def detect_paintings(frame):
    # Contrast Limited Adaptive Histogram Equalization
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equal = clahe.apply(gray)

    # Edge detection
    edges = edge_detection(equal)

    # Filtering
    bil = cv2.bilateralFilter(edges, 5, 200, 200)

    # Thresholding
    thr = np.uint8(bil > 70) * 255

    # Significant contours 1
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = np.zeros_like(frame)
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), thickness=2)
    img_contours = cv2.cvtColor(img_contours, cv2.COLOR_RGB2GRAY)

    # Morphology transformations
    img_morph = img_contours
    # img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_ERODE, (3, 3), iterations=2)  # do not touch
    # img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_DILATE, (3, 3), iterations=5)
    # img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_ERODE, (3, 3), iterations=10)

    # Significant contours 2
    contours, _ = cv2.findContours(img_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Discard false positives
    roi_list, cont_list = discard_false_positives(frame, contours)

    global num_ex
    if num_ex < 50:
        update_histogram(roi_list, frame)

    # draw_contours(thr, cont_list, frame)
    return roi_list, cont_list


def draw_contours(edges, contours, frame):
    img_contours = np.zeros_like(frame)
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), thickness=2)
    img_contours = cv2.cvtColor(img_contours, cv2.COLOR_RGB2GRAY)

    vertical_concat = np.concatenate((edges, img_contours), axis=0)
    cv2.imshow('Contours', cv2.resize(vertical_concat, (int(1600 / 2), 900)))

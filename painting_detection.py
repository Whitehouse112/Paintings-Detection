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
    if _w < 150 or _h < 150:
        return False
    if _w >= 1500 or _h > 1080:
        return False
    if _w / _h > 3 or _h / _w > 3:
        return False
    return True


def histogram_distance(roi):
    global base_hist

    hist = compute_histogram(roi)
    retval = cv2.compareHist(base_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
    similarity = 1 - retval
    return similarity


def bbox_iou(box1, box2):
    """
    From Yolo v3
    Returns the IoU of two bounding boxes
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)

    # Intersection area
    inter_area = (inter_rect_x2 - inter_rect_x1 + 1) * (inter_rect_y2 - inter_rect_y1 + 1)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def merge_overlapping(box, box_cont, roi_list, cont_list):
    for idx, roi in enumerate(roi_list):
        iou = bbox_iou(box, roi)
        if iou > 0:
            print(iou)
            x = min(box[0], roi[0])
            y = min(box[1], roi[1])
            x2 = max(box[0] + box[2], roi[0] + roi[2])
            y2 = max(box[1] + box[3], roi[1] + roi[3])
            w = x2 - x
            h = y2 - y
            box = (x, y, w, h)
            box_cont = np.concatenate((np.array(box_cont), np.array(cont_list[idx])), axis=0)
            roi_list.pop(idx)
            cont_list.pop(idx)
            box, box_cont, roi_list, cont_list = merge_overlapping(box, box_cont, roi_list, cont_list)
            break

    return box, box_cont, roi_list, cont_list


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
        global num_ex
        if num_ex < 50:
            if similarity < 0.2:
                continue
        else:
            if similarity < 0.4:
                continue
        # cv2.imwrite('photos/image.png', roi)

        # Discard overlapping
        box, cont, roi_list, cont_list = merge_overlapping(box, cont, roi_list, cont_list)

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
    thr = np.uint8(bil > 60) * 255

    # Significant contours 1
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = np.zeros_like(frame)
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), thickness=2)
    img_contours = cv2.cvtColor(img_contours, cv2.COLOR_RGB2GRAY)

    # Morphology transformations
    img_morph = img_contours
    img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_ERODE, (3, 3), iterations=2)  # do not touch
    img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_DILATE, (3, 3), iterations=5)
    img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_ERODE, (3, 3), iterations=3)

    # Significant contours 2
    contours, _ = cv2.findContours(img_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Discard false positives
    roi_list, cont_list = discard_false_positives(frame, contours)

    global num_ex
    if num_ex < 50:
        update_histogram(roi_list, frame)

    # draw_contours(img_morph, cont_list, frame)
    return roi_list, cont_list


def draw_contours(edges, contours, frame):
    img_contours = np.zeros_like(frame)
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), thickness=2)
    img_contours = cv2.cvtColor(img_contours, cv2.COLOR_RGB2GRAY)

    vertical_concat = np.concatenate((edges, img_contours), axis=0)
    cv2.imshow('Contours', cv2.resize(vertical_concat, (int(1600 / 2), 900)))

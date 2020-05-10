import cv2
import numpy as np

def cropROI(frame, ROIs):
    frame_h, frame_w = frame.shape[:2]
    image = np.zeros((frame_h, frame_w, 3), np.uint8)

    kernel = np.ones((5, 5), np.uint8)

    hull = []
    blank = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

    img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    mask = np.zeros((frame_h+2, frame_w+2), dtype=np.uint8)
    grad = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    _, thr = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
    thr = cv2.bilateralFilter(thr, 5, 200, 200)
    # thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, ker_er)


    contours = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # test = cv2.drawContours(blank, contours, -1, (0, 255, 0), thickness=2)

    for i in range(len(contours)):
        hull.append(cv2.convexHull(contours[i], True))
    cv2.drawContours(blank, hull, -1, (0, 255, 0), thickness=2)

    #TODO: Trovare un modo intelligente di ricercare un punto esterno ai quadri
    flood = cv2.cvtColor(blank, cv2.COLOR_RGB2GRAY)
    cv2.floodFill(flood, mask, (500, 0), 255)
    inv = cv2.bitwise_not(flood)
    out = thr | inv

    for ROI in ROIs:
        x, y, w, h = ROI
        mask = out[y:y+h, x:x+w]
        rect = frame[y:y+h, x:x+w]
        crop = cv2.bitwise_and(rect, rect, mask=mask)

        image[y:y+h, x:x+w] = crop

    return image
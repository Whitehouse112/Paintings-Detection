import cv2
import numpy as np

# def cropROI(frame, ROIs):
#     #gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#     images = []
#     ker_dil = np.ones((5,5), np.uint8)

#     for ROI in ROIs:
#         x, y, w, h = ROI
#         if x > 20: x-=20
#         if y > 20: y-=20
#         if w < 1180: w+=20
#         if h < 700: h+=20
#         tmp = frame[y:y+h, x:x+w]
#         img = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
#         mask = np.zeros((h+2, w+2), dtype=np.uint8)
#         _, thr = cv2.threshold(img, 0, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#         #thr = cv2.dilate(thr, ker_dil, iterations=1)
#         flood = thr.copy()
#         cv2.floodFill(flood, mask, (0,0), 255)
#         inv = cv2.bitwise_not(flood)

#         img = cv2.bitwise_and(tmp, tmp, mask=thr | inv)
#         images.append(img)

#     return images

# def cropROI2(frame, ROIs, contours):
#     images = []

#     for ROI in ROIs:
#         x, y, w, h = ROI
#         tmp = frame[y:y+h, x:x+w]
#         cont = contours[y:y+h, x:x+w]
#         #img = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
#         mask = np.zeros((h+2, w+2), dtype=np.uint8)
#         #_, thr = cv2.threshold(img, 0, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#         flood = cont.copy()
#         cv2.floodFill(flood, mask, (int(w/2), int(h/2)), 255)
#         inv = cv2.bitwise_not(flood)

#         img = cv2.bitwise_and(tmp, tmp, mask=cont | inv)
#         images.append(flood)

#     return images

def cropROI(frame, ROIs):
    #gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    images = []
    
    #ker_dil = np.ones((5,5), dtype=np.uint8)
    ker_er = np.ones((2,2), dtype=np.uint8)

    for ROI in ROIs:
        x, y, w, h = ROI
        hull = []
        blank = np.zeros((h, w, 3), dtype=np.uint8)

        tmp = np.array(frame[y:y+h, x:x+w])
        img = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
        mask = np.zeros((h+2, w+2), dtype=np.uint8)
        #_, thr = cv2.threshold(img, 0, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        thr = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3)
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, ker_er)

        contours = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        test = cv2.drawContours(blank, contours, -1, (0, 255, 0), thickness=2)

        for i in range(len(contours)):
            hull.append(cv2.convexHull(contours[i], True))
        cv2.drawContours(blank, hull, -1, (0, 255, 0), thickness=2)


        flood = cv2.cvtColor(blank, cv2.COLOR_RGB2GRAY)
        cv2.floodFill(flood, mask, (w//2,h//2), 255)

        img = cv2.bitwise_and(tmp, tmp, mask=flood)
        images.append(img)

    return images
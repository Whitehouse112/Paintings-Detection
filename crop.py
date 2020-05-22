import cv2
import numpy as np

def cropROI(frame, ROIs):
    frame_h, frame_w = frame.shape[:2]

    kernel = np.ones((5, 5), np.uint8)
    ker_close = np.ones((7, 7), np.uint8)
    ker_er = np.ones((2,2), np.uint8)

    hull = []
    blank = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

    img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # grad = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel, iterations=1)
    # _, thr = cv2.threshold(grad, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    # cv2.imshow('Normal', thr)

    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)

    grad = cv2.morphologyEx(cl1, cv2.MORPH_GRADIENT, kernel, iterations=1)
    bil = cv2.bilateralFilter(grad, 5, 200, 200)
    _, thr = cv2.threshold(bil, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
    thr = cv2.dilate(thr, (5, 5), iterations=3)
    
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # test = cv2.drawContours(blank, contours, -1, (0, 255, 0), thickness=cv2.FILLED)

    for i in range(len(contours)):
        hull.append(cv2.convexHull(contours[i], True))
    cv2.drawContours(blank, hull, -1, (0, 255, 0), thickness=2)
    
    mask = np.zeros((frame_h+2, frame_w+2), dtype=np.uint8)

    x0, y0 = findBackground(ROIs, hull, frame_w, frame_h)
    # cv2.circle(blank, (x0, y0), 0, (255, 0, 0), 5)
    flood = cv2.cvtColor(blank, cv2.COLOR_RGB2GRAY)
    cv2.floodFill(flood, mask, (x0, y0), 255)
    inv = cv2.bitwise_not(flood)
    out = thr | inv

    contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dist = frame_w + frame_h
    frames = {}
    for ROI in ROIs:
        x, y, w, h = ROI
        ROI_center = np.array([x+(w//2), y+(h//2)])
        dist = frame_w + frame_h
        for c in range(len(contours)):
            M = cv2.moments(contours[c])
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            center = np.array([cX, cY])
            cur = cv2.norm(ROI_center - center)
            if cur < dist:
                dist = cur
                frames[ROI] = c


    mask = np.zeros((frame_h, frame_w, 3), np.uint8)
    for indx in frames.values():
        cv2.drawContours(mask, contours, indx, (255,255,255), thickness=cv2.FILLED)

    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    crop = cv2.bitwise_and(frame, frame, mask=mask)

    return crop

def findBackground(ROIs, contours, frame_w, frame_h):
    x0 = 0
    y0 = 0
    dist = frame_w + frame_h
    frame_center = np.array([frame_w//2, frame_h//2])
    for ROI in ROIs:
        x, y, w, h = ROI
        center = np.array([x+(w//2), y+(h//2)])
        cur = cv2.norm(frame_center-center)
        if cur < dist:
            dist = cur
            if x != 0 or y != 0:
                x0 = np.max((1, x-5))
                y0 = np.max((1, y-5))
                if checkContour(contours, x0, y0):
                    x0 = np.min((frame_w-1, x+w+5))
                    y0 = np.min((frame_h-1, y+h+5))
            else:
                x0 = np.min((frame_w-1, x+w+5))
                y0 = np.min((frame_h-1, y+h+5))
    
    return x0, y0

def checkContour(contours, x0, y0):
    for contour in contours:
        if cv2.pointPolygonTest(contour, (x0, y0), False) != -1: return True
    return False

def lap(img):
    # blur = cv2.GaussianBlur(img, (3, 3), 0)
    dst = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
    abs_dst = cv2.convertScaleAbs(dst)
    thr = abs_dst
    thr = cv2.bilateralFilter(abs_dst, 3, 150, 150)

    return thr

def sup(frame):
    slic = cv2.ximgproc.createSuperpixelSLIC(frame, region_size=16)
    slic.iterate()
    mask = slic.getLabelContourMask()
    res = cv2.bitwise_or(frame, frame, mask=mask)
    return res

def grab(frame, ROIs):
    h, w = frame.shape[:2]
    blank = np.zeros((h, w, 3), np.uint8)
    mask = np.zeros((h, w), np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    for ROI in ROIs:
        mask, bgdModel, fgdModel = cv2.grabCut(frame,mask,ROI,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        cur = frame*mask[:,:,np.newaxis]
        blank = cv2.bitwise_or(blank, cur)
    return blank

def kmeans(frame, ROIs):
    # centers = np.array((len(ROIs), 2))
    Z = np.float32(frame.reshape((-1 ,3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5 ,1.0)
    K = 4
    # for ROI in ROIs:
    #     x, y, w, h = ROI
    #     centers[RO] = np.array([x+(w//2), y+(h//2)])
    rect, label, center = cv2.kmeans(Z ,K, None, criteria, 5, cv2.KMEANS_USE_INITIAL_LABELS)

    center = np.uint8(center)
    return center[label.flatten()].reshape((frame.shape))
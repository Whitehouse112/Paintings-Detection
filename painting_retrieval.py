import cv2
import numpy as np


db = []


def init_database():
    import os
    global db
    
    orb = cv2.ORB_create()
    tmp = []

    names = [name for name in os.listdir('paintings_db/')]
    for name in names:
        img = cv2.imread(f"paintings_db/{name}")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp, des = orb.detectAndCompute(img, None)
        tmp.append([img, (kp, des)])
    db = tmp


def createMask(painting):
    mask = np.zeros_like((painting), dtype=np.uint8)

    #The painting is passed as a gray scale image
    h = painting.shape[0]
    h_del = round(h*0.2)
    w = painting.shape[1]
    w_del = round(w*0.2)
    mask[h_del:h-h_del, w_del:w-w_del] = 255
    return mask


def findBestMatch(painting_descriptor):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    best = 0
    found = None
    percentage = None
    for i, img in enumerate(db):          
        good_points = []
        matches = matcher.knnMatch(painting_descriptor, img[1][1], k=2)
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good_points.append(m)
        if best < len(good_points):
            best = len(good_points)
            percentage = round(len(good_points)/len(matches)*100)
            found = i
    return found, percentage


def retrieve_paintings(paintings):
    retrieved = []

    for painting in paintings:
        gray = cv2.cvtColor(painting, cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create()
        
        mask = createMask(gray)       
        _, des = orb.detectAndCompute(gray, mask)
        
        found, percentage = findBestMatch(des)
        
        if percentage is not None and found is not None:                  
            if percentage > 5:
                print(f"Percentage: {percentage}%") 
                retrieved.append(db[found][0])
            else:
                retrieved.append(np.zeros_like(painting))
        else:
            retrieved.append(np.zeros_like(painting))
    
    return retrieved
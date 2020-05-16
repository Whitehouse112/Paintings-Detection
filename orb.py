import numpy as np
import cv2
import os
from utility import hw3, resize_concatenate

def init_database():
    
    global db
    orb = cv2.ORB_create()
    #sift = cv2.xfeatures2d.SIFT_create()
    tmp = []

    names = [name for name in os.listdir('paintings_db/')]
    for name in names:
        img = cv2.imread(f"paintings_db/{name}")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #kp, des = sift.detectAndCompute(img,None)
        kp, des = orb.detectAndCompute(img,None)
        tmp.append([img, (kp, des)])
    db = tmp

def orb_retrieve(paintings):
    
    retrieved = []
    for painting in paintings:
        gray = cv2.cvtColor(painting, cv2.COLOR_RGB2GRAY)

        matcher = cv2.BFMatcher()
        orb = cv2.ORB_create()
        _, des = orb.detectAndCompute(gray,None)        

        # sift = cv2.xfeatures2d.SIFT_create()
        # _, des = sift.detectAndCompute(gray,None)
        
        best = 0
        found = 0
        for i, img in enumerate(db):      
            good_points = []
            matches = matcher.match(des, img[1][1])
            for m in matches:
                if m.distance < 100:
                    good_points.append(m)
            if best < len(good_points):
                best = len(good_points)
                found = i
                           
        if best < 5:
            print("No matches found")
        else:
            retrieved.append(db[found][0])

    if len(retrieved) == 0:
        return None
    else:
        return resize_concatenate(retrieved)
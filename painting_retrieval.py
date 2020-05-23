import cv2
import numpy as np
import csv

db = []

def init_database():
    import os
    global db

    orb = cv2.ORB_create()
    tmp = []

    names = [name for name in os.listdir('paintings_db/')]
    for name in names:
        img = cv2.imread(f"paintings_db/{name}")
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        tmp.append([img, (kp, des), name])

    #db is a list in which each element is a list itself structured as below:
    #[RGBimage, (keyPoints, descriptors), painting_name]
    db = tmp


def findName(image_name, file):
    for row in file:
        if image_name in row:
            return row[0]


def createMask(painting):
    mask = np.zeros_like(painting, dtype=np.uint8)

    # The painting is passed as a gray scale image
    h = painting.shape[0]
    h_del = round(h * 0.15)
    w = painting.shape[1]
    w_del = round(w * 0.15)
    mask[h_del:h - h_del, w_del:w - w_del] = 255
    return mask


def findBestMatch(painting_descriptor, file):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    ranking = {}

    #db is a list in which each element is a list itself structured as below:
    #[RGBimage, (keyPoints, descriptors), painting_name]
    for img in db:
        good_points = []
        matches = matcher.knnMatch(painting_descriptor, img[1][1], k=2)    
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_points.append(m) 
        if len(good_points) != 0:
            percentage = len(good_points) / len(matches) * 100
            if percentage > 5:
                ranking[findName(img[2], file)] = (img[0], percentage)      

    order_ranking = {k:v for k,v in sorted(ranking.items(), key=lambda item:item[1][1], reverse=True)}
    return order_ranking


def retrieve_paintings(paintings):
    retrieved = []

    file = open('files/data.csv', 'r')
    reader = csv.reader(file)

    for painting in paintings:
        ranking = {}
        gray = cv2.cvtColor(painting, cv2.COLOR_RGB2GRAY)
        
        #Check painiting dimensions in order to avoid orb error: "(-215) Assertion failed, inv_scale_x > 0"
        if gray.shape[0] > 2 and gray.shape[1] > 2:
            orb = cv2.ORB_create()

            mask = createMask(gray)
            _, des = orb.detectAndCompute(gray, mask)

            ranking = findBestMatch(des, reader)
            print(ranking)
        retrieved.append(ranking)

    file.close()
    #retrieved is a dictionaries list, each dictionary is structured as below:
    #{painting_name: (painting_image, accuracy_percentage)}
    return retrieved
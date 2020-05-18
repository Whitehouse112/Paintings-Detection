import cv2


db = []


def init_database():
    import os
    global db
    
    orb = cv2.ORB_create()
    # sift = cv2.xfeatures2d.SIFT_create()
    tmp = []

    names = [name for name in os.listdir('paintings_db/')]
    for name in names:
        img = cv2.imread(f"paintings_db/{name}")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # kp, des = sift.detectAndCompute(img,None)
        kp, des = orb.detectAndCompute(img, None)
        tmp.append([img, (kp, des)])
    db = tmp


def retrieve_paintings(paintings):
    
    retrieved = []
    for painting in paintings:
        gray = cv2.cvtColor(painting, cv2.COLOR_RGB2GRAY)

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        orb = cv2.ORB_create()
        _, des = orb.detectAndCompute(gray, None)

        # sift = cv2.xfeatures2d.SIFT_create()
        # _, des = sift.detectAndCompute(gray,None)
        
        best = 0
        found = 0
        for i, img in enumerate(db):
            good_points = []
            matches = matcher.knnMatch(des, img[1][1], k=2)
            for m, n in matches:
                if m.distance < 0.6*n.distance:
                    good_points.append(m)
            if best < len(good_points):
                best = len(good_points)
                found = i
                           
        if best < 10:
            print("No matches found")
        else:
            retrieved.append(db[found][0])

    return retrieved

import cv2
import numpy as np
import csv

db = []
data = []


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

    # db is a list in which each element is a list itself structured as below:
    # [RGBimage, (keyPoints, descriptors), painting_name]
    db = tmp


def read_file():
    global data

    try:
        file = open('files/data.csv', 'r')
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
        file.close()
    except IOError:
        print("File data.csv not found.")
        exit(1)


def findRoom(painting_name):
    for row in data:
        if painting_name in row:
            return row[2]
    return 0


def findName(image_name):
    for row in data:
        if image_name in row:
            return row[0]
    return "No name"


def findBestRoom(retrieved):
    rooms = []
    tmp = {}

    for d in retrieved:
        for v in d.values():
            # Save the room only if the accuracy percentage is over the threshold
            if v[1] > 5:
                rooms.append(v[2])
            # The first element is the best one: we can stop the for loop after the first iteration
            break
    for i in rooms:
        if i not in tmp:
            tmp[i] = 1
        else:
            tmp[i] += 1

    if len(rooms) != 0:
        room = max(tmp.items(), key=lambda x: x[1])
        return room[0]
    else:
        return "Unable to find room"


def firstElements(dictionary, number_of_elements):
    tmp = []
    firstN = {}

    for k in dictionary.keys():
        tmp.append(k)
    for i in range(number_of_elements):
        firstN[tmp[i]] = dictionary[tmp[i]]
    return firstN


def createMask(painting):
    mask = np.zeros_like(painting, dtype=np.uint8)

    # The painting is passed as a gray scale image
    h = painting.shape[0]
    h_del = round(h * 0.15)
    w = painting.shape[1]
    w_del = round(w * 0.15)
    mask[h_del:h - h_del, w_del:w - w_del] = 255
    return mask


def findBestMatch(painting_descriptor):
    ranking = {}

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    # db is a list in which each element is a list itself structured as below:
    # [RGBimage, (keyPoints, descriptors), painting_name]
    for img in db:
        good_points = []
        matches = matcher.knnMatch(painting_descriptor, img[1][1], k=2)
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_points.append(m)
        if len(good_points) != 0:
            percentage = len(good_points) / len(matches) * 100
            ranking[findName(img[2])] = (img[0], percentage, findRoom(img[2]))

    order_ranking = {k: v for k, v in sorted(ranking.items(), key=lambda item: item[1][1], reverse=True)}
    if len(order_ranking) > 5:
        return firstElements(order_ranking, 5)
    else:
        return order_ranking


def retrieve_paintings(paintings):
    retrieved = []

    for painting in paintings:
        gray = cv2.cvtColor(painting, cv2.COLOR_RGB2GRAY)

        # Check painiting dimensions in order to avoid orb error: "(-215) Assertion failed, inv_scale_x > 0"
        if not(gray.shape[0] > 2 and gray.shape[1] > 2):
            continue
        
        orb = cv2.ORB_create()

        mask = createMask(gray)
        _, des = orb.detectAndCompute(gray, mask)

        ranking = findBestMatch(des)
        retrieved.append(ranking)

    # retrieved is a dictionaries list, each dictionary is structured as below:
    # {painting_name: (painting_image, accuracy_percentage, room)}
    room = findBestRoom(retrieved)
    return room, retrieved

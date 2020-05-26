import cv2
import numpy as np
import csv

des_db = []         # each element will be a tuple (img_name, descriptors)
paintings_db = {}   # img_name: img
data_csv = {}       # img_name: (title, author, room)


def init_database():
    import os
    global des_db

    orb = cv2.ORB_create()

    img_names = [img_name for img_name in os.listdir('paintings_db/')]
    for img_name in img_names:
        img = cv2.imread(f"paintings_db/{img_name}")
        paintings_db[img_name] = img
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, des = orb.detectAndCompute(gray, None)
        des_db.append((img_name, des))


def read_file():
    global data_csv

    try:
        file = open('files/data.csv', 'r')
        reader = csv.reader(file)
        next(reader)  # skip first line
        for row in reader:
            # row = [Title, Author, Room, Image]
            title, author, room, img_name = row
            data_csv[img_name] = (title, author, room)
        file.close()
    except IOError:
        print("File data.csv not found.")
        exit(1)


def findRoom(retrieved):
    names = retrieved[:, 0, 0]
    rooms_hist = np.zeros((30,), dtype=np.uint8)

    for name in names:
        _, _, room = data_csv[name]
        room = np.uint8(room)
        rooms_hist[room] += 1

    return max(rooms_hist)


def createMask(img):
    mask = np.zeros_like(img, dtype=np.uint8)
    h = img.shape[0]
    h_del = round(h * 0.15)
    w = img.shape[1]
    w_del = round(w * 0.15)
    mask[h_del:h - h_del, w_del:w - w_del] = 255
    return mask


def findBestMatches(painting_descriptors, n_max=5):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    ranking = []
    for img in des_db:
        img_name, img_des = img

        good_points = []
        matches = matcher.knnMatch(painting_descriptors, img_des, k=2)
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_points.append(m)
        if len(good_points) > 0:
            accuracy = len(good_points) / len(matches) * 100
            ranking.append((img_name, accuracy))

    ranking.sort(reverse=True, key=lambda match: match[1])
    if len(ranking) > n_max:
        return ranking[:n_max]
    else:
        return ranking


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

        ranking = findBestMatches(des)  # list of tuples (img_name, accuracy)
        retrieved.append(ranking)
    retrieved = np.array(retrieved)

    room = findRoom(retrieved)

    return room, retrieved

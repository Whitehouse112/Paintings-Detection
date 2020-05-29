import cv2
import numpy as np
import csv

des_db = []         # each element will be a tuple (img_name, descriptors)
paintings_db = {}   # img_name: img
data_csv = {}       # img_name: (title, author, room)


def init_database():
    import os

    orb = cv2.ORB_create()

    img_names = [img_name for img_name in os.listdir('paintings_db/')]
    for img_name in img_names:
        img = cv2.imread(f"paintings_db/{img_name}")
        paintings_db[img_name] = img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, des = orb.detectAndCompute(gray, None)
        des_db.append((img_name, des))


def read_file():
    path = 'files/'

    try:
        file = open(path + 'data.csv', 'r')
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
    accuracy_threshold = 5
    names = [r[0, 0] for r in retrieved if round(np.float32(r[0, 1])) > accuracy_threshold]

    rooms_hist = np.zeros((30,), dtype=np.float32)
    for idx, name in enumerate(names):
        _, _, room = data_csv[name]
        accuracy = np.float32(retrieved[idx, 0, 1])
        room = np.uint8(room)
        rooms_hist[room] += accuracy

    return np.argmax(rooms_hist)


def createMask(img):
    h, w = img.shape[:2]

    h_del = round(h * 0.15)
    w_del = round(w * 0.15)

    mask = np.zeros_like(img, dtype=np.uint8)
    mask[h_del:h - h_del, w_del:w - w_del] = 255
    return mask


def findBestMatches(painting_descriptors, n_max=3):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    ranking = []
    for img in des_db:
        img_name, img_des = img

        matches = matcher.knnMatch(painting_descriptors, img_des, k=2)

        good_points = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_points.append(m)

        if len(good_points) > 0:
            accuracy = len(good_points) / len(matches) * 100
            ranking.append((img_name, accuracy))

    ranking.sort(reverse=True, key=lambda match: match[1])

    # Bring matches to exacly n_max
    while len(ranking) < n_max:
        x = np.random.randint(0, len(des_db))
        img_name = des_db[x][0]
        ranking.append((img_name, 0))
    if len(ranking) > n_max:
        ranking = ranking[:n_max]

    return ranking


def retrieve_paintings(paintings):
    retrieved = []
    for painting in paintings:
        # Check painiting dimensions in order to avoid orb error: "(-215) Assertion failed, inv_scale_x > 0"
        if not(painting.shape[0] > 2 and painting.shape[1] > 2):
            continue

        orb = cv2.ORB_create()

        gray = cv2.cvtColor(painting, cv2.COLOR_BGR2GRAY)
        mask = createMask(gray)
        _, des = orb.detectAndCompute(gray, mask)

        ranking = findBestMatches(des)  # list of tuples (img_name, accuracy)
        retrieved.append(ranking)
    retrieved = np.array(retrieved)

    room = 0
    if len(paintings) > 0:
        room = findRoom(retrieved)

    return room, retrieved

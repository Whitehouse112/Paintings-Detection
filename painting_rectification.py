import numpy as np
import cv2
from utility import hw3


# def segmentation(frame, roi_list):
#     paintings = []
#
#     for roi in roi_list:
#         x, y, w, h = roi
#         painting = np.array(frame[:, y:y + h, x:x + w])
#         gray = cv2.cvtColor(hw3(painting), cv2.COLOR_RGB2GRAY)
#         _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#         # thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3)
#
#         # thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, (3, 3), iterations=1)
#
#         contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         hull = cv2.convexHull(max(contours, key=len))
#         # cv2.drawContours(HW3(painting), [hull], -1, (0, 255, 0), thickness=2)
#
#         img_hull = np.zeros_like(painting)
#         cv2.drawContours(hw3(img_hull), [hull], -1, (0, 255, 0), thickness=cv2.FILLED)
#         img_hull = cv2.cvtColor(hw3(img_hull), cv2.COLOR_RGB2GRAY)
#         painting *= np.uint8(img_hull > 0)
#
#         paintings.append(painting)
#
#     return paintings


def find_intersections(lines):
    horizontals = []
    verticals = []
    for line in lines:
        rho, theta = line[0]  # x*cos(th) + y*sin(th) = rho
        angle = (theta * 360) / (2 * np.pi)
        if -45 <= angle < 45 or 135 <= angle < 225:
            horizontals.append(line)
        else:
            verticals.append(line)

    intersections = []
    for horiz in horizontals:
        for vert in verticals:
            rho1, theta1 = horiz[0]
            rho2, theta2 = vert[0]
            th_mat = np.array([
                [np.cos(theta1), np.sin(theta1)],
                [np.cos(theta2), np.sin(theta2)]
            ])
            b = np.array([[rho1], [rho2]])
            x0, y0 = np.linalg.solve(th_mat, b)
            intersections.append([[x0[0], y0[0]]])

    intersections = np.array(intersections, dtype=np.float32)
    return intersections


def order_centers(centers):
    dtype = [('x', centers.dtype), ('y', centers.dtype)]
    centers = centers.ravel().view(dtype)

    centers.sort(order=['x'])
    left_most = centers[:2]
    right_most = centers[2:]

    left_most.sort(order=['y'])
    tl, bl = left_most
    tl = np.array(list(tl), dtype=np.float32)
    bl = np.array(list(bl), dtype=np.float32)

    right_most.sort(order=['y'])
    tr, br = right_most
    tr = np.array(list(tr), dtype=np.float32)
    br = np.array(list(br), dtype=np.float32)

    return tl, tr, bl, br


def compute_aspect_ratio1(tl, tr, bl, br, frame_shape):
    h1 = bl[1] - tl[1]
    h2 = br[1] - tr[1]
    w1 = tr[0] - tl[0]
    w2 = br[0] - bl[0]
    h = max(h1, h2)
    w = max(w1, w2)

    # image center
    u0 = frame_shape[2] / 2
    v0 = frame_shape[1] / 2

    ar_vis = w / h  # visible aspect ratio
    m1 = np.append(tl, 1)
    m2 = np.append(tr, 1)
    m3 = np.append(bl, 1)
    m4 = np.append(br, 1)

    # calculate the focal distance
    k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
    k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)

    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1

    n21, n22, n23 = n2
    n31, n32, n33 = n3

    if n23 != 0 and n33 != 0:
        f = np.sqrt(np.abs((1 / (n23 * n33)) * ((n21 * n31 - (n21 * n33 + n23 * n31) * u0 + n23 * n33 * (u0 ** 2)) + (
                n22 * n32 - (n22 * n33 + n23 * n32) * v0 + n23 * n33 * (v0 ** 2)))))

        A = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1]], dtype=np.float32)

        At = np.transpose(A)
        Ati = np.linalg.inv(At)
        Ai = np.linalg.inv(A)

        # calculate the real aspect ratio
        ar_real = np.sqrt(np.dot(np.dot(np.dot(n2, Ati), Ai), n2) / np.dot(np.dot(np.dot(n3, Ati), Ai), n3))
    else:
        ar_real = np.sqrt((n21 ** 2 + n22 ** 2) / (n31 ** 2 + n32 ** 2))

    if ar_real < ar_vis:
        w = int(w)
        h = int(w / ar_real)
    else:
        h = int(h)
        w = int(ar_real * h)

    return h, w


def compute_aspect_ratio2(tl, tr, bl, br):
    # da aggiustare
    h_max = max(bl[1] - tl[1], br[1] - tr[1])
    h_min = min(bl[1] - tl[1], br[1] - tr[1])
    w_max = max(tr[0] - tl[0], br[0] - bl[0])
    # w_min = min(tr[0] - tl[0], br[0] - bl[0])
    dim_perc_hmin = (h_min * 100) / h_max
    diff_perc = (100 - dim_perc_hmin) / 3
    w = int((w_max * 100) / (dim_perc_hmin + diff_perc))
    h = int(h_max)
    return h, w


def rectify_paintings(cont_list, frame):
    for contour in cont_list:
        hull = cv2.convexHull(contour)
        img_hull = np.zeros_like(frame)
        cv2.drawContours(hw3(img_hull), [hull], -1, (0, 255, 0), thickness=2)
        img_hull = cv2.cvtColor(hw3(img_hull), cv2.COLOR_RGB2GRAY)

        lines = cv2.HoughLines(img_hull, 1, np.pi / 180, 150)
        if lines is None:
            print("Painting not found.")
            continue

        intersections = find_intersections(lines)
        # N of intersection must be greater than K (number of cluster)
        if len(intersections) < 4:
            print("Painting not found.")
            continue

        # fare funzione per questo (find vertexes?)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(intersections, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.array(centers, dtype=np.float32)

        tl, tr, bl, br = order_centers(centers)

        # find aspect ratio
        # h, w = compute_aspect_ratio1(tl, tr, bl, br, frame.shape)
        h, w = compute_aspect_ratio2(tl, tr, bl, br)

        # rectification
        pts_src = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
        pts_dst = np.array([tl, tr, bl, br], dtype=np.float32)
        m, _ = cv2.findHomography(pts_src, pts_dst, method=cv2.RANSAC)
        painting = cv2.warpPerspective(hw3(frame), m, (w, h), flags=cv2.WARP_INVERSE_MAP)

        # draw
        cv2.imshow("Rectified", painting)
        # cv2.imshow("Frame", hw3(frame))
        # img_lines = np.zeros_like(frame)
        # cv2.drawContours(hw3(img_lines), [hull], -1, (255, 0, 0), thickness=2)
        # for line in lines:
        #     rho, theta = line[0]
        #     a = np.cos(theta)
        #     b = np.sin(theta)
        #     x0 = a * rho
        #     y0 = b * rho
        #     x1 = int(x0 + 2000 * (-b))
        #     y1 = int(y0 + 2000 * a)
        #     x2 = int(x0 - 2000 * (-b))
        #     y2 = int(y0 - 2000 * a)
        #     cv2.line(hw3(img_lines), (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
        # cv2.drawContours(hw3(img_lines), np.expand_dims(centers, axis=1), -1, (0, 0, 255), thickness=5)
        # cv2.imshow("Lines", hw3(img_lines))

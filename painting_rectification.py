import numpy as np
import cv2
from utility import HW3


# refactor di cropROI
def segmentation(frame, roi_list):
    paintings = []

    for roi in roi_list:
        x, y, w, h = roi
        painting = np.array(frame[:, y:y + h, x:x + w])
        gray = cv2.cvtColor(HW3(painting), cv2.COLOR_RGB2GRAY)
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3)

        # thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, (3, 3), iterations=1)

        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(max(contours, key=len))
        # cv2.drawContours(HW3(painting), [hull], -1, (0, 255, 0), thickness=2)

        img_hull = np.zeros_like(painting)
        cv2.drawContours(HW3(img_hull), [hull], -1, (0, 255, 0), thickness=cv2.FILLED)
        img_hull = cv2.cvtColor(HW3(img_hull), cv2.COLOR_RGB2GRAY)
        painting *= np.uint8(img_hull > 0)

        paintings.append(painting)

    return paintings


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
            A = np.array([
                [np.cos(theta1), np.sin(theta1)],
                [np.cos(theta2), np.sin(theta2)]
            ])
            b = np.array([[rho1], [rho2]])
            x0, y0 = np.linalg.solve(A, b)
            x0, y0 = int(np.round(x0)), int(np.round(y0))
            intersections.append([[x0, y0]])
    return intersections


# def rectify_paintings(cont_list):
#     poly_list = []
#     for contour in cont_list:
#         hull = cv2.convexHull(contour)
#         epsilon = 0.07 * cv2.arcLength(hull, True)
#         approxCurve = cv2.approxPolyDP(hull, epsilon, True)
#         poly_list.append(approxCurve)
#
#     return poly_list


def rectify_paintings(cont_list, frame):
    for idx, contour in enumerate(cont_list):
        hull = cv2.convexHull(contour)
        img_hull = np.zeros_like(frame)
        cv2.drawContours(HW3(img_hull), [hull], -1, (0, 255, 0), thickness=2)
        img_hull = cv2.cvtColor(HW3(img_hull), cv2.COLOR_RGB2GRAY)

        lines = cv2.HoughLines(img_hull, 1, np.pi / 180, 200)

        intersections = np.array(find_intersections(lines)).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(intersections, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.array(np.expand_dims(centers, axis=1)).astype(np.int)

        # cercare i due lati più lunghi per creare la nuova figura
        # proiettare

        # draw
        img_lines = np.zeros_like(frame)
        cv2.drawContours(HW3(img_lines), [hull], -1, (255, 0, 0), thickness=2)
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * a)
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * a)
            cv2.line(HW3(img_lines), (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
        cv2.drawContours(HW3(img_lines), centers, -1, (0, 0, 255), thickness=5)
        cv2.imshow("Lines", HW3(img_lines))

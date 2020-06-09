import numpy as np
import cv2


f_tot = 1000
num_f = 1
focal_length = 1000


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


def vertices_kmeans(intersections):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(intersections, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.expand_dims(np.array(centers, dtype=np.float32), axis=1)
    return centers


def check_vertices(tl, tr, bl, br, frame_h, frame_w):

    tl[0] = np.clip(tl[0], -200, frame_w)
    tl[1] = np.clip(tl[1], -200, frame_h)

    tr[0] = np.clip(tr[0], 0, frame_w + 200)
    tr[1] = np.clip(tr[1], -200, frame_h)

    bl[0] = np.clip(bl[0], -200, frame_w)
    bl[1] = np.clip(bl[1], 0, frame_h + 200)

    br[0] = np.clip(br[0], 0, frame_w + 200)
    br[1] = np.clip(br[1], 0, frame_h + 200)

    return tl, tr, bl, br


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


def compute_aspect_ratio(tl, tr, bl, br, frame_shape):
    """
    Whiteboard scanning and image enhancement
    Zhengyou Zhang, Li-Wei He
    https://www.microsoft.com/en-us/research/uploads/prod/2016/11/Digital-Signal-Processing.pdf
    """

    h1 = bl[1] - tl[1]
    h2 = br[1] - tr[1]
    w1 = tr[0] - tl[0]
    w2 = br[0] - bl[0]
    h = max(h1, h2)
    w = max(w1, w2)

    # image center
    u0 = frame_shape[1] / 2
    v0 = frame_shape[0] / 2

    ar_vis = w / h  # visible aspect ratio

    m1 = np.append(tl, 1)
    m2 = np.append(tr, 1)
    m3 = np.append(bl, 1)
    m4 = np.append(br, 1)

    # cross product = prodotto vettoriale
    # dot product = prodotto scalare
    k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
    k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)

    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1

    n21, n22, n23 = n2
    n31, n32, n33 = n3

    if n23 != 0 and n33 != 0:
        f_squared = -((1 / (n23 * n33)) * ((n21 * n31 - (n21 * n33 + n23 * n31) * u0 + n23 * n33 * (u0 ** 2)) + (
                n22 * n32 - (n22 * n33 + n23 * n32) * v0 + n23 * n33 * (v0 ** 2))))

        global focal_length, f_tot, num_f
        if 0 < f_squared < 2000 ** 2 and num_f < 300:
            f = np.sqrt(f_squared)  # focal-lenght in pixels
            f_tot += f
            num_f += 1
            focal_length = f_tot / num_f
        f = focal_length

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


def rectify_paintings(roi_list, cont_list, frame):
    new_roi_list = []
    new_cont_list = []
    rectified = []
    # img_lines = np.zeros_like(frame)
    for idx, contour in enumerate(cont_list):

        # Polygonal approximation
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        img_poly = np.zeros_like(frame)
        cv2.drawContours(img_poly, [approx], -1, (0, 255, 0), thickness=5)
        img_poly = cv2.cvtColor(img_poly, cv2.COLOR_BGR2GRAY)

        # Finding lines with Hough transform
        lines = cv2.HoughLines(img_poly, 1.3, np.pi / 180, 150)
        if lines is None:
            continue

        # Lines intersections
        intersections = find_intersections(lines)
        if len(intersections) < 4:
            continue

        # Average vertices with K-Means
        vertices = vertices_kmeans(intersections)

        # Ordering vertices
        tl, tr, bl, br = order_centers(vertices)
        tl, tr, bl, br = check_vertices(tl, tr, bl, br, frame.shape[0], frame.shape[1])
        frame_h, frame_w = frame.shape[0:2]
        hmax, hmin = max(bl[1] - tl[1], br[1] - tr[1]), min(bl[1] - tl[1], br[1] - tr[1])
        wmax, wmin = max(tr[0] - tl[0], br[0] - bl[0]), min(tr[0] - tl[0], br[0] - bl[0])
        if not(50 <= hmax <= frame_h and 50 <= hmin <= frame_h and 50 <= wmax <= frame_w and 50 <= wmin <= frame_w):
            continue

        # Compute aspect-ratio
        h, w = compute_aspect_ratio(tl, tr, bl, br, frame.shape)

        # Warp perspective
        pts_src = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
        pts_dst = np.array([tl, tr, bl, br], dtype=np.float32)
        m, _ = cv2.findHomography(pts_src, pts_dst, method=cv2.RANSAC)
        warped = cv2.warpPerspective(frame, m, (w, h), flags=cv2.WARP_INVERSE_MAP)

        rectified.append(warped)
        new_roi_list.append(roi_list[idx])
        new_cont_list.append(cont_list[idx])

        # draw_lines(img_lines, approx, lines, vertices)
    # cv2.namedWindow("Lines", flags=cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    # cv2.imshow("Lines", cv2.resize(img_lines, (1280, 720)))
    return rectified, new_roi_list, new_cont_list


def draw_lines(img_lines, contour, lines, vertices=None):
    img_lines = np.zeros_like(img_lines)
    cv2.drawContours(img_lines, [contour], -1, (255, 0, 0), thickness=2)
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
        cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
    if vertices is not None:
        cv2.drawContours(img_lines, np.array(vertices, dtype=np.int), -1, (0, 0, 255), thickness=5)
    # cv2.namedWindow("Lines", flags=cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    # cv2.imshow("Lines", cv2.resize(img_lines, (1280, 720)))

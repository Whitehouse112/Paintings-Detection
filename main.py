import numpy as np
import cv2

video = cv2.VideoCapture('VIRB0392.MP4')

while video.isOpened():
    ret, frame = video.read()
    if ret is False:
        break
    frame = np.swapaxes(np.swapaxes(frame, 0, 2), 1, 2)  # (3, H, W)
    # frame = np.array(frame[::-1, :, :])  # from BGR to RGB

    # Edge detection
    edges = cv2.Canny(np.swapaxes(np.swapaxes(frame, 0, 1), 1, 2), 10, 1000, apertureSize=5, L2gradient=True)

    # Hough transform to find streight lines
    lines = cv2.HoughLines(edges, 1.1, np.pi / 180, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)

            # cv2.line(np.swapaxes(np.swapaxes(frame, 0, 1), 1, 2), (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.line(edges, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # cv2.imshow('frame', np.swapaxes(np.swapaxes(frame, 0, 1), 1, 2))
    cv2.imshow('frame', edges)
    if cv2.waitKey(1000) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

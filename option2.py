import numpy as np
import cv2

# video = cv2.VideoCapture('VIRB0392.MP4')
video = cv2.VideoCapture('GOPR5826.MP4')

n_frame = 0
while video.isOpened():
    ret, frame = video.read()
    if ret is False:
        break
    frame = np.swapaxes(np.swapaxes(frame, 0, 2), 1, 2)  # (3, H, W)
    # frame = np.array(frame[::-1, :, :])  # from BGR to RGB
    n_frame += 1
    print("\nFrame", n_frame)

    # Edge detection
    edges = cv2.Canny(np.swapaxes(np.swapaxes(frame, 0, 1), 1, 2), 10, 100, apertureSize=3, L2gradient=True)

    # Morphology transformation (closing)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, (4, 4), iterations=1)

    # Significant contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = np.zeros_like(frame)
    cv2.drawContours(np.swapaxes(np.swapaxes(img_contours, 0, 1), 1, 2), contours, -1, (0, 255, 0), thickness=2)

    # Probabilistic Hough transform to find line segments
    img_contours = cv2.cvtColor(np.swapaxes(np.swapaxes(img_contours, 0, 1), 1, 2), cv2.COLOR_RGB2GRAY)
    lines = cv2.HoughLinesP(img_contours, 1, np.pi / 180, 100, minLineLength=20, maxLineGap=10)
    img_lines = np.zeros_like(frame)
    for line in lines:
        x1, y1, x2, y2 = line[0, 0], line[0, 1], line[0, 2], line[0, 3]
        cv2.line(np.swapaxes(np.swapaxes(img_lines, 0, 1), 1, 2), (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

    # Morphology transformation (closing)
    img_lines = cv2.cvtColor(np.swapaxes(np.swapaxes(img_lines, 0, 1), 1, 2), cv2.COLOR_RGB2GRAY)
    img_lines = cv2.morphologyEx(img_lines, cv2.MORPH_CLOSE, (4, 4), iterations=2)

    # Significant contours
    contours, _ = cv2.findContours(img_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = np.zeros_like(frame)
    cv2.drawContours(np.swapaxes(np.swapaxes(img_contours, 0, 1), 1, 2), contours, -1, (0, 255, 0), thickness=2)
    img_contours = cv2.cvtColor(np.swapaxes(np.swapaxes(img_contours, 0, 1), 1, 2), cv2.COLOR_RGB2GRAY)

    # Bounding boxes/ROIs
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 100 and h > 100:
            cv2.rectangle(np.swapaxes(np.swapaxes(frame, 0, 1), 1, 2), (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            print((x, y, w, h))

    cv2.imshow('ROIs', np.swapaxes(np.swapaxes(frame, 0, 1), 1, 2))
    # cv2.imshow('Edges', edges)
    # cv2.imshow('Hough Lines', img_lines)
    # cv2.imshow('Contours', img_contours)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

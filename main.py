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
    edges = cv2.Canny(np.swapaxes(np.swapaxes(frame, 0, 1), 1, 2), 10, 150, apertureSize=3, L2gradient=True)

    # Hough transform to find segments
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)
    img_lines = np.zeros_like(frame)
    for line in lines:
        x1, x2, y1, y2 = line[0, 0], line[0, 2], line[0, 1], line[0, 3]
        cv2.line(np.swapaxes(np.swapaxes(img_lines, 0, 1), 1, 2), (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Find single ROI (for every painting)
    img_lines = cv2.cvtColor(np.swapaxes(np.swapaxes(img_lines, 0, 1), 1, 2), cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(img_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(np.swapaxes(np.swapaxes(frame, 0, 1), 1, 2), (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('frame', np.swapaxes(np.swapaxes(frame, 0, 1), 1, 2))
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

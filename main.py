import numpy as np
import cv2

video = cv2.VideoCapture('VIRB0392.MP4')

while video.isOpened():
    ret, frame = video.read()
    if ret is False:
        break
    frame = np.swapaxes(np.swapaxes(frame, 0, 2), 1, 2)  # (3, H, W)
    # frame = np.array(frame[::-1, :, :])  # from BGR to RGB

    edge = cv2.Canny(np.swapaxes(np.swapaxes(frame, 0, 1), 1, 2), 50, 400)
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    maxH, maxW, maxX, maxY = None, None, None, None
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if maxH is None or (h * w / 2) > (maxH * maxW / 2):
            maxH, maxW, maxX, maxY = h, w, x, y

    cv2.rectangle(np.swapaxes(np.swapaxes(frame, 0, 1), 1, 2), (maxX, maxY), (maxX + maxW, maxY + maxH), (0, 255, 0), 2)

    cv2.imshow('frame', np.swapaxes(np.swapaxes(frame, 0, 1), 1, 2))
    if cv2.waitKey(25) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

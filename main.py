import numpy as np
import cv2
import matplotlib.pyplot as plt

video = cv2.VideoCapture('VIRB0392.MP4')

_, frame = video.read()
frame = np.swapaxes(np.swapaxes(frame, 0, 2), 1, 2)  # (3, H, W)
frame = np.array(frame[::-1, :, :])  # from BGR to RGB

edge = cv2.Canny(np.swapaxes(np.swapaxes(frame, 0, 1), 1, 2), 50, 400)
contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

maxH, maxW, maxX, maxY = None, None, None, None
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if maxH is None or (h * w / 2) > (maxH * maxW / 2):
        maxH, maxW, maxX, maxY = h, w, x, y

cv2.rectangle(np.swapaxes(np.swapaxes(frame, 0, 1), 1, 2), (maxX, maxY), (maxX + maxW, maxY + maxH), (0, 255, 0), 2)

plt.imshow(np.swapaxes(np.swapaxes(frame, 0, 1), 1, 2))
plt.show()

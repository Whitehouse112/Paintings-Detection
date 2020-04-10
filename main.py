import numpy as np
import cv2
import matplotlib.pyplot as plt

video = cv2.VideoCapture('VIRB0392.MP4')

_, frame = video.read()
img = np.array(frame[:,:,::-1])
edge = cv2.Canny(img, 160, 320)

cont, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in cont:
    x, y, w, h = cv2.boundingRect(c)
    #if w > 200 and h > 400:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

plt.imshow(img)
plt.show()
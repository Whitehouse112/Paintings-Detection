import numpy as np
import cv2

# video = cv2.VideoCapture('VIRB0392.MP4')    # folder 000
# video = cv2.VideoCapture('VIRB0407.MP4')    # folder 000
video = cv2.VideoCapture('GOPR5826.MP4')    # folder 001
if not video.isOpened():
    print("File not found.")

frame_entr = 6.5
while video.grab():
    _, frame = video.retrieve()
    frame = np.swapaxes(np.swapaxes(frame, 0, 2), 1, 2)     # (3, H, W)

    # Edge detection
    edges = cv2.Canny(np.swapaxes(np.swapaxes(frame, 0, 1), 1, 2), 10, 2000, apertureSize=5, L2gradient=True)

    # Morphqology transformation (closing)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, (4, 4), iterations=1)

    # Significant contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = np.zeros_like(frame)
    cv2.drawContours(np.swapaxes(np.swapaxes(img_contours, 0, 1), 1, 2), contours, -1, (0, 255, 0), thickness=2)

    # Probabilistic Hough transform to find line segments
    img_contours = cv2.cvtColor(np.swapaxes(np.swapaxes(img_contours, 0, 1), 1, 2), cv2.COLOR_RGB2GRAY)
    lines = cv2.HoughLinesP(img_contours, 1, np.pi / 180, 100, minLineLength=20, maxLineGap=10)
    img_lines = np.zeros_like(frame)
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
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
    entropies = [frame_entr]
    out = []    # roi list
    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        if w > 100 and h > 100:
            roi = cv2.cvtColor(np.swapaxes(np.swapaxes(frame, 0, 1), 1, 2), cv2.COLOR_RGB2GRAY)[y:y + h, x:x + w]

            histo = np.float32((np.bincount(roi.ravel(), minlength=256)))
            histo += 1
            histo /= np.sum(histo)

            entropy = -np.sum(histo * np.log2(histo))

            if entropy > frame_entr - 0.5:
                entropies.append(entropy)
                out.append((x, y, w, h))
                cv2.rectangle(np.swapaxes(np.swapaxes(frame, 0, 1), 1, 2), (x, y), (x + w, y + h), (0, 255, 0),
                              thickness=2)
    frame_entr = np.sum(entropies) / len(entropies)
    print("entropies = ", entropies)

    # Show results
    print(out)
    horizontal_concat_1 = np.concatenate(
        (cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB), cv2.cvtColor(img_lines, cv2.COLOR_GRAY2RGB)), axis=1)
    horizontal_concat_2 = np.concatenate(
        (cv2.cvtColor(img_contours, cv2.COLOR_GRAY2RGB), np.swapaxes(np.swapaxes(frame, 0, 1), 1, 2)), axis=1)
    vertical_concat = np.concatenate((horizontal_concat_1, horizontal_concat_2), axis=0)
    cv2.imshow('ROIs', cv2.resize(vertical_concat, (1920, 1080)))

    # Delay & escape-key
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

import numpy as np
import cv2

# video_name = 'VIRB0392.MP4'     # folder 000
# video_name = 'VIRB0407.MP4'     # folder 000
video_name = 'GOPR5826.MP4'     # folder 001
video = cv2.VideoCapture(video_name)
if not video.isOpened():
    print("File not found.")

frame_entr = 6.5
while video.grab():
    _, frame = video.retrieve()
    frame = np.swapaxes(np.swapaxes(frame, 0, 2), 1, 2)     # (3, H, W)

    # Blurring
    gray = cv2.cvtColor(np.swapaxes(np.swapaxes(frame, 0, 1), 1, 2), cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Sobel edge detection
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    edges = np.uint8(mag > 15) * 255

    # Morphqology transformations
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, (3, 3), iterations=1)

    # Significant contours
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        (cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB), cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)), axis=1)
    horizontal_concat_2 = np.concatenate(
        (cv2.cvtColor(img_contours, cv2.COLOR_GRAY2RGB), np.swapaxes(np.swapaxes(frame, 0, 1), 1, 2)), axis=1)
    vertical_concat = np.concatenate((horizontal_concat_1, horizontal_concat_2), axis=0)
    cv2.imshow('ROIs', cv2.resize(vertical_concat, (1920, 1080)))

    # Delay & escape-key
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

import numpy as np
import cv2


def checkDims(_w, _h):
    if _w < 100 or _h < 100:
        return False
    if _w >= 1500 or _h > 1080:
        return False
    if _w / _h > 2.5 or _h / _w > 2.5:
        return False
    return True


# video_name = '20180206_114720.mp4'
# video_name = 'VIRB0392.MP4'     # folder 000
# video_name = 'VIRB0407.MP4'     # folder 000
video_name = 'GOPR5826.MP4'  # folder 001
video = cv2.VideoCapture('videos/' + video_name)
if not video.isOpened():
    print("File not found.")

frame_entr = 7
while video.grab():
    _, frame = video.retrieve()
    frame = np.swapaxes(np.swapaxes(frame, 0, 2), 1, 2)  # (3, H, W)

    # Blurring
    gray = cv2.cvtColor(np.swapaxes(np.swapaxes(frame, 0, 1), 1, 2), cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Sobel edge detection
    grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    edges = np.uint8(mag > 15) * 255

    # Morphqology transformations
    morph = cv2.morphologyEx(edges, cv2.MORPH_OPEN, (3, 3), iterations=1)

    # Significant contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = np.zeros_like(frame)
    cv2.drawContours(np.swapaxes(np.swapaxes(img_contours, 0, 1), 1, 2), contours, -1, (0, 255, 0), thickness=2)
    img_contours = cv2.cvtColor(np.swapaxes(np.swapaxes(img_contours, 0, 1), 1, 2), cv2.COLOR_RGB2GRAY)

    # Compute ROIs
    entropies = [frame_entr]
    out = []  # roi list
    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)  # Bounding boxes
        if checkDims(w, h):
            roi = frame[:, y:y + h, x:x + w]

            roi_unrolled = np.concatenate((roi[0], roi[1], roi[2]), axis=None)
            histo = np.float32((np.bincount(roi_unrolled, minlength=256 * 3)))
            histo += 1
            histo /= np.sum(histo)

            entropy = -np.sum(histo * np.log2(histo))

            if entropy > frame_entr - 0.35:
                find = False
                restart = True
                while restart:
                    restart = False
                    for rect in out:
                        # nuovo è contenuto in uno già esistente
                        if x >= rect[0] and y >= rect[1] and x + w < rect[0] + rect[2] and y + h < rect[1] + rect[3]:
                            find = True
                            break
                        # uno già esistente è più piccolo di uno nuovo
                        if rect[0] >= x and rect[1] >= y and rect[0] + rect[2] <= x + w and rect[1] + rect[3] <= y + h:
                            out.remove(rect)
                            restart = True
                if find is False:
                    out.append((x, y, w, h))
                    entropies.append(entropy)

    for rect in out:
        cv2.rectangle(np.swapaxes(np.swapaxes(frame, 0, 1), 1, 2), (rect[0], rect[1]),
                      (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), thickness=2)
    frame_entr = np.sum(entropies) / len(entropies)
    print("entropies = ", entropies)

    # Show results
    print(out)
    horizontal_concat_1 = np.concatenate(
        (cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB), cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)), axis=1)
    horizontal_concat_2 = np.concatenate(
        (cv2.cvtColor(img_contours, cv2.COLOR_GRAY2RGB), np.swapaxes(np.swapaxes(frame, 0, 1), 1, 2)), axis=1)
    vertical_concat = np.concatenate((horizontal_concat_1, horizontal_concat_2), axis=0)
    cv2.imshow('ROIs', cv2.resize(vertical_concat, (1280, 720)))

    # Delay & escape-key
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

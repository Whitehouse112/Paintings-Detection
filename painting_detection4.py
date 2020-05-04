import numpy as np
import cv2


def HW3(img):
    return np.swapaxes(np.swapaxes(img, 0, 1), 1, 2)


def checkDims(_w, _h):
    if _w < 100 or _h < 100:
        return False
    if _w >= 1500 or _h > 1080:
        return False
    if _w / _h > 3.5 or _h / _w > 3.5:
        return False
    return True


def detect_paintings(frame, frame_entr):

    # Blurring
    gray = cv2.cvtColor(HW3(frame), cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Sobel edge detection
    grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    edges = np.uint8(mag > 15) * 255

    # Morphology transformations
    morph = cv2.morphologyEx(edges, cv2.MORPH_OPEN, (3, 3), iterations=1)

    # Significant contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = np.zeros_like(frame)
    cv2.drawContours(HW3(img_contours), contours, -1, (0, 255, 0), thickness=2)
    # img_contours = cv2.cvtColor(HW3(img_contours), cv2.COLOR_RGB2GRAY)

    # Compute ROI
    entropies = [frame_entr]
    roi_list = []
    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)  # Bounding boxes
        if checkDims(w, h):
            roi = frame[:, y:y + h, x:x + w]

            roi_unrolled = np.concatenate((roi[0], roi[1], roi[2]), axis=None)
            av_color = int(np.sum(roi_unrolled) / len(roi_unrolled))

            roi_gray = cv2.cvtColor(HW3(roi), cv2.COLOR_RGB2GRAY)
            roi_gray_unrolled = np.concatenate(roi_gray, axis=None)
            roi_gray_unrolled += 1
            roi_norm = roi_gray_unrolled / np.sum(roi_gray_unrolled)
            roi_norm += 0.00000001e-05
            entropy = -np.sum(roi_norm * np.log2(roi_norm))

            if entropy > frame_entr - 1.5 and av_color < 105:
                # Discard inner rectangles
                find = False
                restart = True
                while restart:
                    restart = False
                    for rect in roi_list:
                        # nuovo è contenuto in uno già esistente
                        if x >= rect[0] and y >= rect[1] and x + w < rect[0] + rect[2] and y + h < rect[1] + rect[3]:
                            find = True
                            break
                        # uno già esistente è più piccolo di uno nuovo
                        if rect[0] >= x and rect[1] >= y and rect[0] + rect[2] <= x + w and rect[1] + rect[3] <= y + h:
                            roi_list.remove(rect)
                            restart = True
                if find is False:
                    roi_list.append((x, y, w, h))
                    entropies.append(entropy)

    for rect in roi_list:
        cv2.rectangle(HW3(frame), (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), thickness=2)
    frame_entr = np.sum(entropies) / len(entropies)
    # print("entropies = ", entropies)

    # Show results
    # print(out)
    # horizontal_concat_1 = np.concatenate(
    #     (cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB), cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)), axis=1)
    # horizontal_concat_2 = np.concatenate(
    #     (cv2.cvtColor(img_contours, cv2.COLOR_GRAY2RGB), HW3(frame)), axis=1)
    # vertical_concat = np.concatenate((horizontal_concat_1, horizontal_concat_2), axis=0)
    # cv2.imshow('ROI', cv2.resize(vertical_concat, (1280, 720)))

    return roi_list, frame, frame_entr
    # return roi_list, img_contours, frame_entr

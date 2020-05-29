import cv2
import numpy as np


def discard_paintings(boxes, roi_list):
    for box in boxes:
        (bx, by, bw, bh) = box
        for roi in roi_list:
            (px, py, pw, ph) = roi
            if bx >= px and by >= py and bx + bw <= px + pw and by + bh <= py + ph:
                boxes.remove(box)
                return discard_paintings(boxes, roi_list)
    return boxes


def detect_people(frame, roi_list):
    path = 'files/'

    # Give the configuration and weight files for the model and load the network
    net = cv2.dnn.readNetFromDarknet(path + 'yolov3.cfg', path + 'yolov3.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

    # determine the output layer
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), crop=False)

    net.setInput(blob)
    outputs = net.forward(ln)

    frame_h, frame_w = frame.shape[:2]
    confidence_threshold = 0.5  # default = 0.5

    boxes, confidences = [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # classID = 0 refers to person class
            if classID == 0 and confidence > confidence_threshold:
                box = detection[:4] * np.array([frame_w, frame_h, frame_w, frame_h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))

    # Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold=confidence_threshold - 0.1)
    indices = np.array(indices)
    boxes = [boxes[i] for i in indices.flatten()]

    boxes = discard_paintings(boxes, roi_list)

    return boxes

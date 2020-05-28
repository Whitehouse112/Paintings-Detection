import cv2
import numpy as np
import time


def detect_people(frame):

    # Give the configuration and weight files for the model and load the network
    net = cv2.dnn.readNetFromDarknet('files/yolov3.cfg', 'files/yolov3.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    
    # determine the output layer
    ln = net.getLayerNames()
    ln_unconnected = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), crop=False)
    
    net.setInput(blob)
    outputs = net.forward(ln_unconnected)

    boxes = []
    confidences = []
    classIDs = []
    h, w = frame.shape[:2]

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if classID == 0 and  confidence > 0.5:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            # cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return frame
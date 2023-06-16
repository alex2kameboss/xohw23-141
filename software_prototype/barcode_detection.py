# -> image -> bbox

import cv2
import numpy as np

confidence_threshold = 0.3
overlapping_threshold = 0.1
iou_threshold = 0.5

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

yolo_width_height = (416, 416)
yolo_config_path = "./yolov4-tiny-custom.cfg"
yolo_weights_path = "./yolov4-tiny-custom_best.weights"

net = cv2.dnn.readNetFromDarknet(yolo_config_path, yolo_weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def barcode_detection(img_in):
    (H, W) = img_in.shape[:2]
    img_in = cv2.resize(img_in, yolo_width_height)
    blob = cv2.dnn.blobFromImage(img_in, 1 / 255.0, yolo_width_height, swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(getOutputsNames(net))
    boxes = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidence_threshold:
                # Scale the bboxes back to the original image size
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
    bboxes = cv2.dnn.NMSBoxes(
        boxes, confidences, confidence_threshold, overlapping_threshold)

    ret = []
    for index in bboxes:
        ret.append([boxes[index][0], boxes[index][1],
                    boxes[index][0] + boxes[index][2], boxes[index][1] + boxes[index][3],
                    confidences[index]])
    return ret


if __name__ == '__main__':
    colors = [(255, 0, 0)]

    cap = cv2.VideoCapture(0)

    try:
        if not cap.isOpened():
            print("Error opening video stream or file")
            exit(1)

        while True:
            (_, frame) = cap.read()

            bboxes = barcode_detection(frame)

            if len(bboxes) > 0:
                for boxes in bboxes:
                    (x, y) = (boxes[0], boxes[1])
                    (w, h) = (boxes[2], boxes[3])
                    color = colors[0]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            cv2.imshow("YOLOv4 Object Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break the loop
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

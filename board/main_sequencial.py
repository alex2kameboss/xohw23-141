from BarcodeDetector_yolov3 import BarcodeDetector
from barcode_tracking import barcode_tracking
from barcode_reader import barcode_reader, get_barcode
import cv2
import numpy as np

i = 0

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    detector = BarcodeDetector()

    try:
        if not cap.isOpened():
            print("Error opening video stream or file")
            exit(1)

        while True:
            (_, frame) = cap.read()

            detector.run(frame)
            boxes = np.array(detector.boxes, dtype=np.int32)
            boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]
            #print("bboxes", boxes)
            barcodes = barcode_tracking(boxes)
            if len(barcodes) > 0:
                for barcode in barcodes:
                    print("barcode:", barcode_reader(get_barcode(frame, barcode)))

            #cv2.imshow("Barcode Detection my", detector.draw(frame))
            if len(boxes) > 0:
                for box in boxes:
                    color = (255, 0, 0)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

            cv2.imshow("Barcode Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break the loop
            if key == ord("q"):
                break
        
    finally:
        cap.release()
        cv2.destroyAllWindows()

import cv2
from barcode_detection import barcode_detection
from barcode_tracking import barcode_tracking
from barcode_reader import barcode_reader, get_barcode

i = 0


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    try:
        if not cap.isOpened():
            print("Error opening video stream or file")
            exit(1)

        while True:
            (_, frame) = cap.read()

            boxes = barcode_detection(frame)
            # print("bboxes", boxes)
            barcodes = barcode_tracking(boxes)
            if len(barcodes) > 0:
               for barcode in barcodes:
                   print("barcode:", barcode_reader(get_barcode(frame, barcode)))

            if len(boxes) > 0:
                for boxes in boxes:
                    color = (255, 0, 0)
                    cv2.rectangle(frame, (boxes[0], boxes[1]), (boxes[2], boxes[3]), color, 2)

            cv2.imshow("YOLOv4 Object Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break the loop
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
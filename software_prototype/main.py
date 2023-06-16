import cv2
from barcode_detection import barcode_detection
from barcode_tracking import barcode_tracking
from barcode_reader import barcode_reader, get_barcode
from multiprocessing import Process
from multiprocessing import Pipe
from serial import Serial
import time

i = 0


def stage0(cap, conn_right):
    while True:
        (_, frame) = cap.read()

        boxes = barcode_detection(frame)
        # print("bboxes", boxes)
        conn_right.send((boxes, frame))
        #barcodes = barcode_tracking(boxes)
        #if len(barcodes) > 0:
        #    for barcode in barcodes:
        #        print("barcode:", barcode_reader(get_barcode(frame, barcode)))

        if len(boxes) > 0:
            for boxes in boxes:
                color = (255, 0, 0)
                cv2.rectangle(frame, (boxes[0], boxes[1]), (boxes[2], boxes[3]), color, 2)

        cv2.imshow("YOLOv4 Object Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break the loop
        if key == ord("q"):
            conn_right.send((None, None))
            break


def stage1(conn_left, conn_right):
    while True:
        boxes, frame = conn_left.recv()
        if boxes is None and frame is None:
            conn_right.send((None, None))
            break
        barcodes = barcode_tracking(boxes)
        conn_right.send((barcodes, frame))


def stage2(serial, conn_left):
    while True:
        barcodes, frame = conn_left.recv()
        if barcodes is None and frame is None:
            break
        if len(barcodes) > 0:
            for barcode in barcodes:
                value = barcode_reader(get_barcode(frame, barcode))
                print(f"barcode: {value}", flush=True)
                if len(value) == 13:
                    serial.write(display.write(bytes(f'+{value}\n', 'utf-8')))


if __name__ == '__main__':
    display = Serial('/dev/ttyACM0', baudrate=115200)
    cap = cv2.VideoCapture(0)
    print(display.write(b'init\n'))
    time.sleep(0.5)

    try:
        if not cap.isOpened():
            print("Error opening video stream or file")
            exit(1)

        conn0_sender, conn0_receiver = Pipe()
        conn1_sender, conn1_receiver = Pipe()

        stage0_process = Process(target=stage0, args=(cap, conn0_sender,))
        stage1_process = Process(target=stage1, args=(conn0_receiver, conn1_sender,))
        stage2_process = Process(target=stage2, args=(display, conn1_receiver,))

        stage0_process.start()
        stage1_process.start()
        stage2_process.start()

        print(display.write(b'start\n'))

        stage0_process.join()
        stage1_process.join()
        stage2_process.join()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        display.close()

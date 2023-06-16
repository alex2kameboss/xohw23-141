# add pynq packages
import sys
sys.path.insert(0, '/usr/local/share/pynq-venv/lib/python3.10/site-packages')
# -----------------

from pynq_dpu import DpuOverlay
import xir
from typing import List
import vart
import vitis_ai_library
import numpy as np
import cv2


# consts
anchor_list = [10,14,  23,27,  37,58,  81,82,  135,169,  344,319]
anchor_float = [float(x) for x in anchor_list]
anchors = np.array(anchor_float).reshape(-1, 2)
class_names = ['barcode']
masks = [[3, 4, 5], [0, 1, 2]]


class BarcodeDetector():
    def __init__(self):
        # upload bistream
        self.overlay = DpuOverlay("dpu_yolo_550.bit")

        # load yolov3 on dpu and create an object for it
        #self.graph = xir.Graph.deserialize("barcode_detection_200000_0.xmodel")
        #self.runner = vitis_ai_library.GraphRunner.create_graph_runner(self.graph)
        self.overlay.load_model("yolov3-tiny-barcode-finetune.xmodel")
        self.runner = self.overlay.runner

        # create buffers

        # get buffers shapes
        input_tensor_buffers = self.runner.get_input_tensors()
        output_tensor_buffers = self.runner.get_output_tensors()

        # input buffer
        self.shapeIn = tuple(input_tensor_buffers[0].dims)
        #self.input_data = [np.empty(shapeIn, dtype=np.int8, order="C")]
        self.input_data = [np.empty(self.shapeIn, dtype=np.float32, order="C")]
        self.image = self.input_data[0]

        # output buffer
        self.shapes = [tuple(output_tensor_buffers[0].dims),
                        tuple(output_tensor_buffers[1].dims)]
        self.output_data = [np.empty(self.shapes[0], dtype=np.float32, order="C"),
                            np.empty(self.shapes[1], dtype=np.float32, order="C")]


    def run(self, image_in):
        # resize image
        #image = cv2.resize(image_in, (224, 224))
        # write to input buffer
        #self.input_data[0][0, ...] = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_data = np.array(pre_process(image_in, (224, 224)), dtype=np.float32)
        self.image[0,...] = image_data.reshape(self.shapeIn[1:])

        # start dpu
        v = self.runner.execute_async(self.input_data, self.output_data)
        self.runner.wait(v)

        yolo_output = [np.reshape(self.output_data[0], self.shapes[0]), 
                        np.reshape(self.output_data[1], self.shapes[1])]

        self.boxes, self.scores, classes = evaluate(yolo_output, image_in.shape[:2], class_names, anchors)


    def draw(self, image):
        return draw_bbox(image, self.boxes, self.scores)


# for input
'''resize image with unchanged aspect ratio using padding'''
def letterbox_image(image, size):
    ih, iw, _ = image.shape
    w, h = size
    scale = min(w/iw, h/ih)
    #print(scale)
    
    nw = int(iw*scale)
    nh = int(ih*scale)
    #print(nw)
    #print(nh)

    image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_LINEAR)
    new_image = np.ones((h,w,3), np.uint8) * 128
    h_start = (h-nh)//2
    w_start = (w-nw)//2
    new_image[h_start:h_start+nh, w_start:w_start+nw, :] = image
    return new_image


'''image preprocessing'''
def pre_process(image, model_image_size):
    image = image[...,::-1]
    image_h, image_w, _ = image.shape
 
    if model_image_size != (None, None):
        assert model_image_size[0]%32 == 0, 'Multiples of 32 required'
        assert model_image_size[1]%32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
    else:
        new_image_size = (image_w - (image_w % 32), image_h - (image_h % 32))
        boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0) 	
    return image_data


# helpers
'''Draw detection frame'''
def draw_bbox(image, bboxes, scores):
    image_h, image_w, _ = image.shape
    bbox_thick = int(0.6 * (image_h + image_w) / 600)

    for i, bbox in enumerate(bboxes):
        [top, left, bottom, right] = np.array(bbox, dtype=np.int32)
        fontScale = 0.5
        class_ind = 0
        bbox_color = (255, 0, 0)
        c1, c2 = (left, top), (right, bottom)
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
        #cv2.putText(image, str(scores[i]), (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bbox_color, 2)
    return image


def _get_feats(feats, anchors, num_classes, input_shape):
    num_anchors = len(anchors)
    anchors_tensor = np.reshape(np.array(anchors, dtype=np.float32), [1, 1, 1, num_anchors, 2])
    grid_size = np.shape(feats)[1:3]
    nu = num_classes + 5
    predictions = np.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, nu])
    grid_y = np.tile(np.reshape(np.arange(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
    grid_x = np.tile(np.reshape(np.arange(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
    grid = np.concatenate([grid_x, grid_y], axis = -1)
    grid = np.array(grid, dtype=np.float32)

    box_xy = (1/(1+np.exp(-predictions[..., :2])) + grid) / np.array(grid_size[::-1], dtype=np.float32)
    box_wh = np.exp(predictions[..., 2:4]) * anchors_tensor / np.array(input_shape[::-1], dtype=np.float32)
    box_confidence = 1/(1+np.exp(-predictions[..., 4:5]))
    box_class_probs = 1/(1+np.exp(-predictions[..., 5:]))
    return box_xy, box_wh, box_confidence, box_class_probs


def correct_boxes(box_xy, box_wh, input_shape, image_shape):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape, dtype = np.float32)
    image_shape = np.array(image_shape, dtype = np.float32)
    new_shape = np.around(image_shape * np.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([
        box_mins[..., 0:1],
        box_mins[..., 1:2],
        box_maxes[..., 0:1],
        box_maxes[..., 1:2]
    ], axis = -1)
    boxes *= np.concatenate([image_shape, image_shape], axis = -1)
    return boxes


def boxes_and_scores(feats, anchors, classes_num, input_shape, image_shape):
    box_xy, box_wh, box_confidence, box_class_probs = _get_feats(feats, anchors, classes_num, input_shape)
    boxes = correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = np.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = np.reshape(box_scores, [-1, classes_num])
    return boxes, box_scores


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 1)
        h1 = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= 0.55)[0]  # threshold
        order = order[inds + 1]

    return keep


def evaluate(yolo_outputs, image_shape, class_names, anchors):
    score_thresh = 0.3
    anchor_mask = [[3, 4, 5], [0, 1, 2]]
    boxes = []
    box_scores = []
    input_shape = np.shape(yolo_outputs[0])[1 : 3]
    input_shape = np.array(input_shape)*32

    for i in range(len(yolo_outputs)):
        _boxes, _box_scores = boxes_and_scores(
            yolo_outputs[i], anchors[anchor_mask[i]], len(class_names), 
            input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = np.concatenate(boxes, axis = 0)
    box_scores = np.concatenate(box_scores, axis = 0)

    mask = box_scores >= score_thresh
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(len(class_names)):
        class_boxes_np = boxes[mask[:, c]]
        class_box_scores_np = box_scores[:, c]
        class_box_scores_np = class_box_scores_np[mask[:, c]]
        nms_index_np = nms_boxes(class_boxes_np, class_box_scores_np) 
        class_boxes_np = class_boxes_np[nms_index_np]
        class_box_scores_np = class_box_scores_np[nms_index_np]
        classes_np = np.ones_like(class_box_scores_np, dtype = np.int32) * c
        boxes_.append(class_boxes_np)
        scores_.append(class_box_scores_np)
        classes_.append(classes_np)
    boxes_ = np.concatenate(boxes_, axis = 0)
    scores_ = np.concatenate(scores_, axis = 0)
    classes_ = np.concatenate(classes_, axis = 0)

    return boxes_, scores_, classes_

import argparse
import os
import sys
import os.path as osp
import cv2

import torch

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.utils.events import LOGGER
from yolov6.core.inferer import Inferer

CLASSES = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]

def get_inferer(
        weights=osp.join(ROOT, 'yolov6/yolov6Data/yolov6n.pt'),
        source=osp.join(ROOT, 'yolov6/yolov6Data/data/images'),
        yaml=osp.join(ROOT, 'yolov6/yolov6Data/data/coco.yaml'),
        img_size=640,
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device='0',
        classes=None,
        agnostic_nms=False,
        project=osp.join(ROOT, 'runs/inference'),
        name='exp',
        hide_labels=False,
        hide_conf=False,
        half=False,
        ):
    print("loading inferer")
    inferer = Inferer(source, weights, device, yaml, img_size, half)
    return inferer
            

@torch.no_grad()
def classify(
        img_input,
        inferer,
        weights=osp.join(ROOT, 'yolov6Data/yolov6n.pt'),
        source=osp.join(ROOT, 'yolov6Data/data/images'),
        yaml=osp.join(ROOT, 'yolov6Data/data/coco.yaml'),
        img_size=640,
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device='0',
        classes=None,
        agnostic_nms=False,
        project=osp.join(ROOT, 'runs/inference'),
        name='exp',
        hide_labels=False,
        hide_conf=False,
        half=False,
        ):

    # Inference
    detections = inferer.infer(conf_thres, iou_thres, classes, agnostic_nms, max_det, hide_labels, hide_conf, img_input)

    return detections


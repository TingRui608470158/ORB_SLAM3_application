#!/usr/bin/env python

# import sys
# sys.path.append("/home/aa/Desktop/test_ws/src/beginner_tutorials/scripts/SSD/ssd/")


# rosrun beginner_tutorials demo.py --config-file src/beginner_tutorials/src/configs/vgg_ssd300_voc0712.yaml --images_dir src/beginner_tutorials/src/demo --ckpt https://github.com/lufficc/SSD/releases/download/1.2/vgg_ssd300_voc0712.pth

import glob
import os
import time

import torch
import PIL.Image as im
from vizer.draw import draw_boxes

from ssd.config import cfg
from ssd.data.datasets import COCODataset, VOCDataset
import argparse
import numpy as np

from ssd.data.transforms import build_transforms
from ssd.modeling.detector import build_detection_model
from ssd.utils import mkdir
from ssd.utils.checkpoint import CheckPointer

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError

from beginner_tutorials.msg import mobilenetv2ssd

pub_mobilenetv2_ssd = rospy.Publisher('/mobilenetv2_ssd', mobilenetv2ssd, queue_size=1000)

print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
@torch.no_grad()
def run_demo(cfg, ckpt, score_threshold, images_dir, output_dir, dataset_type):
    msg = mobilenetv2ssd()
    if dataset_type == "voc":
        class_names = VOCDataset.class_names
    elif dataset_type == 'coco':
        class_names = COCODataset.class_names
    else:
        raise NotImplementedError('Not implemented now.')
    device = torch.device(cfg.MODEL.DEVICE)

    model = build_detection_model(cfg)
    model = model.to(device)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))
    mkdir(output_dir)

    cpu_device = torch.device("cpu")
    transforms = build_transforms(cfg, is_train=False)
    model.eval()
    while not rospy.is_shutdown():
        for i, image_path in enumerate(image_paths):
            start = time.time()
            image_name = os.path.basename(image_path)

            image = np.array(im.open(image_path).convert("RGB"))
            height, width = image.shape[:2]
            images = transforms(image)[0].unsqueeze(0)
            load_time = time.time() - start

            start = time.time()
            result = model(images.to(device))[0]
            inference_time = time.time() - start

            result = result.resize((width, height)).to(cpu_device).numpy()
            boxes, labels, scores = result['boxes'], result['labels'], result['scores']

            for class_n in range(len(boxes)):
                msg.box0 = boxes[class_n][0]
                msg.box1 = boxes[class_n][1]
                msg.box2 = boxes[class_n][2]
                msg.box3 = boxes[class_n][3]
                msg.ids = labels[class_n]
                msg.score = scores[class_n]
                pub_mobilenetv2_ssd.publish(msg)
            indices = scores > score_threshold
            boxes = boxes[indices]
            labels = labels[indices]
            scores = scores[indices]
            print("boxes = ",boxes)
            print("labels = ",labels)
            print("scores = ",scores)
            meters = ' | '.join(
                [
                    'objects {:02d}'.format(len(boxes)),
                    'load {:03d}ms'.format(round(load_time * 1000)),
                    'inference {:03d}ms'.format(round(inference_time * 1000)),
                    'FPS {}'.format(round(1.0 / inference_time))
                ]
            )
            print('({:04d}/{:04d}) {}: {}'.format(i + 1, len(image_paths), image_name, meters))

            # drawn_image = draw_boxes(image, boxes, labels, scores, class_names).astype(np.uint8)
            # im.fromarray(drawn_image).save(os.path.join(output_dir, image_name))


class image_converter:

  def __init__(self):
    # self.image_pub = rospy.Publisher("image_topic_2",Image)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/usb_cam/image_raw",Image,self.callback)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    # (rows,cols,channels) = cv_image.shape
    # if cols > 60 and rows > 60 :
    #   cv2.circle(cv_image, (50,50), 10, 255)

    cv2.imshow("Image window", cv_image)
    cv2.waitKey(3)

    # try:
    #   self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    # except CvBridgeError as e:
    #   print(e)



def main():
    rospy.init_node('MobileNetV2-SSD', anonymous=True)
    
    ic = image_converter()   

    parser = argparse.ArgumentParser(description="SSD Demo.")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Trained weights.")
    parser.add_argument("--score_threshold", type=float, default=0.7)
    parser.add_argument("--images_dir", default='demo', type=str, help='Specify a image dir to do prediction.')
    parser.add_argument("--output_dir", default='demo/result', type=str, help='Specify a image dir to save predicted images.')
    parser.add_argument("--dataset_type", default="voc", type=str, help='Specify dataset type. Currently support voc and coco.')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    print(args)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        print(config_str)
    print("Running with config:\n{}".format(cfg))

    run_demo(cfg=cfg,
             ckpt=args.ckpt,
             score_threshold=args.score_threshold,
             images_dir=args.images_dir,
             output_dir=args.output_dir,
             dataset_type=args.dataset_type)

    
    rospy.spin()
if __name__ == '__main__':
    main()

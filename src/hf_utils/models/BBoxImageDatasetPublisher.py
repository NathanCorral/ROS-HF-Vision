"""
Datasets todo:
	# https://www.kaggle.com/datasets/dansbecker/cityscapes-image-pairs
	# https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/code
	# https://huggingface.co/datasets/keremberke/german-traffic-sign-detection
This file focuses on:
	https://huggingface.co/datasets/keremberke/german-traffic-sign-detection
"""
# Ros
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.executors import MultiThreadedExecutor
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from std_msgs.msg import Header
import cv2

# Hugging Face/torch
from datasets import load_dataset

# Other
import os
import numpy as np

# See Base Class for pub/sub functions.  
from models.ModelNode import ModelNode


class BBoxImageDatasetPublisher(ModelNode):
    """
    A ROS2 node for publishing images/gt bboxes from:
    	https://huggingface.co/datasets/keremberke/german-traffic-sign-detection
    """
    def __init__(self, node_name='dataset_german_traffic_sign'):
        """
        Initialize the BBoxImageDatasetPublisher node.
        
        Parameters:
            node_name (str): Name of the ROS2 node. Default is 'https://huggingface.co/datasets/keremberke/german-traffic-sign-detection'.
        """
        super().__init__(node_name=node_name)

        self.declare_parameter('pretrained_model_name_or_path', 'keremberke/german-traffic-sign-detection')
        pretrained_model_name_or_path = self.get_parameter('pretrained_model_name_or_path').get_parameter_value().string_value

        self.get_logger().info(f"Loading Dataset:  {pretrained_model_name_or_path}")
        self.ds = load_dataset(pretrained_model_name_or_path, name="full")

        # Actually the "seg_map" publisher will publish that dataset to images...
        	# And use bbox for ground truth bounding boxes
        topic_basename = os.path.dirname(pretrained_model_name_or_path)
        self.create_seg_map_publisher(topic=topic_basename + "/image")
        self.create_bb_publisher(topic=topic_basename + "/gt_bbox")

        # Indicate the id2label mapping
        self.id2label = {i: name for i,name in enumerate(self.ds["train"].features["objects"].feature["category"].names)}
        self.spawn_model_metadata(pretrained_model_name_or_path, self.id2label)

        # Create a timer for publishing 
        fps = 0.5
        self.timer_period = 1./fps
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # Current Index
        self.idx = 0
        self.split = "train"

    def suffle(self):
        pass

    def select_split(self, split="train"):
        pass

    def __len__(self):
        return len(self.ds[self.split])

    def timer_callback(self):
        ex = self.ds[self.split][self.idx % len(self)]
        # self.get_logger().info(f"Timer Callback example:  {ex}")

        # Create the header
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        
        # Create a publish the image
        im_msg = self.convert_pil_to_image_msg(ex["image"], header=header)
        self.seg_publisher.publish(im_msg)

        # Create and publish the bbox ground truth
        bbox_msg = self.convert_bbox_to_msg(ex["objects"], header=header)
        bbox_msg = self.map_bbox_labels(bbox_msg) # re-index labels
        self.bbox_publisher.publish(bbox_msg)

        self.idx += 1

    def convert_pil_to_image_msg(self, pil_image, header=None):
        open_cv_image = np.array(pil_image) 
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)        
        ros_image = self.seg_map_bridge.cv2_to_imgmsg(open_cv_image, encoding="bgr8")
        if header:
            ros_image.header = header
        return ros_image

    def convert_bbox_to_msg(self, hf_bbox, header=None):
        detection_array_msg = Detection2DArray()
        if header:
            detection_array_msg.header = header

        for _id, area, bbox, category in zip(hf_bbox["id"], hf_bbox["area"], hf_bbox["bbox"], hf_bbox["category"]):
            detection_msg = Detection2D()
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(category)
            hypothesis.hypothesis.score = float(1.0) # 100 % confidence in ground truth
            detection_msg.results.append(hypothesis)

            bbox_msg = BoundingBox2D()
            center_x, center_y, size_x, size_y = self.convert_bbox_format_0(bbox)
            bbox_msg.center.position.x = center_x
            bbox_msg.center.position.y = center_y
            bbox_msg.center.theta = 0.
            bbox_msg.size_x = size_x
            bbox_msg.size_y = size_y
            detection_msg.bbox = bbox_msg

            detection_array_msg.detections.append(detection_msg)

        # self.bbox_publisher.publish(detection_array_msg)
        return detection_array_msg

    def convert_bbox_format_0(self, bbox):
        """
        Convert bounding box format from (x, y, width, height) to (center_x, center_y, size_x, size_y).
        
        Parameters:
        bbox (list): List of bounding boxes in the format (x, y, width, height).
        
        Returns:
        list: Bounding boxes in the format (center_x, center_y, size_x, size_y).
        """
        x, y, width, height = bbox
        center_x = x + width / 2
        center_y = y + height / 2
        size_x = width
        size_y = height
        return center_x, center_y, size_x, size_y

def main(args=None):
    rclpy.init(args=args)
    node = BBoxImageDatasetPublisher()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

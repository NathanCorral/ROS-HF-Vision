import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.qos import qos_profile_sensor_data
from ament_index_python.packages import get_package_share_directory

import json
import os
import cv2
from cv_bridge import CvBridge, CvBridgeError


class DetectionVisualizer(Node):
    def __init__(self):
        super().__init__('detection_visualizer')

        # Create subscribers for Image and Detection2DArray messages
        self.image_sub = Subscriber(self, Image, 'camera/image', qos_profile=qos_profile_sensor_data)
        self.detection_sub = Subscriber(self, Detection2DArray, 'obj_det/detections')

        # ApproximateTimeSynchronizer with a queue size of 10 and 0.1 seconds slop
        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.detection_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.callback)

        self.bridge = CvBridge()

        # Load data for convering id to string
        package_share_directory = get_package_share_directory('detr')
        path = os.path.join(package_share_directory, 'coco2017_id2label.json')
        self.id2label = self.load_json_file(path)["id2label"]
        # print(self.id2label)

    def load_json_file(self, filepath):
        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
                return data
        except (FileNotFoundError, json.JSONDecodeError):
            self.get_logger().warn(f'Unable to open and read file: {path}')
            return None

    def callback(self, image, detections):
        # self.get_logger().info('Received synchronized messages')
        try:
            frame = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().warn(f'Could not convert image: {e}')
            return

        # Draw the bounding boxes

        for detection in detections.detections:
            bbox = detection.bbox
            x_min = int(bbox.center.position.x - bbox.size_x / 2)
            y_min = int(bbox.center.position.y - bbox.size_y / 2)
            x_max = int(bbox.center.position.x + bbox.size_x / 2)
            y_max = int(bbox.center.position.y + bbox.size_y / 2)
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

            for result in detection.results:
                score = result.hypothesis.score
                label = result.hypothesis.class_id
                if self.id2label is not None:
                    label_text = f"{self.id2label[label]}: {score:.2f}"
                else:
                    label_text = f"{str(label)}: {score:.2f}"
                cv2.putText(frame, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        cv2.imshow("Detections", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = DetectionVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

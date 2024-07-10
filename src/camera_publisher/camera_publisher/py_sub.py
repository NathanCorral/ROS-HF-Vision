#! /usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from rclpy.qos import qos_profile_sensor_data


class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber_py')
        
        self.declare_parameter('image_topic', 'camera/image')
        topic = self.get_parameter('image_topic').get_parameter_value().string_value

        self.subscription = self.create_subscription(
            Image,
            topic,
            self.listener_callback,
            qos_profile_sensor_data)
        self.subscription  # prevent unused variable warning
        
        self.bridge = CvBridge()

    def listener_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imshow("Camera Image", cv_image)
            cv2.waitKey(1)
        except CvBridgeError as e:
            self.get_logger().error(f'Could not convert image: {e}')

def main(args=None):
    rclpy.init(args=args)
    camera_subscriber = CameraSubscriber()

    try:
        rclpy.spin(camera_subscriber)
    except KeyboardInterrupt:
        pass

    camera_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
#! /usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from rclpy.qos import qos_profile_sensor_data

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher_py')

        self.declare_parameter('image_topic', "camera/image") 
        self.declare_parameter('camera_index', 0) 
        self.declare_parameter('hz', 10) 

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        cap_index = self.get_parameter('camera_index').get_parameter_value().integer_value
        hz = self.get_parameter('hz').get_parameter_value().integer_value
    
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(cap_index, cv2.CAP_V4L)
        self.pub = self.create_publisher(Image, image_topic, qos_profile_sensor_data)
        self.pub_timer = self.create_timer(1./hz, self.timer_callback)
        
        # self.cap.set(3,640)
        # self.cap.set(4,480)

    def timer_callback(self):
        try:
            ret, frame = self.cap.read()
            if ret:
                """
                Encodings: ["bgr8", "rgb8", "mono8"]
                """
                msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                self.pub.publish(msg)
        except CvBridgeError as e:
            self.get_logger().error(e)

    def __del__(self):
        self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    camera_publisher = CameraPublisher()

    try:
        rclpy.spin(camera_publisher)
    except KeyboardInterrupt:
        pass

    camera_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
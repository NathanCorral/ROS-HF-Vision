import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from rclpy.qos import qos_profile_sensor_data
from vision_msgs.msg import Detection2DArray


class ObjectDetectionNode(Node):
    """
    A ROS2 node for object detection that bundles together:
        - Image subscriber
        - Bounding Box Array Publisher
    """
    def __init__(self, node_name='obj_det'):
        """
        Initialize the ObjectDetectionNode.
        
        Parameters:
            node_name (str): Name of the ROS2 node. Default is 'obj_det'.
        """
        super().__init__(node_name=node_name)
        self.declare_parameter('image_topic', 'camera/image')
        self.declare_parameter('bbox_topic', 'obj_det/detections')        
        self.bridge = CvBridge()

    def create_bb_publisher(self):
        """
        Create a publisher for bounding box arrays.
        """
        topic = self.get_parameter('bbox_topic').get_parameter_value().string_value
        self.bbox_publisher = self.create_publisher(Detection2DArray, topic, 10)

    def create_image_callback(self):
        """
        Create a subscriber for images.
        """
        topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.subscription = self.create_subscription(
            Image,
            topic,
            self.image_callback,
            qos_profile_sensor_data)
        self.subscription  # prevent unused variable warning

    def image_callback(self, msg):
        """
        Callback function for image messages. Should be overwritten by a subclass.
        
        Parameters:
            msg (Image): The ROS2 Image message.
        """
        self.get_logger().warn(f'image_callback function from class: {type(self)}  should be overwritten by inherited class')
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return

        print(f'Shape: {cv_image.shape}, dtype: {cv_image.dtype}, max/min: {cv_image.max()}/{cv_image.min()}')
        cv2.imshow("Image Callback", cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    obj_det_node = ObjectDetectionNode()
    
    try:
        rclpy.spin(obj_det_node)
    except KeyboardInterrupt:
        pass

    obj_det_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

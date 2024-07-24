# ROS
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from rclpy.qos import qos_profile_sensor_data
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D

# Other
import numpy as np

class ModelNode(Node):
    """
    A ROS2 node for object detection that bundles together:
        - Image Subscribers 
        - Bounding Box Array Publisher
        - Segmentation Map publisher
    """
    def __init__(self, node_name='model_node'):
        """
        Initialize the ModelNode.
        
        Parameters:
            node_name (str): Name of the ROS2 node. Default is 'obj_det'.
        """
        super().__init__(node_name=node_name)

    def create_bb_publisher(self):
        """
        Create a publisher for bounding box arrays.
        """
        self.declare_parameter('bbox_topic', 'hf_model/detections')        
        topic = self.get_parameter('bbox_topic').get_parameter_value().string_value
        self.bbox_publisher = self.create_publisher(Detection2DArray, topic, 10)

    def create_seg_map_publisher(self):
        """
        Create a publisher for segmentation maps.
        """
        self.declare_parameter('seg_map_topic', 'hf_model/seg_map')        
        topic = self.get_parameter('seg_map_topic').get_parameter_value().string_value
        self.seg_publisher = self.create_publisher(Image, topic, 10)
        self.seg_map_bridge = CvBridge()


    def create_image_callback(self):
        """
        Create a subscriber for images.
        """
        self.declare_parameter('image_topic', 'camera/image')
        topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.subscription = self.create_subscription(
            Image,
            topic,
            self.image_callback,
            qos_profile_sensor_data)
        self.subscription  # prevent unused variable warning
        self.im_callback_bridge = CvBridge()

    def image_callback(self, msg):
        """
        Callback function for image messages. Should be overwritten by a subclass.
        
        Parameters:
            msg (Image): The ROS2 Image message.
        """
        # self.get_logger().warn(f'image_callback function from class: {type(self)}  should be overwritten by inherited class')
        try:
            cv_image = self.callback_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return

        self.get_logger().info(f'Shape: {cv_image.shape}, dtype: {cv_image.dtype}, max/min: {cv_image.max()}/{cv_image.min()}')
        cv2.imshow("Image Callback", cv_image)
        cv2.waitKey(1)

    def create_detections_msg(self, header, results) -> Detection2DArray:
        """
        Transform detection results as a Detection2DArray message.
        
        Parameters:
            header (Header): The ROS2 message header.
            results (dict): Dictionary containing detection results.
                    results["scores"] : list() -- Confidence value
                    results["labels"] : list() -- Integer Class label
                    results["boxes"] : list() -- List of pixel corrdinates corresponding to
                        [0]:  ...
                        [1]:  ...
                        [2]:  ...
                        [3]:  ...

        :return: Detection2DArray
        """
        detection_array_msg = Detection2DArray()
        detection_array_msg.header = header

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detection_msg = Detection2D()
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(label)
            hypothesis.hypothesis.score = float(score)
            detection_msg.results.append(hypothesis)

            bbox = BoundingBox2D()
            bbox.center.position.x = (box[0] + box[2]) / 2.0
            bbox.center.position.y = (box[1] + box[3]) / 2.0
            bbox.center.theta = 0.
            bbox.size_x = float(box[2] - box[0])
            bbox.size_y = float(box[3] - box[1])
            detection_msg.bbox = bbox

            detection_array_msg.detections.append(detection_msg)

        # self.bbox_publisher.publish(detection_array_msg)
        return detection_array_msg

    def create_seg_mask_msg(self, header, mask : np.array, encoding="mono8") -> Image:
        """
        Transform seg results as a Image message.
        
        Parameters:
            header (Header): The ROS2 message header.
            mask (np.array): Integer result of the image segmentation.
            encoding (str):  Encoding in ["mono8", "mono16"]

        :return: sensor_msgs.Image
        """
        assert encoding in ["mono8", "mono16"]

        if encoding == "mono8":
            mask = mask.astype(np.uint8, copy=False)
        else:
            mask = mask.astype(np.uint16, copy=False)

        seg_msg = self.seg_map_bridge.cv2_to_imgmsg(mask, encoding=encoding)
        seg_msg.header = header
        return seg_msg

def main(args=None):
    rclpy.init(args=args)
    model_node = ModelNode()
    model_node.create_image_callback()
    
    try:
        rclpy.spin(model_node)
    except KeyboardInterrupt:
        pass

    model_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

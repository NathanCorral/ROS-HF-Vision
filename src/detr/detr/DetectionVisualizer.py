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
import imageio 

class DetectionVisualizer(Node):
    """
    A ROS2 node for visualizing detections by synchronizing image and detection messages.
    """
    def __init__(self):
        """
        Initialize the DetectionVisualizer node.
        """
        super().__init__('detection_visualizer')

        # Declare a parameter that can be set to instruct the node to make a .gif of the filename
        self.declare_parameter('create_gif', 'output.gif')
        self.clear_parameter()
        self.frames_rgb = []

        # Create subscribers for Image and Detection2DArray messages
        self.declare_parameter('image_topic', 'camera/image')
        self.declare_parameter('bbox_topic', 'obj_det/detections')        
        topic_image = self.get_parameter('image_topic').get_parameter_value().string_value
        topic_bbox = self.get_parameter('bbox_topic').get_parameter_value().string_value
        self.image_sub = Subscriber(self, Image, topic_image, qos_profile=qos_profile_sensor_data)
        self.detection_sub = Subscriber(self, Detection2DArray, topic_bbox)

        # ApproximateTimeSynchronizer with a queue size of 10 and 0.1 seconds slop
        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.detection_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.callback)

        self.bridge = CvBridge()

        # Load data for convering id to string
        package_share_directory = get_package_share_directory('detr')
        path = os.path.join(package_share_directory, 'coco2017_id2label.json')
        self.id2label = self.load_json_file(path)["id2label"]

        # Approximate the real object detection fps
        self.prev_time = None
        self.time_diffs = []
        self.time_diffs_sum = 0.

    def clear_parameter(self):
        """
        Instruct the node to clear the parameter server (so a gif is not created).
        A gif is created whenever the 'create_gif' parameter is ending in a .gif
        """
        cleared_param = rclpy.parameter.Parameter(
            'create_gif',
            rclpy.Parameter.Type.STRING,
            ''
        )
        all_new_parameters = [cleared_param]
        self.set_parameters(all_new_parameters)
        self.frames_rgb = []

    def create_gif(self):
        """
        Create a gif over the last seen frames (drawn with bounding boxes).

        Only is run when the "create_gif" ROS parameter ends with .gif
        """
        filename = self.get_parameter('create_gif').get_parameter_value().string_value
        if not filename.endswith(".gif"):
            return

        self.get_logger().info(f'Saving gif of bbox frames to:  {filename}')
        imageio.mimsave(filename, self.frames_rgb, fps=self.estimate_fps()) 
        self.get_logger().info(f'Writing Gif Finished')
        self.clear_parameter()

    def load_json_file(self, filepath):
        """
        Load a JSON file from the specified filepath.
        
        Parameters:
            filepath (str): Path to the JSON file.
            
        Returns:
            dict: Parsed JSON data.
        """
        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
                return data
        except (FileNotFoundError, json.JSONDecodeError):
            self.get_logger().warn(f'Unable to open and read file: {path}')
            return None

    def callback(self, image, detections):
        """
        Callback function to handle synchronized image and detection messages.
        
        Parameters:
            image (Image): The ROS2 Image message.
            detections (Detection2DArray): The ROS2 Detection2DArray message.
        """
        # self.get_logger().info('Received synchronized messages')
        try:
            frame = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().warn(f'Could not convert image: {e}')
            return

        # Draw the bounding boxes
        for detection in detections.detections:
            frame = self.draw_bbox(frame, detection.bbox, detection.results)

        # Draw the fps
        fps = self.estimate_fps(image.header)
        frame = self.overlay_text(frame, fps)

        cv2.imshow("Detections", frame)
        cv2.waitKey(1)
        self.frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.create_gif()

    def draw_bbox(self, frame, bbox, results=None):
        """
        Draw a bbox on the frame.

        Parameters:
            frame (numpy.ndarray): BGR opencv image representation.
            bbox (vision_msgs.BoundingBox2D): The detection box to draw
            result (vision_msgs.ObjectHypothesisWithPose):  (Optional) the label and classification score

        Returns:
            numpy.ndarray: The image frame with the bounding box drawn in.
        """
        x_min = int(bbox.center.position.x - bbox.size_x / 2)
        y_min = int(bbox.center.position.y - bbox.size_y / 2)
        x_max = int(bbox.center.position.x + bbox.size_x / 2)
        y_max = int(bbox.center.position.y + bbox.size_y / 2)
        
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

        if results is not None:
            sorted_results = sorted(results, key=lambda result: result.hypothesis.score, reverse=True)
        else:
            sorted_results = []

        for result in sorted_results:
            score = result.hypothesis.score
            label = result.hypothesis.class_id
            if self.id2label is not None:
                label_text = f"{self.id2label[label]}: {score:.2f}"
            else:
                label_text = f"{str(label)}: {score:.2f}"
            cv2.putText(frame, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # Place only the highest probability
            break

        return frame

    def overlay_text(self, frame, fps):
        """
        Draws a small text and a number at the top right corner of the given image frame to indicate the fps

        Parameters:
            frame (numpy.ndarray): The image frame on which to draw the text.
            fps (float or int): The number representing the FPS to be drawn alongside the text.

        Returns:
            numpy.ndarray: The image frame with the text and FPS drawn at the top right corner.
        """
        # Define the text to be displayed
        text = f"FPS: {fps:.2f}"
        
        # Define the font, size, and color
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 0, 0)  # Black color
        thickness = 1
        
        # Get the width and height of the text box
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Set the text position at the top right corner
        text_x = frame.shape[1] - text_size[0] - 10  # 10 pixels from the right edge
        text_y = text_size[1] + 10  # 10 pixels from the top edge
        
        # Overlay the text on the frame
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
        
        return frame

    def estimate_fps(self, header=None):
        """
        Use the header to get an estimate of the fps.
        Use last 100 samples to smooth the fps 

        Parameters:
            header (ROS msg header): (Optional) ROS header with time stamp.  
                    If None will not update the running estimate. 

        Returns:
            float: The fps
        """
        fps = 0
        
        # Update the current estimate with a new header
        if header is not None:
            current_time = header.stamp.sec + header.stamp.nanosec/1e+9

            # Calculate the difference from the 2nd timestamp
            if self.prev_time is not None:
                time_diff = current_time - self.prev_time
                self.time_diffs.append(time_diff)
                self.time_diffs_sum += time_diff
                if len(self.time_diffs) > 100: 
                    self.time_diffs_sum -= self.time_diffs.pop(0)

            self.prev_time = current_time

        # Set fps after we have a few samples
        if len(self.time_diffs) >= 5:
            avg_time_diff = self.time_diffs_sum / len(self.time_diffs)
            fps = 1.0 / (avg_time_diff + 1e-5) # Avoid divide by zero

        return fps

def main(args=None):
    rclpy.init(args=args)
    node = DetectionVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

# Ros
import rclpy
from rclpy.time import Time
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import MultiThreadedExecutor
from ament_index_python.packages import get_package_share_directory
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
# Callback types
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge, CvBridgeError

# Other
from datetime import datetime, timedelta

# viz data
from matplotlib_viewer.MatPlotLibViz import MatPlotLibViz

# Custom service
from id2label_mapper_services.srv import GetID2Label

class VizNode(Node, MatPlotLibViz):
    """
    A ROS2 wrapper for subscribing to messages and performing visualization functions 
        using matplotlib animate as teh backend.
    ImageDataManager removes dependency for message_filters.ApproximateTimeSynchronizer
    """
    def __init__(self, node_name='viz'):
        """
        Initialize the VizNode.
        
        Parameters:
            node_name (str): Name of the ROS2 node. Default is 'viz'.
        """
        # Initialize ros node to get parameters
        Node.__init__(self, node_name=node_name)
        
        self.declare_parameter('create_gif', 'output.gif')
        self.clear_create_gif_param()

        self.declare_parameter('image_topic', 'camera/image')
        self.declare_parameter('bbox_topic', 'hf_model/detections')
        self.declare_parameter('seg_map_topic', 'hf_model/seg_map')
        self.declare_parameter('live_display', True)
        self.declare_parameter('live_fps', 10)

        # Initialize the plotter
        live_display = self.get_parameter('live_display').get_parameter_value().bool_value
        MatPlotLibViz.__init__(self, live_display=live_display)

        # Approximate the real object detection fps
        self.prev_time = None
        self.time_diffs = []
        self.time_diffs_sum = 0.

        # Create subscribers
        self.bridge = CvBridge()
        topic_image = self.get_parameter('image_topic').get_parameter_value().string_value
        topic_bbox = self.get_parameter('bbox_topic').get_parameter_value().string_value
        topic_seg = self.get_parameter('seg_map_topic').get_parameter_value().string_value
        self.image_sub = self.create_subscription(Image, topic_image, self.image_callback, qos_profile=qos_profile_sensor_data)
        self.seg_map_sub = self.create_subscription(Image, topic_seg, self.seg_map_callback, 10)
        self.bbox_sub = self.create_subscription(Detection2DArray, topic_bbox, self.bbox_callback, 10)
        
        # Create a timer which will display the images if live
        live_fps = self.get_parameter('live_fps').get_parameter_value().integer_value
        self.timer_period = 1./live_fps
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # Create a service and timer to retrieve the "id2label" mapping
        id2label_callback_group = MutuallyExclusiveCallbackGroup()
        self.id2label_client = self.create_client(GetID2Label, 'get_id_to_label', callback_group=id2label_callback_group)
        id2label_timer_callback_group = MutuallyExclusiveCallbackGroup()
        self._get_id2_label_timer = self.create_timer(5, self._get_id2_label, callback_group=id2label_timer_callback_group)
        self.id2label = { }
        self.map_database_version = -1
        self.service_found = False

    def _get_id2_label(self):
        """
        Timer callback that queries id2label server for any updates
        """
        if not self.service_found and not self.id2label_client.wait_for_service(timeout_sec=0):
            self.get_logger().info('Waiting for id2label service')
            return
        self.service_found = True
        request = GetID2Label.Request()
        future = self.id2label_client.call_async(request)
        future.add_done_callback(self._id2label_future_callback)

    def _id2label_future_callback(self, future):
        """
        Service callback that applies and id2label update

        :param future: ROS2 future continaing GetID2Label.response object
        """
        try:
            response = future.result()
            if response.database_version > 0 and self.map_database_version != response.database_version:
                self.id2label = {elem.class_id: elem.class_name for elem in response.class_map}
                self.map_database_version = response.database_version
                self.get_logger().debug(f'Updated id2label Version:  {self.map_database_version}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

    def timer_callback(self):
        """
        Fast timer callbakc function to:
            - Update the live display
            - Check if we should create a .gif
        """
        # self.get_logger().info(f'Image FPS:  {self.estimate_fps()}')
        if self.live and len(self.data_manager) >= 1:
            self.update(id2label=self.id2label)
        self.check_create_gif()

    def bbox_callback(self, detections : Detection2DArray) -> None:
        """ Callback function for adding a detection bbox """
        timestamp = self.ros_time_to_datetime(detections.header.stamp)
        self.add_bbox(detections, timestamp)

    def image_callback(self, msg : Image) -> None:
        """ Callback function for adding an image """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except CvBridgeError as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return

        # self.get_logger().info(f'Shape: {cv_image.shape}, dtype: {cv_image.dtype}, max/min: {cv_image.max()}/{cv_image.min()}')
        timestamp = self.ros_time_to_datetime(msg.header.stamp)
        self.add_image(cv_image, timestamp)
        _ = self.estimate_fps(msg.header)
        # self.get_logger().info(f'Estimated FPS: {self.estimate_fps(None)}')

    def seg_map_callback(self, msg : Image) -> None:
        """ Callback function for adding a segmentation mask to the image"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono16')
        except CvBridgeError as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return

        # self.get_logger().info(f'Shape: {cv_image.shape}, dtype: {cv_image.dtype}, max/min: {cv_image.max()}/{cv_image.min()}')
        timestamp = self.ros_time_to_datetime(msg.header.stamp)
        self.add_mask(cv_image, timestamp)

    def ros_time_to_datetime(self, ros_time):
        """
        Convert ROS time to Python datetime.
        :param ros_time: ROS time object with 'secs' and 'nsecs' attributes.
        :return: Corresponding Python datetime object.
        """
        # Create a datetime object from the seconds part
        dt = datetime.utcfromtimestamp(ros_time.sec)
        # Add the nanoseconds part as a timedelta
        dt += timedelta(microseconds=ros_time.nanosec / 1000.0)
        return dt

    def clear_create_gif_param(self):
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

    def check_create_gif(self):
        """
        Create a gif over the last seen frames (drawn with bounding boxes).

        Only is run when the "create_gif" ROS parameter ends with .gif
        """
        filename = self.get_parameter('create_gif').get_parameter_value().string_value
        if not filename.endswith(".gif"):
            return

        fps = self.estimate_fps()
        self.get_logger().info(f'Saving gif of bbox frames to:  {filename}')
        self.get_logger().info(f'                    with fps:  {fps}')
        self.create_gif(filename, fps=fps)
        self.get_logger().info(f'Writing Gif Finished')
        self.clear_create_gif_param()

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
                    
                # When replaying rosbags, the current time could be restet.
                # In this case just set the prev time and continue as normal
                if time_diff < 0:
                    self.prev_time = current_time
                    return 0

                self.time_diffs.append(time_diff)
                self.time_diffs_sum += time_diff
                if len(self.time_diffs) > 100: 
                    self.time_diffs_sum -= self.time_diffs.pop(0)
                
                # self.get_logger().warn(f'Time Diff: {time_diff}')
            self.prev_time = current_time

        # Set fps after we have a few samples
        if len(self.time_diffs) >= 5:
            avg_time_diff = self.time_diffs_sum / len(self.time_diffs)
            fps = 1.0 / (avg_time_diff + 1e-5) # Avoid divide by zero

        return fps


def main(args=None):
    rclpy.init(args=args)
    node = VizNode()

    # MultiThreadedExecutor only works without live
    if not node.live:
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')
        executor = MultiThreadedExecutor()
        executor.add_node(node)
    else:
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
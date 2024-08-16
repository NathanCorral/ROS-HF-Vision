# ROS
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.srv import GetParameters
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from ament_index_python.packages import get_package_share_directory

#
from id2label_mapper_services.srv import RegisterDatasetMapJSON, GetLocaltoGlobalMap, GetID2Label, GetDatasetID2Label

# Other
import os
import cv2
import numpy as np
import json


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
        self.init_id2label_srvs()

    #
    # id2label services
    #
    def init_id2label_srvs(self):
        """
        Initialize services for id2label_node
        """
        # Name of the dataset that the model was pretrained on.
        self.dataset_name = None
        self.dataset_map = None
        self.dataset_map_version = -1

        # Update the dataset map on a timer
        # update_map_callback_group = MutuallyExclusiveCallbackGroup()
        update_map_callback_group = None
        self.dataset_map_update_timer = self.create_timer(1., self._update_dataset_map_callback, callback_group=update_map_callback_group)

        # Services used to register a dataset and update the mappint in the timer
        register_callback_group = MutuallyExclusiveCallbackGroup()
        local_to_global_callback_group = MutuallyExclusiveCallbackGroup()
        self.register_client = self.create_client(RegisterDatasetMapJSON, 'register_dataset_map_json', callback_group=register_callback_group)
        self.get_local_to_global_client = self.create_client(GetLocaltoGlobalMap, 'get_local_to_global_map', callback_group=local_to_global_callback_group)
        req_services = [self.register_client, 
                        self.get_local_to_global_client, 
                        ]
        while not all([r.wait_for_service(timeout_sec=1.0) for r in req_services]):
            self.get_logger().info('Service not available, waiting...')

    def _update_dataset_map_callback(self):
        """
        Write a request to GetLocaltoGlobalMap, for our specific dataset.
        If the dataset for this model has not ben defiend, do nothing
        """
        # self.get_logger().info(f"Timer Callback")

        if not self.dataset_name:
            return

        # Update, fix our mappings
        request = GetLocaltoGlobalMap.Request()
        request.dataset_name = self.dataset_name

        response = self.get_local_to_global_client.call(request)
        # !To make async work, need to add callback function and handle response their!
        # future = self.get_local_to_global_client.call_async(request)
        # rclpy.spin_until_future_complete(self, future, timeout_sec=0.75)
        # if future.done():
        #     try:
        #         response = future.result()
        #     except Exception as e:
        #         self.get_logger().error(f'GetLocaltoGlobalMap Service call failed: {e}')
        #         return
        # else:
        #     self.get_logger().error(f'GetLocaltoGlobalMap Service call timed out.')
        #     return

        # self.get_logger().info(f"TimerLoop: Got Request -- {response.database_version}")

        if response.database_version == self.dataset_map_version:
            # Dataset is already up to data
            # self.get_logger().info(f'Dataset up to date.')
            return

        if response.database_version < 0:
            self.get_logger().error(f"Failed to get dataset mapping from:    {dataset_name}")
            return

        # One method is to create a lookup dict
            # self.dataset_map = {loc: glob for loc, glob in zip(response.dataset_ids, response.unique_ids)}
        # Other method is faster and more efficient, especially for segementation maps
            # This could be up to around 131kB, in the case the dataset uses max uint16 to represent its values
        self.dataset_map = np.zeros(max(response.dataset_ids)+1, dtype=np.uint16)
        self.dataset_map[response.dataset_ids] = response.unique_ids
        self.dataset_map_version = response.database_version

        self.get_logger().info(f"Dataset Map updated to version {self.dataset_map_version}")
        # self.get_logger().info(f'Dataset Mapping:   {self.dataset_map}')

    def map_mask_labels(self, seg_mask : np.array):
        """
        Modify the mask labels in place.

        :param seg_mask:  segmentation mask to map the labels.  Will be cast to uint16.
        """
        seg_mask = seg_mask.astype(np.uint16)

        if self.dataset_map is None:
            return seg_mask


        max_val = seg_mask.max()
        # self.get_logger().info(f'Max Value:   {max_val}')
        if max_val >= len(self.dataset_map):
            self.get_logger().warn(f'Invalid dataset map.  Max index:  {max_val},  dataset len:  {len(self.dataset_map)}')
            return seg_mask

        seg_mask = self.dataset_map[seg_mask]
        return seg_mask

    def map_bbox_labels(self, detections : Detection2DArray):
        """
        Modify the mask labels in place.

        :param bbox:  bbox to map labels from
        """
        if self.dataset_map is None:
            return detections

        for detection_id, _ in enumerate(detections.detections):
            for result_id, _ in enumerate(detections.detections[detection_id].results):
                before = np.uint16(detections.detections[detection_id].results[result_id].hypothesis.class_id)
                if before >= len(self.dataset_map):
                    self.get_logger().warn(f'Invalid dataset map.  Max index:  {max_val},  dataset len:  {len(self.dataset_map)}')

                detections.detections[detection_id].results[result_id].hypothesis.class_id = str(self.dataset_map[before])
                after = detections.detections[detection_id].results[result_id].hypothesis.class_id
        return detections


    def spawn_metadata(self, dataset_name, dataset_file):
        """
        :param dataset_name:  Name to associate dataset metedata with
        :param dataset_file:  .json filename located in id2label package.
        """
        # self.get_logger().info("Adding Dataset")
        request = RegisterDatasetMapJSON.Request()
        request.json_id2label_filename = dataset_file
        request.dataset_name = dataset_name

        # _ = self.register_client.call(request)
        future = self.register_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        # Get new datatset metadata by manually calling timer update function
        self.dataset_name = dataset_name
        # self._update_dataset_map_callback()
        # self.get_logger().info("Done Adding Dataset")
        return 

    def spawn_model_metadata(self, model_name, id2label):
        """
        Similar to spawn metadata, but use the config from the model

        :param model_name:  Semantic name of model, will be used to constuct config file on remote
        :param model:  Hugging Face Model, containing config with id2label
        """
        dataset_name = model_name.split("/")[-1]
        remote_filename = model_name + ".json"
        self.create_id2label_json(remote_filename, id2label)
        self.spawn_metadata(dataset_name, remote_filename)

    def create_id2label_json(self, file_name, id2label):
        """
        Default directory is the id2label server.
        """
        # Dump the dictionary to a JSON file with proper formatting
        file_path = get_package_share_directory('id2label_mapper') + '/' + file_name
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump({"id2label": id2label}, f, indent=2)
        self.get_logger().info(f'Wrote to file path:  {file_path}')


    #
    # done id2label services
    #

    def create_bb_publisher(self, topic=None):
        """
        Create a publisher for bounding box arrays.
        """
        if not topic:
            self.declare_parameter('bbox_topic', 'hf_model/detections')        
            topic = self.get_parameter('bbox_topic').get_parameter_value().string_value
        self.bbox_publisher = self.create_publisher(Detection2DArray, topic, 10)

    def create_seg_map_publisher(self, topic=None):
        """
        Create a publisher for segmentation maps.
        """
        if not topic:
            self.declare_parameter('seg_map_topic', 'hf_model/seg_map')        
            topic = self.get_parameter('seg_map_topic').get_parameter_value().string_value
        self.seg_publisher = self.create_publisher(Image, topic, 10)
        self.seg_map_bridge = CvBridge()

    def create_image_callback(self):
        """
        Create a subscriber for images.
        """
        # image_callback_group = MutuallyExclusiveCallbackGroup()
        # image_callback_group = ReentrantCallbackGroup()
        image_callback_group = None

        self.declare_parameter('image_topic', 'camera/image')
        topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.subscription = self.create_subscription(
            Image,
            topic,
            self.image_callback,
            qos_profile_sensor_data,
            callback_group=image_callback_group)
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
            cv_image = self.im_callback_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
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

    def create_seg_mask_msg(self, header, seg_mask : np.array, encoding="mono16") -> Image:
        """
        Transform seg results as a Image message.
        
        Parameters:
            header (Header): The ROS2 message header.
            seg_mask (np.array): Integer result of the image segmentation.
            encoding (str):  Encoding in ["mono8", "mono16"]

        :return: sensor_msgs.Image
        """
        assert encoding in ["mono8", "mono16"]

        if encoding == "mono8":
            seg_mask = seg_mask.astype(np.uint8, copy=False)
        else:
            seg_mask = seg_mask.astype(np.uint16, copy=False)

        seg_msg = self.seg_map_bridge.cv2_to_imgmsg(seg_mask, encoding=encoding)
        seg_msg.header = header
        return seg_msg



def test_label_conv(node):
    from cv_bridge import CvBridge, CvBridgeError
    import numpy as np

    # node.spawn_metadata(dataset_name="COCO2017", dataset_file='coco2017_id2label.json')
    node.spawn_metadata(dataset_name="ADE20K", dataset_file='ade20k_id2label.json')

    # ADE20K num classes are 150
    # COCO2017 are 91
    seg_map = np.arange(150, dtype=np.uint16).reshape((10, 15))

    # Convert to msg
    # bridge = CvBridge()
    # seg_map_msg = bridge.cv2_to_imgmsg(seg_map, encoding='mono16') 

    # print(request.seg_map)
    # response = MapImage.Response()
    # response = id2label_mapper.map_image_srv_callback(request, response)
    # conv_seg_map = bridge.imgmsg_to_cv2(response.seg_map, desired_encoding='mono16')


    print("Seg Map:  ",  seg_map)
    seg_map = node.map_mask_labels(seg_map)
    # print("Map:  ",  node.dataset_map)
    print("Labels Converted:  ",  seg_map)


def main(args=None):
    rclpy.init(args=args)
    model_node = ModelNode()
    # model_node.create_image_callback()
    test_label_conv(model_node)
    
    try:
        rclpy.spin(model_node)
    except KeyboardInterrupt:
        pass

    model_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

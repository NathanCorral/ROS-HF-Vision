# Ros
import rclpy
from rcl_interfaces.msg import SetParametersResult, ParameterEvent
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from cv_bridge import CvBridge, CvBridgeError
from rclpy.executors import MultiThreadedExecutor

# Hugging Face/torch
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import AutoConfig, AutoModel
from transformers.models.detr.modeling_detr import DetrObjectDetectionOutput
# Other
import cv2
# See Base Class for pub/sub functions
from models.ModelNode import ModelNode

class DETR(ModelNode):
    """
    A ROS2 node for object detection using the DETR model from Hugging Face.
    """
    def __init__(self, node_name='detr'):
        """
        Initialize the Detr node.
        
        Parameters:
            node_name (str): Name of the ROS2 node. Default is 'detr'.
        """
        super().__init__(node_name=node_name)

        # https://huggingface.co/transformers/v3.0.2/model_doc/auto.html
        self.declare_parameter('pretrained_model_name_or_path', 'facebook/detr-resnet-50')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('threshold', 0.7)

        self.threshold = self.get_parameter('threshold').get_parameter_value().double_value

        pretrained_model_name_or_path = self.get_parameter('pretrained_model_name_or_path').get_parameter_value().string_value
        self.device = self.get_parameter('device').get_parameter_value().string_value

        self.processor = DetrImageProcessor.from_pretrained(pretrained_model_name_or_path)
        # self.model = AutoModel.from_pretrained(pretrained_model_name_or_path) # Doesnt work, different output format!
        self.model = DetrForObjectDetection.from_pretrained(pretrained_model_name_or_path).to(self.device)

        # From base class, create image callback and bbox publisher
        self.create_image_callback()
        self.create_bb_publisher()
        self.spawn_metadata(dataset_name="COCO2017", dataset_file='coco2017_id2label.json')
        # self.spawn_metadata(dataset_name="ADE20K", dataset_file='ade20k_id2label.json')


    @torch.no_grad()
    def run_torch(self, image):
        """
        Run the DETR model on an input image.
        
        Parameters:
            image (numpy.ndarray): Input image in BGR format.
            
        Returns:
            tuple: A tuple containing the inputs, outputs, and results from the model.
        """
        height, width, _ = image.shape

        # 1. Preprocess inputs
        inputs = self.processor(images=image, return_tensors="pt", do_rescale=True)
        inputs = inputs.to(self.device)

        # 2. Pass to model
        outputs = self.model(**inputs)

        # 3. Postprocess output
        out = DetrObjectDetectionOutput(
            logits=outputs[0],
            pred_boxes=outputs[1],
        )
        target_sizes = torch.tensor([height, width])
        results = self.processor.post_process_object_detection(out, target_sizes=[target_sizes], threshold=self.threshold)[0]
        # Move the results to the cpu
        for k, v in results.items():
            results[k] = v.detach().cpu().numpy()
        return inputs, outputs, results

    def image_callback(self, msg):
        """
        Callback function for image messages. Overwrites the base class implementation.
        
        Parameters:
            msg (Image): The ROS2 Image message.
        """
        self.get_logger().debug(f'Image Recieved')
        try:
            cv_image = self.im_callback_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return

        # Compute bboxs results
        _, _, results = self.run_torch(cv_image)
        # results = {}
        # results["scores"], results["labels"], results["boxes"] = [], [], []
        bbox_msg = self.create_detections_msg(msg.header, results)
        bbox_msg = self.map_bbox_labels(bbox_msg)
        self.bbox_publisher.publish(bbox_msg)
        self.get_logger().debug(f'Detections Published')

def main(args=None):
    rclpy.init(args=args)
    detr = DETR()
    executor = MultiThreadedExecutor()
    executor.add_node(detr)

    try:
        # rclpy.spin(detr)
        executor.spin()
    except KeyboardInterrupt:
        pass

    detr.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
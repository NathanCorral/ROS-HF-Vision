# Ros
import rclpy
from rcl_interfaces.msg import SetParametersResult, ParameterEvent
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
# Hugging Face/torch
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import AutoConfig, AutoModel
from transformers.models.detr.modeling_detr import DetrObjectDetectionOutput
# Other
import cv2
# this repo
from detr.ObjectDetectionNode import ObjectDetectionNode

class Detr(ObjectDetectionNode):
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

        """
        # Feature avaiable after ROS2 Jazzy 
        #    https://github.com/ros2/rclpy/issues/1105


        from rclpy.parameter_event_handler import ParameterEventHandler

        # Create a handler to dynamically adjust the threshold
        self.threshold = self.get_parameter('threshold').get_parameter_value().float_value
        self.handler = ParameterEventHandler(self)
        self.callback_handle = self.handler.add_parameter_callback(
            parameter_name="threshold",
            node_name=node_name,
            callback=self.parameter_callback,
        )

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'threshold':
                self.threshold = param.value
                self.get_logger().info(f'Parameter {param.name} changed to {self.threshold}')
                
        return SetParametersResult(successful=True)
        """

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
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return

        # Compute bboxs results
        _, _, results = self.run_torch(cv_image)
        # results = {}
        # results["scores"], results["labels"], results["boxes"] = [], [], []
        self.publish_detections(msg, results)

    def publish_detections(self, image_msg, results):
        """
        Publish detection results as a Detection2DArray message.
        
        Parameters:
            image_msg (Image): The ROS2 Image message.
            results (dict): Dictionary containing detection results.
        """
        detection_array_msg = Detection2DArray()
        detection_array_msg.header = image_msg.header

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

        self.bbox_publisher.publish(detection_array_msg)

def main(args=None):
    rclpy.init(args=args)
    detr = Detr()

    try:
        rclpy.spin(detr)
    except KeyboardInterrupt:
        pass

    detr.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
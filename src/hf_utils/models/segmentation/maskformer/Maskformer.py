# Ros
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.executors import MultiThreadedExecutor

# Hugging Face/torch
import torch
from transformers import (
    MaskFormerImageProcessor,
    AutoImageProcessor,
    MaskFormerForInstanceSegmentation,
)

# Other
import os
import numpy as np
from datetime import datetime


# See Base Class for pub/sub functions
from models.ModelNode import ModelNode

class MaskFormer(ModelNode):
    """
    A ROS2 node for image segmentation using the Facebook Maskformer model hosted by Hugging Face.
    """
    def __init__(self, node_name='maskformer'):
        """
        Initialize the MaskFormerNode node.
        
        Parameters:
            node_name (str): Name of the ROS2 node. Default is 'maskformer'.
        """
        super().__init__(node_name=node_name)

        self.declare_parameter('pretrained_model_name_or_path', 'facebook/maskformer-swin-base-coco')
        self.declare_parameter('device', 'cpu')
        
        self.device = self.get_parameter('device').get_parameter_value().string_value

        pretrained_model_name_or_path = self.get_parameter('pretrained_model_name_or_path').get_parameter_value().string_value
        self.get_logger().info(f"Loading model:  {pretrained_model_name_or_path}")
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name_or_path)
        self.model = MaskFormerForInstanceSegmentation.from_pretrained(pretrained_model_name_or_path).to(self.device)
        
        self.create_image_callback()
        self.create_seg_map_publisher()

        """
        Before
        dataset_name = pretrained_model_name_or_path.split("/")[-1]
        remote_filename = pretrained_model_name_or_path + ".json"

        # Will create an id2label json map in "install/id2label_mapper/share/id2label_mapper/$remote_filename"
        # self.get_logger().info(f'Model Config:  {self.model.config}')
        self.create_id2label_json(remote_filename, self.model)


        self.spawn_metadata(dataset_name=dataset_name, dataset_file=remote_filename)
        # self.spawn_metadata(dataset_name="ADE20K", dataset_file='ade20k_id2label.json')
        # self.spawn_metadata(dataset_name="COCO2017", dataset_file='coco2017_id2label.json')
        """
        # Combined:
        self.spawn_model_metadata(pretrained_model_name_or_path, self.model)

    @torch.no_grad()
    def run_torch(self, image):
        """
        Run the Maskformer model on an input image.
        
        Parameters:
            image (numpy.ndarray): Input image in BGR format.
            
        Returns:
            tuple: A tuple containing the inputs, outputs, and results from the model.
        """
        _, height, width  = image.shape
        # height, width = image.size

        # 1. Preprocess inputs
        inputs = self.processor(images=image, return_tensors="pt", do_rescale=True)
        inputs = inputs.to(self.device)

        # 2. Pass to model
        outputs = self.model(**inputs)
        # self._print_shapes(outputs)
        # 3. Postprocess output

        # results = self.processor.post_process_semantic_segmentation(
        #     outputs, target_sizes=[[height, width]]
        # )
        # By setting target_sizes to None, leave in compressed form for transport
        results = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=None
        )

        # Move the results to the cpu
        for i in range(len(results)):
            results[i] = results[i].detach().cpu().numpy()

        return inputs, outputs, results

    def _print_shapes(self, outputs):
        self.get_logger().info(f'Output type:  {type(outputs)}')
        # self.get_logger().info(f'Output type:  {type(outputs)}')

        # https://huggingface.co/docs/transformers/main/en/model_doc/maskformer :: MaskFormerForInstanceSegmentationOutput
        self.get_logger().info(f'class_queries_logits:  {outputs.class_queries_logits.shape}')
        self.get_logger().info(f'masks_queries_logits:  {outputs.masks_queries_logits.shape}')


    def image_callback(self, msg):
        """
        Callback function for image messages
        
        Parameters:
            msg (Image): The ROS2 Image message.
        """
        # self.get_logger().info(f"Im Callback")
        try:
            cv_image = self.im_callback_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # cv_image = self.im_callback_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except CvBridgeError as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return

        cv_image = cv_image.transpose((2, 0, 1))
        _, _, results = self.run_torch(cv_image)

        # Transform results

        # self.get_logger().info(f'Unique Seg Class list:   {np.unique(results[0])}')
        # self.get_logger().info(f'Seg Map shape:   {results[0].shape}')
        seg_mask = self.map_mask_labels(results[0])
        # self.get_logger().info(f'seg_mask Class list:   {np.unique(seg_mask)}')

        seg_msg = self.create_seg_mask_msg(msg.header, seg_mask)
        self.seg_publisher.publish(seg_msg)
        # self.get_logger().info(f"Pubbed")



def main(args=None):
    rclpy.init(args=args)
    maskformer = MaskFormer()
    executor = MultiThreadedExecutor()
    executor.add_node(maskformer)


    try:
        # rclpy.spin(maskformer)
        executor.spin()
    except KeyboardInterrupt:
        pass

    maskformer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
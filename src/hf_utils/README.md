# HF Utils



This package implements the bridge between some [Hugging Face hosted models/datasets](https://huggingface.co/models) and ROS2 Messages.  

The main object which all nodes inherit from is in the *models/ModelNode.py* file.  This implements common ROS publisher/subscribers used in HF models that the child class can dynamically instantiate.  ModelNode functions include:

- `spawn_model_metadata(..)`
  - Along with a HF model id (e.g. model.config.id2label), this will propagate the id2label service with the model's pretrained dataset mapping.
  - Currently, the id2label service needs to be running on the same machine for this function to work!
- `_update_dataset_map_callback(..)`
  - This is a timer callback (1 second) which checks to make sure the id2label local-to-global map is up to date. 
  - The timer callback is created in the `init_id2label_srvs(..)` function and is done automatically on ModelNode initialization.
- `map_mask_labels(..)`, `map_bbox_labels(..)`
  - These will transform the class_id's of the messages (or arrays) from their local to the global mapping.
- `create_image_callback(..)`
  - This will create a callback to the "image_topic" parameter.
  - The callback function `image_callback(..)` should be implemented by the child class.
- `create_seg_map_publisher(..)`, `create_bb_publisher(..)`
  - This will create the *self.seg_publisher* and *self.bbox_publisher* objects, respectively. 
  - See the specific functions for details.
- `create_detections_msg(..)`
  - This will create a [Detection2DArray](https://github.com/ros-perception/vision_msgs/tree/ros2/vision_msgs/msg) from a HF "results" dict containing the keys:  ["scores", "labels", "boxes"].



For examples on how these ROS functions are used in bounding box predictions and segmentation mask predictions, see *models/obj_det/detr/DETR.py* or *models/segmentation/maskformer/Maskformer.py*

For an example on downloading and playing a dataset see *models/BBoxImageDatasetPublisher.py*



For usage and install examples, see the main README.md of this repository.

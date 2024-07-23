# Ros
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge, CvBridgeError
# Hugging Face/torch
import torch
from transformers import (
    MaskFormerImageProcessor,
    AutoImageProcessor,
    MaskFormerForInstanceSegmentation,
)
# Other
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np



# Data manager class
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import bisect

@dataclass
class ImageDataEntry:
    timestamp: datetime
    image: Optional[Any] = None
    mask: Optional[Any] = None
    bbox: Optional[Any] = None

class ImageDataManager:
    """
    Images should be recieved sequentially.  So that a bbox/mask is not left outstanding without an image.
    """
    def __init__(self, skew: timedelta = timedelta(seconds=1)):
        self.data: List[ImageDataEntry] = []
        self.skew = skew

    def add_image(self, image: Any, timestamp: Optional[datetime] = None):
        timestamp = timestamp or datetime.now()
        self._add_entry(timestamp, image=image)

    def add_mask(self, mask: Any, timestamp: Optional[datetime] = None):
        timestamp = timestamp or datetime.now()
        self._add_entry(timestamp, mask=mask)

    def add_bbox(self, bbox: Any, timestamp: Optional[datetime] = None):
        timestamp = timestamp or datetime.now()
        self._add_entry(timestamp, bbox=bbox)

    def _add_entry(self, timestamp, image=None, mask=None, bbox=None):
        # First check if we have a new image to add and are not missing an image on the most recent entry
        if len(self) != 0 and image is not None and self.get_latest()["image"] is not None:
            # Set the next image
            self._insert_new_entry(timestamp=timestamp, image=image, mask=mask, bbox=bbox)
            return
        
        index = self._find_nearest_index(timestamp)
        if index is not None and abs(self.data[index].timestamp - timestamp) <= self.skew:
            existing_entry = self.data[index]
            if image is not None:
                existing_entry.image = image
            if mask is not None:
                existing_entry.mask = mask
            if bbox is not None:
                existing_entry.bbox = bbox
        else:
            self._insert_new_entry(timestamp=timestamp, image=image, mask=mask, bbox=bbox)
            if mask is not None or bbox is not None:
                print(f"Warning: Added mask or bbox with no nearby image within skew of {self.skew}")

    def _insert_new_entry(self, image=None, mask=None, bbox=None, timestamp=None):
        new_entry = ImageDataEntry(timestamp=timestamp, image=image, mask=mask, bbox=bbox)
        bisect.insort(self.data, new_entry, key=lambda entry: entry.timestamp)

    def _find_nearest_index(self, timestamp):
        if not self.data:
            return None
        # timestamps = [entry.timestamp for entry in self.data]
        # index = bisect.bisect_left(timestamps, timestamp)
        index = bisect.bisect_left(self.data, timestamp, key=lambda entry: entry.timestamp)
        if index == len(self):
            index -= 1
        return index

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if not self.data:
            raise IndexError("No data available.")
        entry = self.data[index]
        return {
            'timestamp': entry.timestamp,
            'image': entry.image,
            'mask': entry.mask,
            'bbox': entry.bbox
        }

    def __len__(self):
        return len(self.data)

    def get_latest(self) -> Dict[str, Any]:
        return self[-1]

    def get_latest_image(self) -> Dict[str, Any]:
        """
        :returns: None on get_latest_image.  May be the cause of a thrown error.
        """
        idx = -1
        ret = self[idx]
        while (abs(idx) < len(self) and ret["image"] is None):
            idx -= 1
            ret = self[idx]

        if ret["image"] is None:
            return None

        return ret

    def get_latest_full(self) -> Dict[str, Any]:
        """
        Todo return latest image, bbox, and mask
        """
        pass

# Example usage
# if __name__ == "__main__":
#     data_manager = DataManager(skew=timedelta(seconds=1))
#     data_manager.add_image("image_1", datetime.now())
#     data_manager.add_mask("mask_1", datetime.now() + timedelta(seconds=0.5))
#     data_manager.add_bbox("bbox_1", datetime.now() + timedelta(seconds=2))  # This should print a warning
#     print(data_manager.get_latest())

class MatPlotLibVisualizer:
    def __init__(self, live_display=True, num_cmap_colors=14):
        """
        :param udpate_n_frames:  Update display every n frames.  Set to zero to disable live display (Can still be used to save gif).
        :param live_display:  When set to false used for creating and saving gifs. Does not launch a gui/call plt.show().
        :param num_cmap_colors:  Number of cmap colors for the segmentation mask.  Changing the parameter will resample the colors.
        """
        self.live = live_display
        self.fig = None

        self.reset()
        self.init_cmap(num_cmap_colors)

    def reset(self):
        # Keep list of what we have seen/classified
        self.data_manager = ImageDataManager()

        # Approximate the real object detection fps
        self.prev_time = None
        self.time_diffs = []
        self.time_diffs_sum = 0.

        # Approximate the real object detection fps
        self.prev_time = None
        self.time_diffs = []
        self.time_diffs_sum = 0.

        # Delete fig if it exist
        if self.fig is not None:
            plt.cla()
            plt.close(self.fig)

        # Create a fig/ax for live display 
        # self.fig, self.axes = plt.subplots(111) # Todo enable multiple axes
        self.fig, self.ax = plt.subplots()
        self.im = None
        if self.live:
            plt.ion()

        self.frame_count = 0

    def close(self):
        plt.cla()
        plt.close(self.fig)

    def init_cmap(self, max_num_classes=40):
        self.max_num_classes = max_num_classes
        self.seen_classes = 0
        self.cmap = plt.cm.get_cmap("hsv", self.max_num_classes)
        self.label_colors = {}

    def _clear_previous_overlays(self):
        """Clear the previous mask overlay."""
        while len(self.ax.images) > 1:
            self.ax.images[-1].remove()
            if self.ax.images[-1].colorbar is not None:
                self.ax.images[-1].colorbar.remove()
    
    def _get_label_color(self, label_id):
        """Get or generate a consistent color for a given label ID."""
        if label_id not in self.label_colors:
            self.label_colors[label_id] = self.cmap((label_id+self.seen_classes) % self.max_num_classes)[:3]
            self.seen_classes += 1

        return self.label_colors[label_id]

    def add_image(self, image, timestamp=None):
        self.data_manager.add_image(image, timestamp=timestamp)

    def add_mask(self, mask, timestamp=None):
        self.data_manager.add_mask(mask, timestamp=timestamp)

    def add_bbox(self, bbox, timestamp=None):
        self.data_manager.add_bbox(bbox, timestamp=timestamp)

    def set_image(self, image):
        pass
        """Set the current image and start displaying it."""
        self.current_image = image
        if self.im is None:
            self.im = self.ax.imshow(self.current_image)
        else:
            self.im.set_data(self.current_image)
        self.update_figure()        

    def update_figure(self):
        self.fig.canvas.draw()
        self.frame_count += 1

        if self.live:
            plt.draw()
            plt.show(block=False)
            plt.pause(0.01)

    def overlay_mask(self, mask, alpha=0.5, legend=None):
        """Overlay a mask on the current image."""
        unique_labels = np.unique(mask)
        colored_mask = np.zeros((*mask.shape, 4))
        for label_id in unique_labels:
            color = self._get_label_color(label_id)
            colored_mask[mask == label_id, :3] = color
            colored_mask[mask == label_id, 3] = alpha

        self.ax.imshow(colored_mask, interpolation="nearest", alpha=alpha)

        if legend:
            # TODO
            patches = [mpatches.Patch(color=self._get_label_color(label), label=label) for label in legend]
            self.ax.legend(handles=patches, loc='upper right')

        # plt.draw()

    def display_side_by_side(self, mask, n=4):
        """
        Display the current image and segmentation map side-by-side. 
            Seg map should have full alpha.
        """
        pass

    def update(self, timestamp=None, idx=None) -> int:
        # Check that an image has been seen.
        # TODO
        #   Else if idx is specified grab image from index
        #   Else update from timestamp if its not None
        latest = self.data_manager.get_latest_image()
        if latest is None:
            # return failure
            return -1

        # Retrieve Data
        new_image = latest["image"]
        new_bbox = latest["bbox"]
        new_mask = latest["mask"]

        # Draw image, seg map, bbox's, text, fps, ..
        # But first create image, if not image has been created
        if self.im is None:
            # Create first image
            self.im = self.ax.imshow(new_image)
        else:
            # Redraw updated image
            self.im.set_data(new_image)

        if new_bbox is not None:
            # TODO draw bbox's
            pass

        if new_mask is not None:
            self._clear_previous_overlays()
            self.overlay_mask(new_mask)



        # Update figure
        self.update_figure()
        return 0

    def update_display(self, new_image):
        pass
        """Update the displayed image."""
        if self.im is None:
            self.set_image(new_image)
        else:
            self.im.set_data(new_image)
        
        self.update_figure()

    def animate_updates(self, image_sequence, interval=500):
        """Animate the updates for a sequence of images."""
        # if self.im is None:
        #     self.set_image(image_sequence[0])
        
        self.animation_frame = 0
        def update(frame):
            self.update(image_sequence[frame])
            self.animation_frame += 1

        self.ani = animation.FuncAnimation(self.fig, update, frames=len(image_sequence), interval=interval)
        plt.show()


class SegmentationVisualizer:
    def __init__(self, udpate_n_frames=1):
        """
        :param udpate_n_frames:  Update display every n frames.  Set to zero to disable live display (Can still be used to save gif).
        """
        self.current_image = None
        self.current_mask = None
        self.fig, self.ax = plt.subplots()
        self.im = None
        self.ani = None
        self.udpate_n_frames = udpate_n_frames
        if self.udpate_n_frames > 0:
            plt.ion()  # turning interactive mode on
        self.frame_count = 0
        self.init_cmap()
        
    def init_cmap(self, max_num_classes=40):
        self.max_num_classes = max_num_classes
        self.seen_classes = 0
        self.cmap = plt.cm.get_cmap("hsv", self.max_num_classes)
        self.label_colors = {}

    def _clear_previous_overlays(self):
        """Clear the previous mask overlay."""
        while len(self.ax.images) > 1:
            self.ax.images[-1].remove()
            if self.ax.images[-1].colorbar is not None:
                self.ax.images[-1].colorbar.remove()
    
    def _get_label_color(self, label_id):
        """Get or generate a consistent color for a given label ID."""
        if label_id not in self.label_colors:
            self.label_colors[label_id] = self.cmap((label_id+self.seen_classes) % self.max_num_classes)[:3]
            self.seen_classes += 1

        return self.label_colors[label_id]

    def set_image(self, image):
        """Set the current image and start displaying it."""
        self.current_image = image
        if self.im is None:
            self.im = self.ax.imshow(self.current_image)
        else:
            self.im.set_data(self.current_image)
        self.update_figure()        

    def update_figure(self):
        if self.udpate_n_frames > 0 and self.frame_count % self.udpate_n_frames == 0:
            self.fig.canvas.draw()
            plt.show(block=False)
            plt.pause(0.01)
        self.frame_count += 1

    def overlay_mask(self, mask, alpha=0.5, legend=None):
        """Overlay a mask on the current image."""
        self.current_mask = mask
        if self.current_image is None:
            raise ValueError("Current image is not set.")

        self._clear_previous_overlays()

        unique_labels = np.unique(mask)
        colored_mask = np.zeros((*mask.shape, 4))
        for label_id in unique_labels:
            color = self._get_label_color(label_id)
            colored_mask[mask == label_id, :3] = color
            colored_mask[mask == label_id, 3] = alpha

        self.ax.imshow(colored_mask, interpolation="nearest", alpha=alpha)

        if legend:
            patches = [mpatches.Patch(color=self._get_label_color(label), label=label) for label in legend]
            self.ax.legend(handles=patches, loc='upper right')

        plt.draw()

    def display_side_by_side(self, mask):
        """Display the current image and segmentation map side-by-side."""
        if self.current_image is None or mask is None:
            raise ValueError("Image or mask is not set.")
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(self.current_image)
        axs[0].set_title("Original Image")
        axs[0].axis("off")

        axs[1].imshow(self.current_image)
        axs[1].imshow(mask, cmap="nipy_spectral", alpha=0.5)
        axs[1].set_title("Segmentation Map")
        axs[1].axis("off")

        plt.tight_layout()
        plt.show()

    def update_display(self, new_image):
        """Update the displayed image."""
        if self.im is None:
            self.set_image(new_image)
        else:
            self.im.set_data(new_image)
        
        self.update_figure()

    def animate_updates(self, image_sequence, interval=500):
        """Animate the updates for a sequence of images."""
        if self.im is None:
            self.set_image(image_sequence[0])
        
        def update(frame):
            self.update_display(image_sequence[frame])

        self.ani = animation.FuncAnimation(self.fig, update, frames=len(image_sequence), interval=interval)
        plt.show()

# Example usage:
# visualizer = SegmentationVisualizer()
# visualizer.set_image(image_array)
# visualizer.overlay_mask(segmentation_map, legend=id2label)
# visualizer.display_side_by_side(segmentation_map)
# visualizer.animate_updates(image_sequence)

class MaskFormerNode(Node):
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
        self.declare_parameter('image_topic', 'camera/image')

        pretrained_model_name_or_path = self.get_parameter('pretrained_model_name_or_path').get_parameter_value().string_value
        self.device = self.get_parameter('device').get_parameter_value().string_value

        self.get_logger().info(f"Loading model:  {pretrained_model_name_or_path}...")
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name_or_path)
        self.model = MaskFormerForInstanceSegmentation.from_pretrained(pretrained_model_name_or_path).to(self.device)

        self.create_image_callback()

        self.viz = MatPlotLibVisualizer()


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
        self.bridge = CvBridge()
    
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

        # 3. Postprocess output
        results = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[[height, width]]
        )

        # Move the results to the cpu
        for i in range(len(results)):
            results[i] = results[i].detach().cpu().numpy()

        return inputs, outputs, results

    def image_callback(self, msg):
        """
        Callback function for image messages
        
        Parameters:
            msg (Image): The ROS2 Image message.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'Could not convert image: {e}')
            return

        self.viz.add_image(cv_image)
        cv_image = cv_image.transpose((2, 0, 1))
        _, _, results = self.run_torch(cv_image)
        self.get_logger().info(f"Unique Seg Class list:   {np.unique(results[0])}")
        self.get_logger().info(f"Seg Map shape:   {results[0].shape}")
        # self.viz.overlay_mask(results[0])
        self.viz.add_mask(results[0])
        self.viz.update()


    def publish_detections(self, image_msg, results):
        """
        Publish detection results as a Detection2DArray message.
        
        Parameters:
            image_msg (Image): The ROS2 Image message.
            results (dict): Dictionary containing detection results.
        """
        pass

def test_image(args=None):
    from PIL import Image
    import requests
    rclpy.init(args=args)
    print("Downloading Image")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    pix = np.array(image.getdata()).reshape(image.size[1], image.size[0], 3).transpose((2, 0, 1))
    maskformer = MaskFormerNode() 
    print("Input Shape:  ",  pix.shape)
    _, _, results = maskformer.run_torch(pix)
    print(np.unique(results))

    try:
        rclpy.spin(maskformer)
    except KeyboardInterrupt:
        pass
    maskformer.destroy_node()
    rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    maskformer = MaskFormerNode()

    try:
        rclpy.spin(maskformer)
    except KeyboardInterrupt:
        pass

    maskformer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    # main()
    test_image()
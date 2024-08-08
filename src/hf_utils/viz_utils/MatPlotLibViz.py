import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
import scipy.ndimage
from matplotlib.patches import Patch

# viz data
from .ImageDataManager import ImageDataManager

class MatPlotLibViz:
    def __init__(self, live_display=True, num_cmap_colors=14):
        """
        :param udpate_n_frames:  Update display every n frames.  Set to zero to disable live display (Can still be used to save gif).
        :param live_display:  When set to false used for creating and saving gifs. Does not launch a gui/call plt.show().
        :param num_cmap_colors:  Number of cmap colors for the segmentation mask.  Changing the parameter will resample the colors.
        """
        self.live = live_display
        self.fig = None

        self.reset()
        self.reset_fig()
        self.init_cmap(num_cmap_colors)
        self.init_hover()


        # Track objects added from the bbox detections
        self.patches = []
        self.texts = []

    def init_hover(self):
        """
        Create functionality so the user can hover over (with mouse) a point on the window and have
         it dynmically create what label they are over.
        """
        self.xdata, self.ydata = None, None

        if not self.live:
            self.live_highlight = False
            self.hover_annot = None
            return
        self.live_highlight = True

        # Connect the hover function to the motion_notify_event
        self.fig.canvas.mpl_connect("motion_notify_event", self.hover)

        self.hover_annot = self.ax.annotate("", 
                    xy=(0,0), xytext=(20,20),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
        self.hover_annot.set_visible(False)


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
        self.frame_count = 0

    def reset_fig(self):
        """
        Reset the figure and ax objects.
        Remove the first image
        """
        self.close()
        self.fig, self.ax = plt.subplots()
        self.im = None
        if self.live:
            plt.ion()

    def hover(self, event):
        self.xdata, self.ydata = event.xdata, event.ydata
        # if event.inaxes == self.ax:
        #     pass

    def close(self):
        if self.fig is not None:
            plt.cla()
            plt.close(self.fig)
            plt.close("all")
            self.fig = None

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

    def _clear_previous_bboxs(self):
        """Clear the previous bboxs and text added."""
        for patch in self.patches:
            patch.remove()
        for text in self.texts:
            text.remove()
        self.patches = []
        self.texts = []
        
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

    def update_figure(self):
        self.fig.canvas.draw()
        self.frame_count += 1

        if self.live:
            plt.draw()
            plt.show(block=False)
            plt.pause(0.01)

    def overlay_mask(self, mask, alpha=0.6, id2label=None):
        """Overlay a mask on the current image."""
        unique_labels = np.unique(mask)
        colored_mask = np.zeros((*mask.shape, 4))
        patches = []
        for label_id in unique_labels:
            color = self._get_label_color(label_id)
            colored_mask[mask == label_id, :3] = color
            colored_mask[mask == label_id, 3] = alpha
            if id2label is not None:
                if label_id in id2label.keys():
                    patches.append(Patch(color=color, label=id2label[label_id]))
                else:
                    patches.append(Patch(color=color, label=label_id))

        self.ax.imshow(colored_mask, interpolation="nearest", alpha=alpha)
        # self.ax.imshow(colored_mask, interpolation="nearest")

        # if legend:
            # TODO
            # patches = [mpatches.Patch(color=self._get_label_color(label), label=label) for label in legend]
            # self.ax.legend(handles=patches, loc='upper right')
        if id2label is not None:
            self.ax.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.draw()

    def display_side_by_side(self, mask, n=4):
        """
        Display the current image and segmentation map side-by-side. 
            Seg map should have full alpha.
        """
        pass

    def interpolate_uint16_array(self, input_array: np.ndarray, height: int, width: int, method: str = 'nearest') -> np.ndarray:
        """
        Interpolates a np.uint16 2D array to a new size (height, width) using the specified interpolation method.

        Parameters:
        input_array (np.ndarray): The input 2D array of type np.uint16.
        height (int): The desired height of the output array.
        width (int): The desired width of the output array.
        method (str): The interpolation method to use ('nearest' by default).

        Returns:
        np.ndarray: The interpolated 2D array of type np.uint16 with the specified size.
        """
        if input_array.dtype != np.uint16:
            raise ValueError("Input array must be of type np.uint16")
        
        if method not in ['nearest', 'bilinear', 'bicubic']:
            raise ValueError("Unsupported interpolation method. Use 'nearest', 'bilinear', or 'bicubic'.")

        zoom_factors = (height / input_array.shape[0], width / input_array.shape[1])
        if method == 'nearest':
            order = 0
        elif method == 'bilinear':
            order = 1
        elif method == 'bicubic':
            order = 3
        
        interpolated_array = scipy.ndimage.zoom(input_array, zoom_factors, order=order).astype(np.uint16)
        
        return interpolated_array

    def update(self, timestamp=None, idx=None, id2label={}) -> int:
        # Check that an image has been seen.
        # TODO
        #   Else update from timestamp if its not None
        if idx is not None: 
            latest = self.data_manager.get_left(idx)
        elif timestamp is not None:
            raise NotImplementedError("todo")
        else:
            latest = self.data_manager.get_latest_image()
            if latest is None:
                return 2

            latest_bbox = self.data_manager.get_latest_bbox()
            latest["bbox"] = latest_bbox["bbox"] if latest_bbox is not None else None
            latest_mask = self.data_manager.get_latest_mask()
            latest["mask"] = latest_mask["mask"] if latest_mask is not None else None

        if latest is None:
            # return failure, no data to fetch
            return 2

        # Retrieve Data
        new_image = latest["image"]
        new_bbox = latest["bbox"]
        new_mask = latest["mask"]
        timestamp = latest["timestamp"]

        if new_image is None:
            # return failure to update
            return 1
        height, width = new_image.shape[:2]

        # Draw image, seg map, bbox's, text, fps, ..
        # But first create image, if not image has been created
        if self.im is None:
            # Create first image
            self.im = self.ax.imshow(new_image)
        else:
            # Redraw updated image
            self.im.set_data(new_image)

        if new_mask is not None:
            new_mask = self.interpolate_uint16_array(new_mask, height, width)
            self._clear_previous_overlays()
            self.overlay_mask(new_mask, id2label=id2label)

        if new_bbox is not None:
            self._clear_previous_bboxs()
            self.draw_bbox(new_bbox, id2label=id2label)

        # Update figure
        self.update_hover(new_mask, id2label)
        self.update_figure()
        return 0

    def update_hover(self, seg_mask, id2label):
        if not self.live_highlight  \
                or not self.xdata   \
                or not self.ydata   \
                or seg_mask is None \
                or id2label is None:

            self.hover_annot.set_visible(False)
            return

        _id = seg_mask[int(self.ydata), int(self.xdata)]
        if _id not in id2label.keys():
            label = _id
        else:
            label = id2label[_id]
        text = f'{label}'
        self.hover_annot.xy = (int(self.xdata), int(self.ydata))
        self.hover_annot.set_text(text)
        self.hover_annot.set_visible(True)


    def create_gif(self, filename, fps):
        temp_live = self.live
        self.live = False
        self.reset_fig()

        # Animate function 
        self.animation_frame = 0
        def animate(frame):
            self.update(idx=self.animation_frame)
            self.animation_frame += 1

        ani = animation.FuncAnimation(self.fig, animate, repeat=False, 
            frames=len(self.data_manager)-1)

        writer = animation.PillowWriter(fps)
        ani.save(filename, writer=writer)
        
        self.live = temp_live
        self.reset_fig()

    def draw_bbox(self, detections, id2label=None):
        """
        Draw bounding boxes on the frame using Matplotlib.

        Parameters:
            detections (vision_msgs.msg.Detection2DArray): The array of detections.

        Returns:
            None
        """
        for detection in detections.detections:
            bbox = detection.bbox
            x_min = bbox.center.position.x - bbox.size_x / 2
            y_min = bbox.center.position.y - bbox.size_y / 2
            width = bbox.size_x
            height = bbox.size_y

            # Draw bounding box
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='g', facecolor='none')
            self.ax.add_patch(rect)
            self.patches.append(rect)

            if detection.results:
                sorted_results = sorted(detection.results, key=lambda result: result.hypothesis.score, reverse=True)
                best_result = sorted_results[0]
                score = best_result.hypothesis.score
                label = best_result.hypothesis.class_id
                label_text = f"{id2label[int(label)] if id2label else label}: {score:.2f}"

                # Draw label
                text = self.ax.text(x_min, y_min - 10, label_text, fontsize=9, color='black', bbox=dict(facecolor='white', alpha=0.5))
                self.texts.append(text)

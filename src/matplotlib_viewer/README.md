# Matplotlib Viewer

This package uses matplotlib as a back end to view (through animation) and create gifs from ROS2 topics.

The core message it subscribes to is an Image, which it can display live as part of a timer callback.  In addition, it can annotate this image with:

- Bounding Boxes (vision_msgs.Detection2DArray)
- Segmentation Masks (sensor_msgs.Image)

Class id's are converted to human readable strings through the id2label services (as part of this repo).



## Files and Function

The main files and their description are:

- **matplotlib_viewer/ImageDataManager.py**
  - This associates image, bounding boxes, and segmentation masks with a timestamp, allowing asynchronous receipt of messages.
  - The images, bounding boxes, and segmentation masks can then be associated by looking left on the storage array.
  - Data is collected and stored forever, it should therefore only be used for visualization purposes and run at a limited time scope.
- **matplotlib_viewer/MatPlotLibViz.py**
  - This contains the ImageDataManager object to store data.
  - It then implements functionality such as:
    - Live display and updating of the matplotlib figure/axis
    - Detecting a mouse hovering over the image
    - Creating .gif's from the collected data
    - Drawing Bounding boxes, segmentation overlays, labels, and a legend for the segmentation map
  - Colors for the segmentation id's are chosen at random but the association of each id and color is saved and looked up.  
    - By default the max number of colors in the cmap is 20
- **matplotlib_viewer/VizNode.py**
  - This node implements the ROS2 callbacks and functionality
  - It inherits from an rclpy.Node and the matplotlib_viewer.MatPlotLibViz objects
  - It interfaces with the id2label service center to get the transformation of integer label id's into label strings.
  - It monitors the parameter server for the "create_gif" parameter, which will then cause it to create a gif using the parameters value as a filename.



## Building and Usage Example

For building and usage, see the main README.md of the repo.

To test this individual package:

```bash
# id2label and id2label_services are required dependencies
colcon build --packages-select id2label_mapper_services id2label_mapper matplotlib_viewer
source install/setup.bash
ros2 run matplotlib_viewer viz 

# In a new terminal:
# Launch some other nodes that publish to camera/image (rosbag, camera_publisher, hf_dataset, ...)
#   - and it should display on the matplotlib viewer
# e.g.
ros2 bag play data/rosbag2_2024_07_11-14_13_26/rosbag2_2024_07_11-14_13_26_0.db3

# In a new terminal:
# To create a gif with the published messages, run:
ros2 param set /viz create_gif "example.gif"
#	- This should freeze the viewer, until creation of the example.gif is complete
```


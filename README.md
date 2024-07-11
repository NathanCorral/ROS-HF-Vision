# ROS-HF-Vision

## Overview

This project consists of a ROS 2-based object detection and visualization system leveraging the DETR (DEtection TRansformers) model from Hugging Face. 

Components from the 'src/detr' directory include:

1. **ObjectDetectionNode** (`ObjectDetectionNode.py`) - A base ROS2 node for subscribing to image topics and publishing detected bounding boxes.
2. **Detr Node** (`Detr.py`) - Extends the base (ObjectDetectionNode.py) node to implement object detection using the DETR model, including image pre-processing, model inference, post-processing and detection result publication.
3. **DetectionVisualizer** (`DetectionVisualizer.py`) - A ROS2 node for visualizing the detection results by synchronizing image and detection messages and optionally creating GIFs of the detections.
4. **Launch File** (`detr.launch.py`) - Manages the launch configuration for various nodes including camera publisher, DETR node, and visualization node with parameters for camera index, image topic, bounding box topic, etc.

The other nodes in 'src/camera_publisher' are included to publish images from a /dev/video* device.

The project is designed to be expandable, eventually adding in other object detection models from the [Hugging Face Models Repository](https://huggingface.co/models).



## Requirements and Building

Disclaimer:  I have not yet verified these on a cleanly installed machine, and have ROS 2 (Humble) Desktop installed.

Clone the repo with:

```bash
git clone https://github.com/NathanCorral/ROS-HF-Vision.git
```

### ROS Package Dependencies:

Additional Packages: (check current packages with `ros2 pkg list`)

- `vision_msgs`:   https://github.com/ros-perception/vision_msgs/tree/ros2

- `cv_bridge`:   https://github.com/ros-perception/vision_opencv/tree/humble/cv_bridge 

- `sensor_msgs`:  https://docs.ros.org/en/ros2_packages/humble/api/sensor_msgs/

- `image_transport`:  https://github.com/ros-perception/image_transport_tutorials

  

### Virutalenv / Pip

-  Before installing, I had already sourced my setup.bash with:

  ```bash
  source /opt/ros/humble/setup.bash
  source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash
  ```

- Follow the guide for setting up a virtual environment [here](https://docs.ros.org/en/humble/How-To-Guides/Using-Python-Packages.html#installing-via-a-virtual-environment), 

  ```bash
  sudo apt install virtualenv # If not yet installed
  # Make a virtual env and activate it
  virtualenv -p python3 ./venv
  source ./venv/bin/activate
  # Make sure that colcon doesn’t try to build the venv
  touch ./venv/COLCON_IGNORE
  ```

- Activate the newly created environment and install the packages:

  ```bash
  source ./venv/bin/activate
  python3 -m pip install -r requirements.txt
  ```

  - Instead of installing all the requirements, I suggest you first install the following packages and then look through the *requirements.txt* file to select the remaining required on your setup.  Especially since versions of CUDA may not match!
    - Pytorch:  https://pytorch.org/
    - Hugging Face Transformers:  https://huggingface.co/docs/transformers/installation
      - Also install "accelerate", "timm"
    - Opencv:  https://pypi.org/project/opencv-python/
    - Numpy



### Building

Although maybe unnecessary, I tried to manually link my virtual environment during the build process with:

```bash
source ./venv/bin/activate
colcon build --cmake-args -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python" --packages-select detr camera_publisher
```

Normally, building can be done with

```bash
colcon build --packages-select detr camera_publisher
```



## Running

![Example showing full resolution obj detection](./doc/gifs/ex_1920x1080.gif)

**!Important!**    Before running, make sure ROS 2 can find the virtual environement packages by exporting the PYTHONPATH as described in [this stackoverflow post](https://robotics.stackexchange.com/questions/98214/how-to-use-python-virtual-environments-with-ros2):

```bash
# Activate the virutal environment
source ./venv/bin/activate 
# Export Python Path
export PYTHONPATH=$PYTHONPATH:$VIRTUAL_ENV/lib/python3.10/site-packages
# Source local packages
source install/setup.bash
```

Then, launch the camera publisher (c++ node by default), the DETR object detection node, and the synchronized visualization tool with:

```bash
ros2 launch detr detr.launch.py device:="cuda:0" 
```

- If using the "cpu" instead, I suggest adjusting the "slop" setting of the `ApproximateTimeSynchronizer` in the file *./src/detr/detr/DetectionVisualizer.py* from 0.1 to a higher value, since the CPU will take much more time to run the network.

- To see other arguments, use

  ```
  ros2 launch detr detr.launch.py --show-args
  ```



To save a .gif, write a parameter (the filename ending in .gif) for the DetectionVisualizerNode by:

```bash
ros2 param set /viz create_gif "output.gif"
```




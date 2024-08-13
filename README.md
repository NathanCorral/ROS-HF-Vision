# ROS-HF-Vision

## Overview

This projects seeks to integrate [Hugging Face models](https://huggingface.co/models) into ROS 2. 

The example models (DETR, Maskformer), running on my laptop with a 4GB NVIDIA GeForce GTX 1650 Ti, are shown here:

![](./doc/gifs/ex_german_roads.gif)



### Sub-packages 

The main sub-packages in the *src/* directory contain their own README.md, with a description of their files and contents.  Here is a brief overview of their function:

- *id2label_mapper/*, *id2label_mapper_services/*
  - The idea is that, since models like DETR and Maskformer are pre-trained on different datasets (COCO2017 and COCO panoptic segmentation), the class id's they use for the same objects will be different.
  - The id2label services provides a mapping from the local dataset to the collection of all objects identified by all registered models.
- *camera_publisher/*
  - This contains simple nodes for publishing Images from a  /dev/video*, allowing live inference of the models.
- *hf_utils/*
  - Main source code in this repo.
  - This contains the main model nodes and code dependent on the Hugging Face library.
- *hf_launch/*
  - Launch files used for this repo.
- *matplotlib_viewer/*
  - This subscribes to segmentation masks, images, and bounding boxes to display the results of the HF models.



## Requirements and Building



Disclaimer:  I have not yet verified these on a cleanly installed machine, and have ROS 2 (Humble) Desktop installed.

Clone the repo (without including the large .gif files):

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NathanCorral/ROS-HF-Vision.git
```

### ROS Package Dependencies:

(Mabey more) Additional Packages: (check current packages with `ros2 pkg list`)

- `vision_msgs`:   https://github.com/ros-perception/vision_msgs/tree/ros2

- `cv_bridge`:   https://github.com/ros-perception/vision_opencv/tree/humble/cv_bridge 

- `sensor_msgs`:  https://docs.ros.org/en/ros2_packages/humble/api/sensor_msgs/

- `image_transport`:  https://github.com/ros-perception/image_transport_tutorials

  

### Virutalenv / Pip

- Before installing, I had already sourced my setup.bash with:

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
    - scipy
    - matplotlib



### Building

Although maybe unnecessary, I tried to manually link my virtual environment during the build process with:

```bash
source ./venv/bin/activate
colcon build --cmake-args -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python"
```

Normally, building can be done with

```bash
colcon build --packages-select camera_publisher id2label_mapper_services id2label_mapper matplotlib_viewer hf_utils hf_launch
```



## Running

(Hint) To display the launch arguments:

```bash
source install/setup.bash
source ./venv/bin/activate
ros2 launch hf_launch default.launch.py -s # -s == --show-arguments
```



**Ros Bag Image Source:**

To launch the Hugging Face models, id2mapper services, and matplotlib visualization without any Image data publishers, use the default launch file:

```bash
ros2 launch hf_launch default.launch.py device:='cuda:0' live_viz:=True detr:=True maskformer:=False save_bag:=False
```

And then in another termial, replay the .bag file included as an example with ([requires git large files](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage), ~0.5GB):

```bash
ros2 bag play data/rosbag2_2024_07_11-14_07_48/ --remap /camera/image:=/image
```

The following example .gif was created by waiting until the bag finished playing and setting the parameter by:

```bash
ros2 param set /viz create_gif "example.gif"
```

![](./doc/gifs/example_home3.gif)

**Hugging Face Dataset Source:**

To create the above .gif from the [Hugging Face Dataset of German Traffic Signs](https://huggingface.co/datasets/keremberke/german-traffic-sign-detection) with:

```bash
ros2 launch hf_launch dataset.launch.py live_viz:=True detr:=True maskformer:=True save_bag:=False
```

- Automatically creating the .gif create_gif param doesnt work as well since it cuts out the Maskformer legend, and was created with [simplescreenrecorder](https://www.maartenbaert.be/simplescreenrecorder/).

- To view the ground truth bounding boxes contained in the dataset (optional to use Maskformer, but will not work with detr enabled):

  ```
  ros2 launch hf_launch dataset.launch.py live_viz:=True detr:=False maskformer:=True save_bag:=False gt_bbox_topic:="/hf/bbox"
  ```

  ![](./doc/images/ex_GermanTrafficSigns_Maskformer.png)

  - A feature of the Matplotlib visualizer is that with a segmentation mask you can use the mouse to hover over a section and it will display the label name.



**Live Camera Inference**

Inference can be done from a /dev/video/* device by running:

```bash
ros2 launch hf_launch camera.launch.py hz:=5 camera_index:=0
```





## Future Work

Future work for this repository could be:

- Fine-tune a model for a specific task (e.g. identifying German traffic signs).
- Integrate more models with different modalities (video, audio, language models).
- Use datasets to evaluate the throughput and accuracy of models.
- Speed up evaluation with ONNX and NVIDIA TensorRT.
- Convert the model preprocessing and current python pipeline into C++ components.
- Demonstrating deployment on end-point devices.



In addition, the id2label could be changed to subscribe to and publish a class_map[], which would allow nodes to add their local dataset labels by publishing (instead of creating a .json file) and also allow the rosbag to capture this information (so replaying the detection results have interpretable labels).













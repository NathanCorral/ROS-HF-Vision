# Id2Label Mapper -- ROS2 Package

## Overview

The Id2Label Mapper package is designed for managing and converting dataset-specific class IDs to a globally unique set of IDs across various machine learning models and datasets. This allows for consistent labeling across different models that may have been trained on different datasets. The primary functionality includes registering datasets, retrieving mappings from local dataset IDs to global IDs, in order to convert these IDs within images or other data structures.



## Services

- **register_dataset_map_json**: Registers a dataset using a JSON file containing the ID-to-label mapping.
- **get_local_to_global_map**: Retrieves the mapping from local dataset IDs to global unique IDs.
- **get_id_to_label**: Provides the global ID-to-label mapping.
- **get_dataset_id_to_label**: Provides the ID-to-label mapping for a specific dataset.



## Installation

To install this package, clone the repository into your ROS2 workspace and build the package using `colcon`.

```bash
cd ~/ros2_ws/src
git clone <repository_url>
cd ~/ros2_ws
colcon build --packages-select id2label_mapper id2label_mapper_services
```



 ## Running

```bash
. ./install/setup.bash
ros2 launch id2label_mapper test_id2label.launch.py
```




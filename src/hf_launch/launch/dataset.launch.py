from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (LaunchConfiguration, PythonExpression)
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource

from ament_index_python.packages import get_package_share_directory

import os

def generate_launch_description():
    # See default arguments
    image_topic = LaunchConfiguration('image_topic')
    gt_bbox_topic = LaunchConfiguration('gt_bbox_topic')
    gt_bbox_topic_arg = DeclareLaunchArgument(
        'gt_bbox_topic',
        default_value="/german_traffic_signs_dataset/bbox_gt",
        description='Topic to publish the ground truth values of the bounding boxes.'
    )

    default_nodes = IncludeLaunchDescription(
      PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('hf_launch')),
         '/default.launch.py']),
    )



    dataset_node = Node(
        package='hf_utils',
        executable='german_traffic_signs_dataset',
        name='german_traffic_signs_dataset',
        output='screen',
        parameters=[
            {'seg_map_topic': image_topic},
            {'bbox_topic': gt_bbox_topic},
        ],
    )

    return LaunchDescription([
        # Include other launch files
        default_nodes,
        # ml_nodes,

        # Dataset args
        gt_bbox_topic_arg,

        # Publisher Node #
        dataset_node,
    ])
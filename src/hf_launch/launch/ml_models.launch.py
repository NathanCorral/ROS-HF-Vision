from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (LaunchConfiguration, PythonExpression)
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource

import os
from ament_index_python.packages import get_package_share_directory

# TODO
#  - ml models
#  - camera options
#  - rosbag
#  - rosbag play

def generate_launch_description():
    """
    This generates some of the default parameters and nodes used by the repo
    """
    detr = LaunchConfiguration('detr')
    maskformer = LaunchConfiguration('maskformer')
    # Default topics/arguments
    device = LaunchConfiguration('device')
    image_topic = LaunchConfiguration('image_topic')
    bbox_topic = LaunchConfiguration('bbox_topic')
    seg_mask_topic = LaunchConfiguration('seg_mask_topic')

    # Enable/Disable the models
    detr_arg = DeclareLaunchArgument(
        'detr',
        default_value="True",
        description='Set to False to disable the DETR node.'
    )
    maskformer_arg = DeclareLaunchArgument(
        'maskformer',
        default_value="True",
        description='Set to False to disable the Maskformer node.'
    )

    # Additional model specific arguments (TODO, move to separate launch file and include)
    detr_thresh = LaunchConfiguration('detr_threshold')
    detr_thresh_arg = DeclareLaunchArgument(
        'detr_threshold',
        default_value="0.7",
        description='Threshhold value used by DETR to recognize objects.'
    )


    launch_maskformer_node_condition = IfCondition(PythonExpression([maskformer]))
    maskformer_node = Node(
        package='hf_utils',
        executable='maskformer',
        name='maskformer',
        output='screen',
        parameters=[
            {'device': device},
            {'image_topic': image_topic},
            {'seg_map_topic': seg_mask_topic},
        ],
        condition=launch_maskformer_node_condition,
    )
    launch_detr_node_condition = IfCondition(PythonExpression([detr]))
    detr_node = Node(
        package='hf_utils',
        executable='detr',
        name='detr',
        output='screen',
        parameters=[
            {'device': device},
            {'image_topic': image_topic},
            {'bbox_topic': bbox_topic},
        ],
        condition=launch_detr_node_condition,
    )


    # default_nodes = IncludeLaunchDescription(
    #   PythonLaunchDescriptionSource([os.path.join(
    #      get_package_share_directory('hf_launch')),
    #      '/default.launch.py']),
    # )
    return LaunchDescription([
        # default_nodes,

        # ML Arguments
        detr_arg,
        maskformer_arg,
        detr_thresh_arg,        

        # ML Models #
        maskformer_node,
        detr_node,
    ])
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
    """
    This generates some of the default parameters and nodes used by the repo
    """
    device = LaunchConfiguration('device')
    viz = LaunchConfiguration('viz')
    live_viz = LaunchConfiguration('live_viz')
    image_topic = LaunchConfiguration('image_topic')
    bbox_topic = LaunchConfiguration('bbox_topic')
    seg_mask_topic = LaunchConfiguration('seg_mask_topic')
    save_bag = LaunchConfiguration('save_bag')

    device_arg = DeclareLaunchArgument(
        'device',
        default_value="cuda:0",
        description='Pytorch device to launch the machine learning models on.'
    )
    viz_arg = DeclareLaunchArgument(
        'viz',
        default_value='False',
        description='Control whether or not to launch the vizualization node'
    )
    live_viz_arg = DeclareLaunchArgument(
        'live_viz',
        default_value='False',
        description='Control whether or not to launch the vizualization node, and will also create a live display'
    )

    #
    # Parameters that control the message topics
    #
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/image',
        description='Image topic that node subscribe to.'
    )
    bbox_topic_arg = DeclareLaunchArgument(
        'bbox_topic',
        default_value='hf/bbox',
        description='Bounding box topic which HF nodes publish to.'
    )
    seg_mask_topic_arg = DeclareLaunchArgument(
        'seg_mask_topic',
        default_value='hf/segmask',
        description='Segmentation Mask which HF nodes publish to.'
    )

    #
    # Viz Node
    #
    # Control if the vizualization node will be launched or not
    launch_viz_node_condition = IfCondition(PythonExpression([
            viz,
            ' or ',
            live_viz,
        ]))
    viz_node = Node(
        package='matplotlib_viewer',
        executable='viz',
        name='viz',
        output='screen',
        parameters=[
            {'live_display': live_viz},
            {'image_topic': image_topic},
            {'bbox_topic': bbox_topic},
            {'seg_map_topic': seg_mask_topic},
        ],
        condition=launch_viz_node_condition,
    )

    #
    # Or store all data in a bag
    #
    save_bag_arg = DeclareLaunchArgument(
        'save_bag',
        default_value='False',
        description='Set to "True" to save the published topics in a bag.'
    )
    save_bag_condition = IfCondition(PythonExpression([save_bag]))
    topics_to_record = [
        image_topic,
        bbox_topic,
        seg_mask_topic,
    ]
    save_bag_node = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '--compression-mode', 'file', '--compression-format', 'zstd'] + topics_to_record,        
        output='screen',
        condition=save_bag_condition,
    )

    #
    # Id2Label Node
    #
    id2label_node = Node(
        package='id2label_mapper',
        executable='id2label_mapper.py',
        name='id2label_mapper',
        output='screen',
        parameters=[],
    )

    ml_nodes = IncludeLaunchDescription(
      PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('hf_launch')),
         '/ml_models.launch.py']),
    )


    return LaunchDescription([
        # Args #
        device_arg,
        viz_arg,
        live_viz_arg,        
        image_topic_arg,
        bbox_topic_arg,
        seg_mask_topic_arg,

        # ML Models
        ml_nodes,

        # Utils #
        id2label_node,

        # Vizualization #
        viz_node,

        # Data logging #
        save_bag_node,
    ])
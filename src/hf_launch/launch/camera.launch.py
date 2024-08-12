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
    """
    # Default arguments
    image_topic = LaunchConfiguration('image_topic')

    # Camera Arguments
    camera_index_arg = DeclareLaunchArgument(
        'camera_index',
        default_value='0',
        description='The /dev/video* integer index of the device to use.'
    )
    hz_arg = DeclareLaunchArgument(
        'hz',
        default_value='5',
        description='Frequency of publishing camera images'
    )
    camera_index = LaunchConfiguration('camera_index')
    hz = LaunchConfiguration('hz')

    camera_node_cpp = Node(
        package='camera_publisher',
        executable='cam_pub',
        name='cam_pub',
        output='screen',
        parameters=[
            {'image_topic': image_topic},
            {'camera_index': camera_index},
            {'hz': hz}
        ]
    )


    default_nodes = IncludeLaunchDescription(
      PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('hf_launch')),
         '/default.launch.py']),
    )
    ml_nodes = IncludeLaunchDescription(
      PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('hf_launch')),
         '/ml_models.launch.py']),
    )
    return LaunchDescription([
        # Args #
        camera_index_arg,
        hz_arg,

        # Other nodes
        default_nodes,
        # ml_nodes,

        # Camera Node
        camera_node_cpp,
    ])
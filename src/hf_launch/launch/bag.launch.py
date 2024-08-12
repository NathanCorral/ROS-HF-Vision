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
    play_bag = LaunchConfiguration('play_bag')
    play_bag_arg = DeclareLaunchArgument(
        'play_bag',
        default_value="rosbag2_2024_08_12-18_33_01",
        description='Folder of rosbag node to play.'
    )

    play_bag_node = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', 
        # "--loop", 
        play_bag],
        output='screen'
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
        # Include other launch files
        default_nodes,
        # ml_nodes,

        # Dataset args
        play_bag_arg,

        # Publisher Node #
        play_bag_node,
    ])
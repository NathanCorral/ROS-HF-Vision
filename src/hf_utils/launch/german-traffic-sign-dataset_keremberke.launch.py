from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.actions import ExecuteProcess

def generate_launch_description():
  
    dataset_node = Node(
        package='hf_utils',
        executable='german_traffic_signs_dataset',
        name='german_traffic_signs_dataset',
        output='screen',
        parameters=[
        ]
    )

    viz_node = Node(
        package='hf_utils',
        executable='viz',
        name='viz',
        output='screen',
        parameters=[
            {'image_topic': "/keremberke/image"},
            {'bbox_topic': "/keremberke/gt_bbox"},
        ]
    )

    id2label_node = Node(
        package='id2label_mapper',
        executable='id2label_mapper.py',
        name='id2label_mapper',
        output='screen',
        parameters=[
        ]
    )

    return LaunchDescription([
        # Args #

        # Publisher Nodes #
        dataset_node,

        # ML Models #

        # Utils #
        id2label_node,

        # Vizualization #
        viz_node,
    ])


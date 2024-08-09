from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.actions import ExecuteProcess

def generate_launch_description():
    
    id2label_node = Node(
        package='id2label_mapper',
        executable='id2label_mapper.py',
        name='id2label_mapper',
        output='screen',
        parameters=[
        ]
    )

    id2label_test_node = Node(
        package='id2label_mapper',
        executable='test_id2label.py',
        name='test_id2label',
        output='screen',
        parameters=[
        ]
    )


    return LaunchDescription([
        id2label_node,
        id2label_test_node,
    ])

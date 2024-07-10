from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='camera_publisher',
            executable='py_pub.py',
            name='camera_publisher_py',
            output='screen',
            parameters=[
                {'image_topic': 'camera/image'},
                {'camera_index': 0},
                {'hz': 60}
            ]
        ),
        Node(
            package='detr',
            executable='main.py',
            name='detr_main',
            output='screen',
            parameters=[
                {'image_topic': 'camera/image'}
            ]
        )
    ])

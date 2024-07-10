from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    camera_node_py = Node(
            package='camera_publisher',
            executable='py_pub.py',
            name='camera_publisher_py',
            output='screen',
            parameters=[
                {'image_topic': 'camera/image'},
                {'camera_index': 0},
                {'hz': 30}
            ]
        )

    camera_node_cpp = Node(
            package='camera_publisher',
            executable='cam_pub',
            name='cam_pub',
            output='screen',
            parameters=[
                {'image_topic': 'camera/image'},
                {'camera_index': 0},
                {'hz': 30}
            ]
        )

    detr_node = Node(
            package='detr',
            executable='detr',
            name='detr',
            output='screen',
            parameters=[
                {'image_topic': 'camera/image'},
                {'device': 'cuda:0'}
            ]
        )


    viz_node = Node(
            package='detr',
            executable='viz',
            name='viz',
            output='screen',
            parameters=[
            ]
        )

    return LaunchDescription([
        camera_node_py,
        # camera_node_cpp,
        detr_node,
        viz_node,
    ])

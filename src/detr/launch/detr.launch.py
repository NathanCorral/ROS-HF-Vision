from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    camera_index_arg = DeclareLaunchArgument(
        'camera_index',
        default_value='0',
        description='The /dev/video* integer index of the device to use.'
    )

    device_arg = DeclareLaunchArgument(
        'device',
        default_value="cuda:0",
        description='Pytorch device to launch the detr on.'
    )

    hz_arg = DeclareLaunchArgument(
        'hz',
        default_value='5',
        description='Frequency of publishing camera images'
    )

    threshold_arg = DeclareLaunchArgument(
        'threshold',
        default_value='0.7',
        description='Probability threshold for determining a valid detr bbox prediciton.'
    )

    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='camera/image',
        description='Image topic to subscribe to.'
    )

    bbox_topic_arg = DeclareLaunchArgument(
        'bbox_topic',
        default_value='detr/bbox',
        description='Bounding box topic to publish to.'
    )


    camera_index = LaunchConfiguration('camera_index')
    device = LaunchConfiguration('device')
    hz = LaunchConfiguration('hz')
    threshold = LaunchConfiguration('threshold')
    image_topic = LaunchConfiguration('image_topic')
    bbox_topic = LaunchConfiguration('bbox_topic')

    camera_node_py = Node(
        package='camera_publisher',
        executable='py_pub.py',
        name='camera_publisher_py',
        output='screen',
        parameters=[
            {'image_topic': image_topic},
            {'camera_index': camera_index},
            {'hz': hz}
        ]
    )

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

    detr_node = Node(
        package='detr',
        executable='detr',
        name='detr',
        output='screen',
        parameters=[
            {'image_topic': image_topic},
            {'bbox_topic': bbox_topic},
            {'device': device},
            {'threshold': threshold}
        ]
    )

    viz_node = Node(
        package='detr',
        executable='viz',
        name='viz',
        output='screen',
        parameters=[
            {'image_topic': image_topic},
            {'bbox_topic': bbox_topic}
        ]
    )

    return LaunchDescription([
        # Args
        camera_index_arg,
        device_arg,
        hz_arg,
        threshold_arg,
        image_topic_arg,
        bbox_topic_arg,

        # Nodes
        #camera_node_py,
        camera_node_cpp,
        detr_node,
        viz_node,
    ])

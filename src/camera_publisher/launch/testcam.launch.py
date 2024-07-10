from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():

  pub_node = ComposableNode(
    package='camera_publisher',
    plugin='camera_publisher::CameraPublisher_',
    name='camera_publisher',
    parameters=[
      {"hz": 5},
      {"camera_index": 0},
    ]       
  )

  display_node = ComposableNode(
    package='camera_publisher',
    plugin='camera_publisher::CameraDisplay_',
    name='camera_display',
  )

  container = ComposableNodeContainer(
    name='my_container',
    namespace='',
    package='rclcpp_components',
    executable='component_container',
    composable_node_descriptions=[
      pub_node,
      display_node
    ],
    output='screen',
  )

  return LaunchDescription([
    container
  ])

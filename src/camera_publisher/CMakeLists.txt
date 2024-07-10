cmake_minimum_required(VERSION 3.8)
project(camera_publisher)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(image_transport REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(rclcpp_components REQUIRED)
# For adding python nodes
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Include directories
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

# Composable Nodes
add_library(cam_pub_component SHARED src/cam_pub.cpp)
ament_target_dependencies(cam_pub_component rclcpp sensor_msgs cv_bridge image_transport OpenCV rclcpp_components)
rclcpp_components_register_node(
    cam_pub_component
    PLUGIN "camera_publisher::CameraPublisher_"
    EXECUTABLE cam_pub
)
add_library(cam_display_component SHARED src/cam_display.cpp)
ament_target_dependencies(cam_display_component rclcpp sensor_msgs cv_bridge image_transport OpenCV rclcpp_components)
rclcpp_components_register_node(
    cam_display_component
    PLUGIN "camera_publisher::CameraDisplay_"
    EXECUTABLE cam_display
)

# Export Composable libraries
ament_export_targets(export_cam_pub_component)
ament_export_targets(export_display_pub_component)
install(TARGETS cam_pub_component
        EXPORT export_cam_pub_component
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
)
install(TARGETS cam_display_component
        EXPORT export_display_pub_component
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  camera_publisher/py_pub.py
  camera_publisher/py_sub.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

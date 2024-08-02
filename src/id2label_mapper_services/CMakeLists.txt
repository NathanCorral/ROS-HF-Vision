cmake_minimum_required(VERSION 3.8)
project(id2label_mapper_services)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

# msg dependencies
find_package(vision_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# For adding python nodes
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# srv generator
find_package(rosidl_default_generators REQUIRED)


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

# install(
#   DIRECTORY id2label_mapper/
#   DESTINATION lib/${PROJECT_NAME}
#   USE_SOURCE_PERMISSIONS
# )
set(srv_files
  # Mapping Server 
  "srv/GetDatasetID2Label.srv"
  "srv/GetID2Label.srv"
  "srv/GetLocaltoGlobalMap.srv"
  "srv/RegisterDatasetMapJSON.srv"

  # Example usage
  "srv/MapImage.srv"
  # "srv/MapDetection2D.srv"
  )

rosidl_generate_interfaces(${PROJECT_NAME}
  ${srv_files}
  DEPENDENCIES 
    std_msgs 
    sensor_msgs
    vision_msgs 
    builtin_interfaces
    # Add packages that above messages depend on, in this case geometry_msgs for Sphere.msg
)
ament_export_dependencies(
  rosidl_default_runtime
)

ament_package()
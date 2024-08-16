cmake_minimum_required(VERSION 3.8)
project(id2label_mapper)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(id2label_mapper_services REQUIRED)

# For adding python nodes
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# Install python files
ament_python_install_package(${PROJECT_NAME}_py)
install(
  PROGRAMS ${PROJECT_NAME}_py/id2label_mapper.py
            # ${PROJECT_NAME}_py/mapper.py
            ${PROJECT_NAME}_py/test_id2label.py
  DESTINATION lib/${PROJECT_NAME}
)

# Add shared files to project
install(
  DIRECTORY share/
  DESTINATION share/
)

# Install launch files.
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
#include "camera_publisher/cam_display.hpp"

#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <string>

#include <stdio.h>

namespace camera_publisher {

// void static ic2(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
// {
//   try {
//     cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);
//     cv::waitKey(10);
//   } catch (const cv_bridge::Exception & e) {
//     auto logger = rclcpp::get_logger("my_subscriber");
//     RCLCPP_ERROR(logger, "Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
//   }
// }

CameraDisplay_::CameraDisplay_(const rclcpp::NodeOptions & options)
: Node("camera_display", options), 
  // https://robotics.stackexchange.com/questions/102145/how-to-initialize-image-transport-using-rclcpp
  node_handle_(std::shared_ptr<CameraDisplay_>(this, [](auto *) {})),
  image_transport_(node_handle_)
{
  // Declare node parameters
  this->declare_parameter<std::string>("image_topic", "camera/image");
  this->declare_parameter<std::string>("image_transport", "compressed");

  // Create the debug window
  cv::namedWindow(CV_WINDOW_NAME);
  // Uncommenting the following line causes a segfault:
  //     https://github.com/ros-perception/image_common/issues/122
  // cv::startWindowThread();

  // Create a subscriber
  std::string topic = this->get_parameter("image_topic").as_string();
  image_transport::TransportHints hints(node_handle_.get());
  sub = image_transport_.subscribe(topic, 
    1, 
    std::bind(&CameraDisplay_::imageCallback, this, std::placeholders::_1), 
    nullptr,
    &hints);
}

CameraDisplay_::~CameraDisplay_() 
{
  cv::destroyWindow(CV_WINDOW_NAME);
}

void CameraDisplay_::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  try {
    cv::imshow(CV_WINDOW_NAME, cv_bridge::toCvShare(msg, "bgr8")->image);
    cv::waitKey(10);
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}

}  // namespace camera_publisher

#include <rclcpp_components/register_node_macro.hpp>

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(camera_publisher::CameraDisplay_)
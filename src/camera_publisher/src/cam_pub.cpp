#include "camera_publisher/cam_pub.hpp"

#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <string>

// #include <stdio.h>

namespace camera_publisher
{

CameraPublisher_::CameraPublisher_(const rclcpp::NodeOptions & options)
: Node("camera_publisher", options), 
  // https://robotics.stackexchange.com/questions/102145/how-to-initialize-image-transport-using-rclcpp
  node_handle_(std::shared_ptr<CameraPublisher_>(this, [](auto *) {})),
  image_transport_(node_handle_)
{
  // Declare node parameters
  this->declare_parameter<std::string>("image_topic", "camera/image");
  this->declare_parameter<int>("camera_index", 0);
  this->declare_parameter<int>("hz", 10);

  // Get parameters
  auto topic = this->get_parameter("image_topic").as_string();
  auto cap_index = this->get_parameter("camera_index").as_int();
  auto hz = this->get_parameter("hz").as_int();

  // Open cap
  cap_.open(cap_index); 
  if (!cap_.isOpened())
  {
      RCLCPP_ERROR(this->get_logger(), "Failed to open camera /dev/video%ld.", cap_index);
  }

  // Create publisher and timer callback
  // auto qos = rclcpp::QoS(rclcpp::KeepLast(10)); -- need int or rmw_qos_profile
  publisher_ =  image_transport_.advertise(this->get_parameter("image_topic").as_string(), 10);
  auto d = round<std::chrono::milliseconds>(std::chrono::duration<double>{1./hz});
  timer_ = this->create_wall_timer(
      d,
      std::bind(&CameraPublisher_::on_timer, this)
  );
}

void CameraPublisher_::on_timer()
{
  if (!cap_.isOpened())
  {
      return;
  }

  cv::Mat frame;
  cap_ >> frame;

  if (!frame.empty())
  {
      std_msgs::msg::Header header;
      header.stamp = this->now();
      cv_bridge::CvImage cv_image(header, "bgr8", frame);
      publisher_.publish(cv_image.toImageMsg());
  }
}

}  // namespace camera_publisher

#include <rclcpp_components/register_node_macro.hpp>

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(camera_publisher::CameraPublisher_)
#ifndef CAMERA_PUBLISHER__CAM_DISPLAY_HPP_
#define CAMERA_PUBLISHER__CAM_DISPLAY_HPP_

#include "rclcpp/rclcpp.hpp"
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>


#define CV_WINDOW_NAME "view"

namespace camera_publisher
{

class CameraDisplay_ : public rclcpp::Node
{
  /*
  Component Class for subscribing to and displaying a camera image published by the image_transoport library.
  */
public:
  explicit CameraDisplay_(const rclcpp::NodeOptions & options);

  ~CameraDisplay_();

protected:
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);

private:
  rclcpp::Node::SharedPtr node_handle_;
  image_transport::ImageTransport image_transport_;
  image_transport::Subscriber sub;
  // cv_bridge::CvImageConstPtr last_image;
};

}  // namespace camera_publisher

#endif  // CAMERA_PUBLISHER__CAM_DISPLAY_HPP_
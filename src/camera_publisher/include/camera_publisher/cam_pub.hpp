#ifndef CAMERA_PUBLISHER__CAM_PUB_HPP_
#define CAMERA_PUBLISHER__CAM_PUB_HPP_

#include "rclcpp/rclcpp.hpp"
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>

namespace camera_publisher
{

class CameraPublisher_ : public rclcpp::Node
{
public:
  explicit CameraPublisher_(const rclcpp::NodeOptions & options);

protected:
  void on_timer();

private:
  rclcpp::Node::SharedPtr node_handle_;
  image_transport::ImageTransport image_transport_;
  image_transport::Publisher publisher_;
  cv::VideoCapture cap_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace camera_publisher

#endif  // CAMERA_PUBLISHER__CAM_PUB_HPP_
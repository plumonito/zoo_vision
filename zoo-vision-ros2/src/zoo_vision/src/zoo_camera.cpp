// This file is part of zoo_vision.
//
// zoo_vision is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later
// version.
//
// zoo_vision is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with
// zoo_vision. If not, see <https://www.gnu.org/licenses/>.

#include "zoo_vision/zoo_camera.hpp"

#include <chrono>
using namespace std::chrono_literals;

#include "cv_bridge/cv_bridge.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"

namespace zoo {

ZooCamera::ZooCamera(const rclcpp::NodeOptions &options) : Node("input_camera", options) {
  publisher_ = image_transport::create_publisher(this, "input_camera/image");
  timer_ = create_wall_timer(500ms, std::bind(&ZooCamera::on_timer, this));
}

void ZooCamera::on_timer() {
  cv::Mat image = cv::imread("data/sample_frame.png", cv::IMREAD_COLOR);
  if (image.empty()) {
    image = cv::Mat3b(cv::Size(500, 500), cv::Vec3b(0, 0, 255));
  }
  std_msgs::msg::Header hdr;
  sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(hdr, "bgr8", image).toImageMsg();
  publisher_.publish(msg);
}

} // namespace zoo

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(zoo::ZooCamera)
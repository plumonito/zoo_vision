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

#include "rclcpp/time.hpp"
#include <chrono>
using namespace std::chrono_literals;

#include "cv_bridge/cv_bridge.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "zoo_vision/utils.hpp"

namespace {
const std::string DEFAULT_VIDEO_NAME = "sample_video.mp4";
}
namespace zoo {

ZooCamera::ZooCamera(const rclcpp::NodeOptions &options) : Node("input_camera", options) {
  videoUrl_ = declare_parameter<std::string>("videoUrl", getDataPath() / DEFAULT_VIDEO_NAME);

  bool ok = cvStream_.open(videoUrl_);
  if (ok) {
    frameWidth_ = cvStream_.get(cv::CAP_PROP_FRAME_WIDTH);
    frameHeight_ = cvStream_.get(cv::CAP_PROP_FRAME_HEIGHT);
    RCLCPP_INFO(get_logger(), "Opened video %s (%dx%d)", videoUrl_.c_str(), frameWidth_, frameHeight_);
  } else {
    RCLCPP_ERROR(get_logger(), "Failed to open video %s", videoUrl_.c_str());
    frameWidth_ = frameHeight_ = 500;
  }
  frameIndex_ = 0;

  rmw_qos_profile_t qos = {RMW_QOS_POLICY_HISTORY_SYSTEM_DEFAULT,
                           10,
                           RMW_QOS_POLICY_RELIABILITY_SYSTEM_DEFAULT,
                           RMW_QOS_POLICY_DURABILITY_SYSTEM_DEFAULT,
                           RMW_DURATION_UNSPECIFIED,
                           RMW_DURATION_UNSPECIFIED,
                           RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
                           RMW_DURATION_UNSPECIFIED,
                           false};
  publisher_ = image_transport::create_publisher(this, "input_camera/image", qos);
  timer_ = create_wall_timer(30ms, [this]() { this->onTimer(); });
}

void ZooCamera::onTimer() {
  std_msgs::msg::Header header;
  header.stamp = now();

  cv::Mat3b image(frameHeight_, frameWidth_);

  if (cvStream_.isOpened()) {
    // RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500, "Ptr before %ld", reinterpret_cast<intptr_t>(image.data));
    cvStream_ >> image;
    // RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500, "Ptr After  %ld", reinterpret_cast<intptr_t>(image.data));
    if (image.empty()) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500, "Video EOF, restarting");
      cvStream_.set(cv::CAP_PROP_POS_FRAMES, 0);
      frameIndex_ = 0;
      cvStream_ >> image;
    }
    header.frame_id = std::to_string(frameIndex_);

    frameIndex_++;
  }
  if (image.empty()) {
    header.frame_id = "error";
    image.setTo(cv::Vec3b(0, 0, 255));
  }

  sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
  publisher_.publish(msg);
}

} // namespace zoo

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(zoo::ZooCamera)
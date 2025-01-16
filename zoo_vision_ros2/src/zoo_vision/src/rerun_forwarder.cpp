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

#include "zoo_vision/rerun_forwarder.hpp"

#include "rclcpp/time.hpp"
#include <chrono>
using namespace std::chrono_literals;

#include "cv_bridge/cv_bridge.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include <format>
using CImage12m = zoo_msgs::msg::Image12m;
using CImage4m = zoo_msgs::msg::Image4m;
extern "C" {
extern uint32_t zoo_rs_init(void **zoo_rs_handle);
extern uint32_t zoo_rs_test_me(void *zoo_rs_handle, char const *const frame_id);
extern uint32_t zoo_rs_image_callback(void *zoo_rs_handle, char const *const channel, const CImage12m *);
extern uint32_t zoo_rs_mask_callback(void *zoo_rs_handle, char const *const channel, const CImage4m *);
}
namespace {
const std::string DEFAULT_VIDEO_URL = "data/sample_video.mp4";
}
namespace zoo {

auto timeFromRosTime(const builtin_interfaces::msg::Time &stamp) {
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::seconds(stamp.sec) +
                                                                        std::chrono::nanoseconds(stamp.nanosec));
  return std::chrono::system_clock::time_point{duration};
}

RerunForwarder::RerunForwarder(const rclcpp::NodeOptions &options)
    : Node("rerun_forwarder", options) /*, rerunStream_("zoo_vision")*/ {

  auto subscribeImage = [&](const char *const channel) {
    imageSubscribers_.push_back(rclcpp::create_subscription<zoo_msgs::msg::Image12m>(
        *this, channel, 10, [this, channel](const zoo_msgs::msg::Image12m &msg) { this->onImage(channel, msg); }));
  };
  // subscribeImage("input_camera/image");
  subscribeImage("input_camera/detections/image");
  RCLCPP_INFO(get_logger(), "Image subscriber can loan messages: %d", imageSubscribers_[0]->can_loan_messages());

  auto subscribeMask = [&](const char *const channel) {
    maskSubscribers_.push_back(rclcpp::create_subscription<zoo_msgs::msg::Image4m>(
        *this, channel, 10, [this, channel](const zoo_msgs::msg::Image4m &msg) { this->onMask(channel, msg); }));
  };
  subscribeMask("input_camera/detections/mask");

  zoo_rs_init(&rsHandle_);
}

void RerunForwarder::onImage(const char *const channel, const zoo_msgs::msg::Image12m &msg) {
  // auto frame_id = reinterpret_cast<const char *>(&msg.header.frame_id.data);
  // RCLCPP_INFO(get_logger(), "Received img (id=%s)", frame_id);
  zoo_rs_image_callback(rsHandle_, channel, &msg);
}

void RerunForwarder::onMask(const char *const channel, const zoo_msgs::msg::Image4m &msg) {
  // auto frame_id = reinterpret_cast<const char *>(&msg.header.frame_id.data);
  // RCLCPP_INFO(get_logger(), "Received img (id=%s)", frame_id);
  // zoo_rs_test_me(rsHandle_, frame_id);
  zoo_rs_mask_callback(rsHandle_, channel, &msg);
}

} // namespace zoo

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(zoo::RerunForwarder)
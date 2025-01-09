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

#include "rerun.hpp"
#include "rerun/datatypes.hpp"

#include "rclcpp/time.hpp"
#include <chrono>
using namespace std::chrono_literals;

#include "cv_bridge/cv_bridge.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include <format>
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
    : Node("rerun_forwarder", options), rerunStream_("zoo_vision") {
  // DEADEND: Serve does not work in C++
  // rerun::Error ok = rerunStream_.serve();
  auto rerunOptions = rerun::SpawnOptions();
  rerunOptions.memory_limit = "4GB";
  rerun::Error ok = rerunStream_.spawn(rerunOptions);

  if (!ok.is_ok()) {
    RCLCPP_INFO(get_logger(), "Failed to connect to rerun viewer. Error=%s", ok.description.c_str());
  }

  // imageSubscriber_ =
  //     image_transport::create_subscription(this, "input_camera/image", [this](auto msg) { this->onImage(msg); },
  //     "raw");
  imageSubscriber_ = rclcpp::create_subscription<zoo_msgs::msg::Image12m>(
      *this, "input_camera/image", 10, [this](const zoo_msgs::msg::Image12m &msg) { this->onImage(msg); });
  RCLCPP_INFO(get_logger(), "Can loan messages: %d", imageSubscriber_->can_loan_messages());
  // RCLCPP_INFO(get_logger(), "Publisher count: %ld", imageSubscriber_.getNumPublishers());
}

void RerunForwarder::onImage(const zoo_msgs::msg::Image12m &msg) {
  // auto time = timeFromRosTime(msg.header.stamp);
  // RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "Received img (time=%s)",
  //                      std::format("{:%Y-%m-%d %H:%M:%S}", time).c_str());
  // RCLCPP_INFO(get_logger(), "Received img (time=%s, id=%s)", std::format("{:%Y-%m-%d %H:%M:%S}", time).c_str(),
  //             msg.header.frame_id.c_str());
  // std::string_view frame_id(reinterpret_cast<const char *>(&msg.header.frame_id.data), msg.header.frame_id.size);
  auto frame_id = reinterpret_cast<const char *>(&msg.header.frame_id.data);
  RCLCPP_INFO(get_logger(), "Received img (id=%s)", frame_id);

  // rerunStream_.set_time("camera", time);
  // rerun::Image rerunImage(msg.data, {msg.width, msg.height}, rerun::datatypes::ColorModel::BGR,
  // rerun::datatypes::ChannelDatatype::U8);
  ;
  // rerunStream_.log("camera", rerunImage);
}

} // namespace zoo

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(zoo::RerunForwarder)
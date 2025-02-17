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

#include "zoo_vision/utils.hpp"

#include <cv_bridge/cv_bridge.hpp>
#include <nlohmann/json.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rclcpp/time.hpp>

#include <chrono>
#include <format>
#include <fstream>

using namespace std::chrono_literals;
using json = nlohmann::json;

using CImage12m = zoo_msgs::msg::Image12m;
using CImage4m = zoo_msgs::msg::Image4m;
using CDetection = zoo_msgs::msg::Detection;
extern "C" {
extern uint32_t zoo_rs_init(void **zoo_rs_handle, char const *const data_path);
extern uint32_t zoo_rs_test_me(void *zoo_rs_handle, char const *const frame_id);
extern uint32_t zoo_rs_image_callback(void *zoo_rs_handle, char const *const cameraTopic, char const *const channel,
                                      const CImage12m *);
extern uint32_t zoo_rs_detection_callback(void *zoo_rs_handle, char const *const cameraTopic, char const *const channel,
                                          const CDetection *);
}
namespace zoo {

auto timeFromRosTime(const builtin_interfaces::msg::Time &stamp) {
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::seconds(stamp.sec) +
                                                                        std::chrono::nanoseconds(stamp.nanosec));
  return std::chrono::system_clock::time_point{duration};
}

RerunForwarder::RerunForwarder(const rclcpp::NodeOptions &options)
    : Node("rerun_forwarder", options) /*, rerunStream_("zoo_vision")*/ {

  // Load config
  std::vector<std::string> cameraNames;
  {
    std::ifstream f(getDataPath() / "config.json");
    json data = json::parse(f);
    for (auto &item : data["cameras"].items()) {
      cameraNames.push_back(item.key());
    }
  }
  {
    std::string cameraNamesDescr;
    {
      for (const auto &name : cameraNames) {
        if (!cameraNamesDescr.empty()) {
          cameraNamesDescr.append(", ");
        }
        cameraNamesDescr.append(name);
      }
      cameraNamesDescr = "[" + cameraNamesDescr + "]";
    }
    RCLCPP_INFO(get_logger(), "Configured cameras=%s", cameraNamesDescr.c_str());
  }

  auto subscribeImage = [&](std::string cameraTopic, std::string channel) {
    auto subscription = rclcpp::create_subscription<zoo_msgs::msg::Image12m>(
        *this, channel, 10,
        [this, cameraTopic, channel](const zoo_msgs::msg::Image12m &msg) { this->onImage(cameraTopic, channel, msg); });
    RCLCPP_INFO(get_logger(), "Subscribed to detection image %s (loans=%d)", channel.c_str(),
                subscription->can_loan_messages());
    imageSubscribers_.push_back(std::move(subscription));
  };

  auto subscribeDetection = [&](std::string cameraTopic, std::string channel) {
    auto subscription = rclcpp::create_subscription<zoo_msgs::msg::Detection>(
        *this, channel, 10, [this, cameraTopic, channel](const zoo_msgs::msg::Detection &msg) {
          this->onDetection(cameraTopic, channel, msg);
        });
    RCLCPP_INFO(get_logger(), "Subscribed to detection results %s (loans=%d)", channel.c_str(),
                subscription->can_loan_messages());
    detectionSubscribers_.push_back(std::move(subscription));
  };

  // Subscribe to all cameras
  for (const auto &name : cameraNames) {
    subscribeImage(name, name + "/detections/image");
    subscribeDetection(name, name + "/detections");
  }

  zoo_rs_init(&rsHandle_, getDataPath().c_str());
}

void RerunForwarder::onImage(const std::string &cameraTopic, const std::string &channel,
                             const zoo_msgs::msg::Image12m &msg) {
  // auto frame_id = reinterpret_cast<const char *>(&msg.header.frame_id.data);
  // RCLCPP_INFO(get_logger(), "Received img (id=%s)", frame_id);
  zoo_rs_image_callback(rsHandle_, cameraTopic.c_str(), channel.c_str(), &msg);
}

void RerunForwarder::onDetection(const std::string &cameraTopic, const std::string &channel,
                                 const zoo_msgs::msg::Detection &msg) {
  // auto frame_id = reinterpret_cast<const char *>(&msg.header.frame_id.data);
  // RCLCPP_INFO(get_logger(), "Received img (id=%s)", frame_id);
  // zoo_rs_test_me(rsHandle_, frame_id);
  zoo_rs_detection_callback(rsHandle_, cameraTopic.c_str(), channel.c_str(), &msg);
}

} // namespace zoo

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(zoo::RerunForwarder)
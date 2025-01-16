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

#include "rclcpp/rclcpp.hpp"
#include "zoo_msgs/msg/image12m.hpp"
#include "zoo_msgs/msg/image4m.hpp"
#include <image_transport/image_transport.hpp>

namespace zoo {
class RerunForwarder : public rclcpp::Node {
public:
  explicit RerunForwarder(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

  void onImage(const char *const channel, const zoo_msgs::msg::Image12m &msg);
  void onMask(const char *const channel, const zoo_msgs::msg::Image4m &msg);

  void *rsHandle_;
  std::vector<std::shared_ptr<rclcpp::Subscription<zoo_msgs::msg::Image12m>>> imageSubscribers_;
  std::vector<std::shared_ptr<rclcpp::Subscription<zoo_msgs::msg::Image4m>>> maskSubscribers_;
};
} // namespace zoo
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

#include "zoo_msgs/msg/detection.hpp"
#include "zoo_msgs/msg/image12m.hpp"
#include "zoo_msgs/msg/image4m.hpp"

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <torch/script.h>

namespace zoo {

using float32_t = float;

class Segmenter : public rclcpp::Node {
public:
  explicit Segmenter(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

  void loadModel();
  void onImage(const zoo_msgs::msg::Image12m &msg);

private:
  Eigen::Matrix3f H_world2FromCamera_;
  Eigen::Matrix3f H_mapFromWorld2_;
  torch::jit::script::Module model_;

  std::shared_ptr<rclcpp::Subscription<zoo_msgs::msg::Image12m>> imageSubscriber_;

  std::shared_ptr<rclcpp::Publisher<zoo_msgs::msg::Image12m>> detectionImagePublisher_;
  std::shared_ptr<rclcpp::Publisher<zoo_msgs::msg::Detection>> detectionPublisher_;
};
} // namespace zoo
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
#include "zoo_vision/rerun_forwarder.hpp"
#include "zoo_vision/segmenter.hpp"
#include "zoo_vision/utils.hpp"
#include "zoo_vision/zoo_camera.hpp"

#include <memory>

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);

  // Load config once before initializing all nodes
  zoo::loadConfig();

  std::vector<std::string> cameraNames = {"zag_elp_cam_016", "zag_elp_cam_017", "zag_elp_cam_018", "zag_elp_cam_019"};

  rclcpp::executors::MultiThreadedExecutor exec{rclcpp::ExecutorOptions(), cameraNames.size() + 1};

  rclcpp::NodeOptions options;
  options.use_intra_process_comms(true);

  std::vector<std::shared_ptr<rclcpp::Node>> nodes;

  int index = 0;
  for (const auto &cameraName : cameraNames) {
    rclcpp::NodeOptions optionsCamera = options;
    optionsCamera.append_parameter_override("camera_name", cameraName);

    nodes.push_back(std::make_shared<zoo::ZooCamera>(optionsCamera, index));
    nodes.push_back(std::make_shared<zoo::Segmenter>(optionsCamera, index));
    index += 1;
  }

  nodes.push_back(std::make_shared<zoo::RerunForwarder>(options));

  for (const auto &node : nodes) {
    exec.add_node(node);
  }

  exec.spin();
  rclcpp::shutdown();
  return 0;
}
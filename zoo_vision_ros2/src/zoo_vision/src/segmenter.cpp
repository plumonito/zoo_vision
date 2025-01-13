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

#include "zoo_vision/segmenter.hpp"

#include "zoo_vision/utils.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/time.hpp>

#include <chrono>
#include <string.h>
using namespace std::chrono_literals;

namespace zoo {

Segmenter::Segmenter(const rclcpp::NodeOptions &options) : Node("segmenter", options), ortSession_{nullptr} {
  std::string modelPath = getDataPath().parent_path() / "models/camera0/segmentation/saved_model.onnx";
  ortSession_ = Ort::Session{ortEnv_, modelPath.c_str(), Ort::SessionOptions{nullptr}};

  imageSubscriber_ = rclcpp::create_subscription<zoo_msgs::msg::Image12m>(
      *this, "input_camera/image", 10, [this](const zoo_msgs::msg::Image12m &msg) { this->onImage(msg); });
}

void Segmenter::loadModel() {}

void Segmenter::onImage(const zoo_msgs::msg::Image12m &msg) {

  // DANGER: casting away const. Data may be modified by opencv if we're not careful
  auto *dataPtr = reinterpret_cast<cv::Vec3b *>(const_cast<unsigned char *>(msg.data.data()));
  const cv::Mat3b img(msg.height, msg.width, dataPtr, msg.step);

  cv::Mat3b imgFixSize;
  cv::resize(img, imgFixSize, {1024, 1024});

  // TODO: accept input as uint8
  cv::Mat3f imgFixSizef;
  imgFixSize.convertTo(imgFixSizef, CV_32FC3, 1.0f / 255);

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::array<int64_t, 4> imageTensorSizes{1, 1024, 1024, 3};
  Ort::Value imageTensor =
      Ort::Value::CreateTensor<float>(memory_info, reinterpret_cast<float *>(imgFixSizef.data), 1024 * 1024 * 3,
                                      imageTensorSizes.data(), imageTensorSizes.size());
}

} // namespace zoo

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(zoo::Segmenter)
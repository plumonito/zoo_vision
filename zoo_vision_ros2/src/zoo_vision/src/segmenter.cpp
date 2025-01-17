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

#include "zoo_vision/json_eigen.hpp"
#include "zoo_vision/utils.hpp"

#include <ATen/core/List.h>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/time.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <string.h>

using namespace std::chrono_literals;
using namespace torch::indexing;
using json = nlohmann::json;

namespace zoo {

Segmenter::Segmenter(const rclcpp::NodeOptions &options) : Node("segmenter", options) {
  const auto cameraDir = getDataPath().parent_path() / "models/camera0";

  // Load camera calibration
  {
    std::ifstream f(cameraDir / "calib.json");
    json data = json::parse(f);
    H_worldFromCamera_ = data["H_worldFromCamera"];
  }
  std::cout << "H: " << H_worldFromCamera_ << std::endl;

  // Load model
  {
    const auto modelDir = cameraDir / "segmentation";
    const std::string modelPath = modelDir / "torch.pt";

    model_ = torch::jit::load(modelPath, torch::kCUDA);
    // DEBUG print model info
  }

  // Subscribe to receive images from camera
  imageSubscriber_ = rclcpp::create_subscription<zoo_msgs::msg::Image12m>(
      *this, "input_camera/image", 10, [this](const zoo_msgs::msg::Image12m &msg) { this->onImage(msg); });

  // Publish detections
  detectionImagePublisher_ =
      rclcpp::create_publisher<zoo_msgs::msg::Image12m>(*this, "input_camera/detections/image", 10);
  detectionPublisher_ = rclcpp::create_publisher<zoo_msgs::msg::Detection>(*this, "input_camera/detections", 10);
}

void Segmenter::loadModel() {}

void Segmenter::onImage(const zoo_msgs::msg::Image12m &imageMsg) {
  // Allocate detection message so we can already start putting things here
  auto detectionMsg = std::make_unique<zoo_msgs::msg::Detection>();
  detectionMsg->header = imageMsg.header;

  const cv::Mat3b img = wrapMat3bFromMsg(imageMsg);
  const float inputAspect = static_cast<float>(imageMsg.width) / imageMsg.height;
  const int DETECTION_HEIGHT = 600;

  torch::IValue detectionResult;
  {
    auto detectionImageMsg = std::make_unique<zoo_msgs::msg::Image12m>();
    detectionImageMsg->header = imageMsg.header;
    setMsgString(detectionImageMsg->encoding, sensor_msgs::image_encodings::BGR8);
    detectionImageMsg->height = DETECTION_HEIGHT;
    detectionImageMsg->width = DETECTION_HEIGHT * inputAspect;
    detectionImageMsg->step = detectionImageMsg->width * 3 * sizeof(char);
    cv::Mat3b detectionImage = wrapMat3bFromMsg(*detectionImageMsg);

    cv::resize(img, detectionImage, detectionImage.size());

    // TODO: accept input as uint8
    cv::Mat3f detectionImagef;
    detectionImage.convertTo(detectionImagef, CV_32FC3, 1.0f / 255);

    torch::Tensor imageTensor =
        torch::from_blob(detectionImagef.data, {detectionImagef.rows, detectionImagef.cols, detectionImagef.channels()},
                         torch::TensorOptions().dtype(torch::kFloat32));

    imageTensor = imageTensor.permute({2, 0, 1}).to(torch::kCUDA);
    c10::List<torch::Tensor> imageList({imageTensor});

    // TorchScript models require a List[IValue] as input
    detectionResult = model_.forward({imageList});

    // Publish image to have it in sync with masks
    detectionImagePublisher_->publish(std::move(detectionImageMsg));
  }

  const auto detections = detectionResult.toTuple()->elements()[1].toList()[0].get().toGenericDict();

  // Results in gpu
  const torch::Tensor masksfGpu = detections.at("masks").toTensor().squeeze(1);
  const torch::Tensor boxesGpu = detections.at("boxes").toTensor();
  // const torch::Tensor scores = detections.at("scores").toTensor().to(torch::kCPU);

  // Masks to u8
  const torch::Tensor masksGpu = masksfGpu.mul(255).clamp(0, 255).to(torch::kU8);

  // Check dimensions
  assert(boxesGpu.dim() == 2);

  constexpr int64_t MAX_DETECTION_COUNT = 5;
  const int64_t detectionCount = std::min(MAX_DETECTION_COUNT, boxesGpu.sizes()[0]);

  assert(masksGpu.dim() == 3);
  assert(boxesGpu.sizes()[0] == masksGpu.sizes()[0]);
  const int64_t maskHeight = masksGpu.sizes()[1];
  const int64_t maskWidth = masksGpu.sizes()[2];
  const float32_t resizeFactor = static_cast<float32_t>(img.rows) / maskHeight;

  detectionMsg->detection_count = detectionCount;
  detectionMsg->masks.sizes[0] = detectionCount;
  detectionMsg->masks.sizes[1] = maskHeight;
  detectionMsg->masks.sizes[2] = maskWidth;
  torch::Tensor masksMap = mapRosTensor(detectionMsg->masks);

  // Move all to cpu
  const torch::Tensor boxesNet = boxesGpu.index({Slice(0, detectionCount)}).to(torch::kCPU);
  masksMap.copy_(masksGpu.index({Slice(0, detectionCount)}));

  Eigen::Map<Eigen::Matrix3Xf> worldPositionsMap{detectionMsg->world_positions.data(), 3, detectionCount};

  for (int i = 0; i < detectionCount; ++i) {
    const torch::Tensor bbox = boxesNet[i];

    // Bbox
    const Eigen::Vector2f x0{bbox[0].item<float32_t>(), bbox[1].item<float32_t>()};
    const Eigen::Vector2f x1{bbox[2].item<float32_t>(), bbox[3].item<float32_t>()};
    const Eigen::Vector2f center = (x0 + x1) / 2;
    const Eigen::Vector2f halfSize = x1 - center;
    detectionMsg->bboxes[i].center[0] = center[0];
    detectionMsg->bboxes[i].center[1] = center[1];
    detectionMsg->bboxes[i].half_size[0] = halfSize[0];
    detectionMsg->bboxes[i].half_size[1] = halfSize[1];

    // Project to world
    const Eigen::Vector2f imagePosition = (center + Eigen::Vector2f{0, halfSize[1]}) * resizeFactor;

    const Eigen::Vector3f worldPosition =
        (H_worldFromCamera_ * imagePosition.homogeneous()).hnormalized().homogeneous();
    worldPositionsMap.col(i) = worldPosition;
  }
  detectionPublisher_->publish(std::move(detectionMsg));
}

} // namespace zoo

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(zoo::Segmenter)
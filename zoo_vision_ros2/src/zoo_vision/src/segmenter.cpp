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
#include <fstream>
#include <string.h>

using namespace std::chrono_literals;

namespace zoo {

Segmenter::Segmenter(const rclcpp::NodeOptions &options) : Node("segmenter", options), ortSession_{nullptr} {
  // Load model
  const auto modelDir = getDataPath().parent_path() / "models/camera0/segmentation";
  const std::string modelPath = modelDir / "saved_model.onnx";
  Ort::SessionOptions sessionOpts;

  // auto api = Ort::GetApi();
  // OrtCUDAProviderOptionsV2 *cuda_options = nullptr;
  // Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda_options));

  // std::vector<const char *> keys{"device_id",
  //                                "gpu_mem_limit",
  //                                "arena_extend_strategy",
  //                                "cudnn_conv_algo_search",
  //                                "do_copy_in_default_stream",
  //                                "cudnn_conv_use_max_workspace",
  //                                "cudnn_conv1d_pad_to_nc1d"};
  // std::vector<const char *> values{"0", "2147483648", "kSameAsRequested", "DEFAULT", "1", "1", "1"};
  // Ort::ThrowOnError(api.UpdateCUDAProviderOptions(cuda_options, keys.data(), values.data(), keys.size()));

  // sessionOpts.AppendExecutionProvider_CUDA_V2(*cuda_options);

  ortSession_ = Ort::Session{ortEnv_, modelPath.c_str(), sessionOpts};

  // api.ReleaseCUDAProviderOptions(cuda_options);

  // DEBUG print model info
  Ort::AllocatorWithDefaultOptions allocator;
  std::cout << "Segmenter model loaded from: " << modelPath << std::endl;
  std::cout << "Inputs: ";
  for (size_t i = 0; i < ortSession_.GetInputCount(); i++) {
    std::string input_name = ortSession_.GetInputNameAllocated(i, allocator).get();
    std::cout << input_name << ", ";
  }
  std::cout << std::endl;
  std::cout << "Outputs: ";
  for (size_t i = 0; i < ortSession_.GetOutputCount(); i++) {
    std::string input_name = ortSession_.GetOutputNameAllocated(i, allocator).get();
    std::cout << input_name << ", ";
  }
  std::cout << std::endl;

  // Load anchors
  const auto anchorsPath = modelDir / "anchors.bin";
  assert(std::filesystem::exists(anchorsPath));
  const size_t anchorsByteSize = std::filesystem::file_size(anchorsPath);
  const size_t anchorCount = anchorsByteSize / 4 / sizeof(float32_t);
  std::vector<float32_t> anchors;
  anchors_.resize(anchorCount, 4);
  {
    std::ifstream f(anchorsPath, std::ios::binary);
    f.read(reinterpret_cast<char *>(anchors_.data()), anchorsByteSize);
  }

  // Subscribe to receive images from camera
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

  // Input tensors
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::array<int64_t, 4> imageTensorSizes{1, 1024, 1024, 3};
  Ort::Value imageTensor =
      Ort::Value::CreateTensor<float>(memory_info, reinterpret_cast<float *>(imgFixSizef.data), 1024 * 1024 * 3,
                                      imageTensorSizes.data(), imageTensorSizes.size());

  std::array<int64_t, 2> imageMetaTensorSizes{1, 14};
  std::array<float32_t, 14> imageMeta{0.0f, 1024.0f, 1024.0f, 3.0f,    1024.0f, 1024.0f, 3.0f,
                                      0.0f, 0.0f,    1024.0f, 1024.0f, 1.0f,    0.0f,    0.0f};
  Ort::Value imageMetaTensor = Ort::Value::CreateTensor<float>(
      memory_info, imageMeta.data(), imageMeta.size(), imageMetaTensorSizes.data(), imageMetaTensorSizes.size());

  std::array<int64_t, 3> anchorTensorSizes{1, anchors_.rows(), anchors_.cols()};
  std::cout << std::endl;
  Ort::Value anchorsTensor =
      Ort::Value::CreateTensor<float>(memory_info, anchors_.data(), anchors_.rows() * anchors_.cols(),
                                      anchorTensorSizes.data(), anchorTensorSizes.size());

  constexpr int INPUT_COUNT = 3;
  std::array<const char *, INPUT_COUNT> inputNames = {"input_image", "input_image_meta", "input_anchors"};
  std::array<Ort::Value, INPUT_COUNT> inputTensors = {std::move(imageTensor), std::move(imageMetaTensor),
                                                      std::move(anchorsTensor)};

  // Output tensors
  std::array<int64_t, 3> mrcnnDetectionTensorSizes{1, 10, 6};
  Eigen::Matrix<float32_t, 10, 6, Eigen::RowMajor> mrcnnDetection;
  Ort::Value mrcnnDetectionTensor =
      Ort::Value::CreateTensor<float>(memory_info, mrcnnDetection.data(), mrcnnDetection.rows() * mrcnnDetection.cols(),
                                      mrcnnDetectionTensorSizes.data(), mrcnnDetectionTensorSizes.size());
  constexpr int OUTPUT_COUNT = 1;
  std::array<const char *, OUTPUT_COUNT> outputNames = {"mrcnn_detection"};
  std::array<Ort::Value, OUTPUT_COUNT> outputTensors = {std::move(mrcnnDetectionTensor)};

  // Run
  Ort::RunOptions runOptions;
  ortSession_.Run(runOptions, inputNames.data(), inputTensors.data(), inputTensors.size(), outputNames.data(),
                  outputTensors.data(), outputTensors.size());
}

} // namespace zoo

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(zoo::Segmenter)
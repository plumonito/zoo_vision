cmake_minimum_required(VERSION 3.24)
project(zoo_vision)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(image_transport REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(zoo_msgs REQUIRED)
# find_package(zoo_vision_rs REQUIRED)

find_package( OpenCV 4.6.0 EXACT REQUIRED )

# Rerun SDK
include(FetchContent)
FetchContent_Declare(rerun_sdk URL https://github.com/rerun-io/rerun/releases/download/0.21.0/rerun_cpp_sdk.zip)
FetchContent_MakeAvailable(rerun_sdk)

####################################
# zoo_vision package
add_executable(zoo_vision 
  # Headers
  include/zoo_vision/zoo_camera.hpp
  include/zoo_vision/utils.hpp
  # Sources
  src/utils.cpp
  src/zoo_camera.cpp
  src/main.cpp)

target_include_directories(zoo_vision PUBLIC include)
target_compile_features(zoo_vision PUBLIC cxx_std_20)

ament_target_dependencies(zoo_vision
  rclcpp
  rclcpp_components
  image_transport
  cv_bridge
  OpenCV
  zoo_msgs
)
####################################

####################################
# zoo-vision2 package
add_executable(zoo_vision2
  # Headers
  include/${PROJECT_NAME}/rerun_forwarder.hpp
  # Sources
  src/rerun_forwarder.cpp
  src/main2.cpp)

target_include_directories(zoo_vision2 PUBLIC include)
target_compile_features(zoo_vision2 PUBLIC cxx_std_20)

ament_target_dependencies(zoo_vision2
  rclcpp
  rclcpp_components
  image_transport
  zoo_msgs
  cv_bridge
  OpenCV
)
# target_link_libraries(zoo_vision2 rerun_sdk)
target_link_libraries(zoo_vision2 /home/dherrera/git/zoo_vision/zoo_vision_ros2/build/zoo_vision_rs/release/libzoo_vision_rs.a)
####################################

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()


install(TARGETS
  ${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME})

  ament_package()

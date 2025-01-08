source ~/colcon_venv/venv/bin/activate    
source /opt/ros/jazzy/setup.zsh
source ~/git/ros2_rust_ws/install/setup.sh
source install/setup.sh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export RMW_FASTRTPS_USE_QOS_FROM_XML=1
export FASTRTPS_DEFAULT_PROFILES_FILE="$HOME/git/zoo_vision/zoo_vision_ros2/src/fast_dds_config.xml"

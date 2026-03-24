#!/bin/bash
set -e

# 載入 ROS 2 Humble 環境
source /opt/ros/humble/setup.bash

# 載入你自己的工作區環境 (如果還沒 colcon build，這行可能會報錯，可以先加 || true 避免腳本中斷)
source /ros2_ws/install/local_setup.bash || true

exec "$@"
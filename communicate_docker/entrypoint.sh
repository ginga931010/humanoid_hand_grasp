#!/bin/bash
set -e

# 載入 ROS 2 Humble 環境
source /opt/ros/humble/setup.bash

# 載入編譯好的 micro-ROS Agent 環境
source /uros_ws/install/local_setup.bash

exec "$@"
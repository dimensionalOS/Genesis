#!/bin/bash

set -e

cd /Genesis
pip install --no-cache-dir .
source /opt/ros/humble/setup.bash
source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash
# source /home/ros/dev_ws/install/setup.bash
# source /home/ros/ws_moveit2/install/setup.bash
cd /workspace/dimensional
python3 alfred_server.py

echo "Provided arguments: $@"

exec $@
#!/bin/bash

xhost +local:root

docker run --rm -it \
--name dim_genesis_cont \
--runtime=nvidia \
--gpus all \
--privileged \
--network host \
-p 10000:10000 \
-p 5901:5901 \
-p 6901:6901 \
-e DISPLAY=$DISPLAY \
-v /dev/dri:/dev/dri \
-v /tmp/.X11-unix/:/tmp/.X11-unix \
-v $PWD:/workspace \
genesis_test_img bash

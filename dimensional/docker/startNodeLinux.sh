#!/bin/bash

docker run -dt \
--user ros \
--name dim_cont \
--runtime=nvidia \
--gpus all \
--privileged \
--network host \
-p 10000:10000 \
-p 5901:5901 \
-p 6901:6901 \
headless_dim_img bash

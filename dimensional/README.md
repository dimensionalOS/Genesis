### Build Genesis Stream
```bash
sudo docker build -t dimos_genesis_stream -f docker/Dockerfile-ros-stream-ec2 docker
```

### Run Genesis Stream server
```bash
sudo docker run --gpus all --rm -it \
-e DISPLAY=$DISPLAY \
-v /dev/dri:/dev/dri \
-v /tmp/.X11-unix/:/tmp/.X11-unix \
-v $PWD:/workspace \
-p 5000:5000 \
dimos_genesis_stream
```

### View stream in browser
```
http://localhost:5000/

http://<public-ip>:5000/
```

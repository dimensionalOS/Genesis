### Build Genesis Stream
```bash
sudo docker build -t dimos_genesis_stream -f docker/Dockerfile-ros-stream-ec2 docker
```

### Run Genesis Stream server via ssh
```bash
sudo docker run --gpus all --rm -it \
-v $PWD:/workspace \
-p 5000:5000 \
dimos_genesis_stream
```

### Run Genesis Stream server via EC2 user_data 
```bash
sudo docker run --gpus all --rm -d \
-v $PWD:/workspace \
-p 5000:5000 \
dimos_genesis_stream

### View stream in browser
```
http://localhost:5000/

http://<public-ip>:5000/
```

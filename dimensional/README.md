### Build Genesis Stream
```bash
docker build -f docker/Dockerfile-ros-stream-ec2 -t dimos_genesis_stream .
```

### Run Genesis Stream server
```bash
docker run --gpus all -p 5000:5000 --rm -it dimos_genesis_stream
```

### View stream in browser
```
http://localhost:5000/

http://<public-ip>:5000/
```

# FROM osrf/ros:humble-desktop-full
FROM ros:humble-perception-jammy

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all

# Install required packages
RUN apt-get update \
    && apt-get install -y \
    build-essential \
    cmake \
    python3-colcon-common-extensions \
    python3-flake8 \
    python3-rosdep \
    python3-setuptools \
    python3-vcstool \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    nano \
    wget \
    curl \
    vim \
    git \
    x11-apps \
    tmux \
    ros-humble-foxglove-bridge \
    ros-humble-moveit \
    ros-humble-moveit-visual-tools \
    ros-humble-moveit-ros-visualization \
    ros-humble-moveit-servo \
    ros-humble-joint-state-publisher-gui \
    ros-humble-rosbridge-suite \
    ros-humble-xacro \
    ros-humble-robot-state-publisher \
    ros-humble-teleop-twist-keyboard \
    ros-humble-teleop-twist-joy \
    ros-humble-joy \
    ros-humble-controller-manager \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-joint-trajectory-controller \
    ros-humble-joint-state-broadcaster \
    ros-humble-vision-msgs \
    ros-humble-ackermann-msgs \
    iputils-ping \
    net-tools \
    htop \
    python3-pip \
    ros-humble-tf* \
    ros-humble-gazebo-ros-pkgs \
    dos2unix \
    python3-genmsg \
    gpg \
    pass \
    ros-humble-depthai-ros \
    zstd \
    bash-completion \
    libgl1 \
    libgl1-mesa-glx \
    libegl-dev \
    libegl1 \
    libxrender1 \
    libglib2.0-0 \
    ffmpeg \
    libgtk2.0-dev \
    pkg-config \
    libvulkan-dev \
    libgles2 \
    libglvnd0 \
    libglx0

RUN apt-get upgrade -y

# Set up tmux tpm
RUN git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm

# Install Python packages
RUN pip3 install rospkg numpy==1.24.4 jsonpickle scipy easydict matplotlib==3.9.1 opencv-python pyyaml pyquaternion pybullet \
    requests pillow open3d
RUN pip3 install torch torchvision torchaudio transformers

# Install CUDA (example for CUDA 11.6)
# RUN apt-get update && apt-get install -y \
#     cuda-toolkit-12-1 \
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
# RUN dpkg -i cuda-keyring_1.0-1_all.deb
# RUN apt-get update && apt-get install -y \
#     cuda \

# RUN apt-get update && apt-get -y install cuda-toolkit nvidia-gds \

# Set CUDA_HOME environment variable
ENV CUDA_HOME=/usr/local/cuda-12.4
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64

EXPOSE 8765
EXPOSE 10000

# # Installing SAM2
# RUN mkdir -p /deps
# RUN cd /deps && git clone https://github.com/facebookresearch/segment-anything-2.git
# RUN cd /deps/segment-anything-2 && pip install -e .
# RUN cd /deps/segment-anything-2/checkpoints && wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt || { echo "Failed to download checkpoint for SAM 2.1 tiny"; exit 1; }
# RUN cd /dev_ws
# RUN chown -R $USER_UID:$USER_GID /

RUN pip install --upgrade pip setuptools wheel packaging
# Installing Genesis
ARG PYTHON_VERSION=3.10
RUN cd / && git clone https://github.com/dimensionalOS/Genesis.git
RUN cd /Genesis && pip install . && pip install --no-cache-dir PyOpenGL==3.1.5

# Installing OMPL
RUN PYTHON_MAJOR_MINOR=$(echo ${PYTHON_VERSION} | tr -d '.') && \
    wget https://github.com/ompl/ompl/releases/download/prerelease/ompl-1.6.0-cp${PYTHON_MAJOR_MINOR}-cp${PYTHON_MAJOR_MINOR}-manylinux_2_28_x86_64.whl && \
    pip install ompl-1.6.0-cp${PYTHON_MAJOR_MINOR}-cp${PYTHON_MAJOR_MINOR}-manylinux_2_28_x86_64.whl && \
    rm ompl-1.6.0-cp${PYTHON_MAJOR_MINOR}-cp${PYTHON_MAJOR_MINOR}-manylinux_2_28_x86_64.whl

# Copy the entrypoint and bashrc scripts so we have 
# our container's environment set up correctly
COPY entrypoint.sh /entrypoint.sh
COPY bashrc /.bashrc
COPY .tmux.conf /.tmux.conf

RUN rm -rf /var/lib/apt/lists/*

# Set up entrypoint and default command
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
# ENTRYPOINT ["/dockerstartup/vnc_startup.sh"]
CMD ["bash"]
# CMD ["--wait"]
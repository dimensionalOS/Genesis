# Depth to PointCloud Conversion with LCM

This package provides a solution for converting depth images to point clouds in the Genesis simulator environment using LCM (Lightweight Communications and Marshalling).

## Overview

The system consists of two main components:

1. `alfred_lcm.py` - Modified to publish camera intrinsic parameters along with RGB and depth images
2. `depth_to_pointcloud.py` - Subscribes to depth images and camera info, converts depth to point clouds

## Features

- Fast, efficient point cloud generation optimized for 30fps depth streams
- Uses concurrent processing to avoid blocking the main thread
- Supports downsampling for faster processing on lower-end hardware
- Proper filtering of invalid depth values
- Caching of camera intrinsics and projection maps for better performance

## Usage Instructions

### Step 1: Start the Genesis simulator with LCM support

```bash
# Navigate to the directory
cd /home/yashas/Documents/dimensional/Genesis/dimensional

# Run the modified Alfred LCM script
python alfred_lcm.py
```

This will:
- Start the Genesis simulator
- Load the robot model
- Begin publishing RGB images, depth images, and camera info to LCM topics

### Step 2: Run the depth to pointcloud converter

```bash
# In a separate terminal window
cd /home/yashas/Documents/dimensional/Genesis/dimensional
python depth_to_pointcloud.py
```

This will:
- Subscribe to the depth image and camera info topics
- Convert depth images to point clouds
- Publish point clouds to the `head_cam_pointcloud` LCM topic

### LCM Topics

The system uses the following LCM topics:

| Topic Name | Message Type | Description |
|------------|--------------|-------------|
| `head_cam_rgb` | `Image` | RGB camera image |
| `head_cam_rgb_info` | `CameraInfo` | RGB camera intrinsic parameters |
| `head_cam_depth` | `Image` | Depth camera image (32-bit float in meters) |
| `head_cam_depth_info` | `CameraInfo` | Depth camera intrinsic parameters |
| `head_cam_pointcloud` | `PointCloud2` | Generated point cloud from depth |

### Visualization

You can visualize the point clouds using any LCM-compatible visualization tool. For example:

1. **Using lcm-spy**:
   ```bash
   lcm-spy
   ```

2. **Using RViz with LCM-ROS bridge**:
   If you have ROS installed, you can use the LCM-ROS bridge to visualize the point clouds in RViz.

### Performance Tuning

If you experience performance issues, you can adjust the following parameters in the `depth_to_pointcloud.py` script:

- `downsample_factor`: Increase this value to downsample the depth image for faster processing
- `max_workers`: Adjust the number of worker threads in the thread pool
- `filter_threshold`: Adjust the minimum distance filter
- `max_depth`: Adjust the maximum distance filter

## Troubleshooting

1. **No point clouds published**:
   - Ensure both scripts are running
   - Check that the camera info is being published
   - Verify that depth images are being published with the correct encoding

2. **Poor performance**:
   - Increase the downsample factor
   - Reduce the publishing rate in alfred_lcm.py
   - Check CPU usage and consider running on a more powerful machine

3. **Memory issues**:
   - Increase filtering thresholds to reduce the number of points
   - Increase downsampling to reduce the resolution of the point cloud
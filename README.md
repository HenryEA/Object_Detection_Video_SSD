# Object Detection in Video Streams (SSD-based)
This repository implements a deep learning-based object detection system that processes video streams frame-by-frame using the SSD-ResNet101 architecture from TensorFlow’s Object Detection API. It leverages pretrained models on the COCO dataset to detect and annotate a variety of object classes in real-time video frames.

## Project Overview
The objective of this system is to identify and label multiple object categories (e.g., person, car, traffic light, dog) across video frames, outputting annotated versions of the original input videos. It provides a hands-on demonstration of integrating TensorFlow's pre-trained models into an efficient detection pipeline.
Key Features

    Frame-by-frame object detection using SSD (Single Shot Detector) with ResNet101 as backbone.

    Pretrained on the COCO dataset (90 object categories).

    Annotated video generation with bounding boxes and class labels.

    Runs within a Google Colab environment and supports GPU acceleration.

    Uses TensorFlow's Object Detection API for model inference and visualization.

## Components
1. Environment and Setup
    - Compile protocol buffer files (.proto) required for model configuration and obtain the necessary dependencies from the Object Detection API.
    - Configure GPU memory growth to prevent pre-allocation in TensorFlow.

3. Model and Label Configuration
    Obtain ssd_resnet101_v1_fpn_640x640_coco17_tpu-8 model from TensorFlow Model Zoo and fetch the COCO label map and create a category index for class interpretation.
   

4. Object Detection Pipeline
    Use imageio to read frames from .mp4 video files.
    Convert each frame to tensor format, pass through the detection model, and extract results.
    Overlay bounding boxes and class names using visualization_utils.
    Recompile frames into an annotated video.

Technical Stack
    Python, TensorFlow 2.x, TensorFlow Object Detection API, imageio, NumPy, Matplotlib, SSD with ResNet101-FPN backbone, COCO 2017 (pretrained)

## Results
The annotated videos can be viewed at:
traffic_annotated.mp4 https://github.com/HenryEA/Object_Detection_Video_SSD/blob/main/traffic_annotated.mp4?raw=true
catdog_annotated.mp4 https://github.com/HenryEA/Object_Detection_Video_SSD/blob/main/catdog_annotated.mp4?raw=true

Upon visualizing the annotated videos, the following Quantitative Performances were observed:
    Annotated videos processed at ~0.20 FPS 
    Accuracy aligns with the pretrained model's performance on COCO object categories
    Detections are visualized for predictions with confidence ≥ 40%

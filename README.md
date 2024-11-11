# Real-Time-Action-Recognition-using-Action-Transformer-AcT-
A real-time human action recognition system implementing the Action Transformer (AcT) architecture for pose-based action classification using webcam input.
# Real-Time Action Recognition using Action Transformer (AcT)




A real-time human action recognition system implementing the Action Transformer (AcT) architecture for pose-based action classification using webcam input.






## Overview

This project implements a real-time human action recognition system that:
- Uses MediaPipe for pose estimation
- Processes live webcam feed for action recognition
- Implements temporal sequence modeling using Action Transformer
- Handles occlusions and low-visibility scenarios
- Provides real-time visual feedback

## Features

- Real-time pose estimation using MediaPipe
- Temporal sequence modeling for action recognition
- Adaptive confidence thresholding
- Temporal smoothing for stable predictions
- Support for both webcam and IP camera inputs

## Requirements

```bash
pip install -r requirements.txt



@article{mazzia2021action,
  title={Action Transformer: A Self-Attention Model for Short-Time Pose-Based Human Action Recognition},
  author={Mazzia, Vittorio and Angarano, Simone and Salvetti, Francesco and Angelini, Federico and Chiaberge, Marcello},
  journal={Pattern Recognition},
  pages={108487},
  year={2021},
  publisher={Elsevier}
}

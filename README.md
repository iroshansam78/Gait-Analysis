# Gait-Based Anomaly Detection from Video Using YOLOv8 Pose Estimation

A modular, real-time gait analysis system for detecting anomalous walking behavior in surveillance footage using pose estimation and biomechanical feature extraction.

This project leverages `yolov8n-pose.pt` from the Ultralytics YOLOv8 family to extract joint keypoints from video frames. It computes clinically relevant gait features and overlays them on the video to support anomaly detection, behavior profiling, and movement disorder analysis.

---

## Project Scope

- **Domain**: Gait analysis, pose estimation, surveillance, behavioral anomaly detection  
- **Approach**: Keypoint detection → Gait feature extraction → Real-time metric computation  
- **Target Users**: AI researchers, computer vision engineers, healthcare technologists, surveillance analysts

---

## Core Features

- **Pose Detection** – Based on YOLOv8 keypoint models (COCO-17)
- **Feature Extraction** – Computes 8+ gait parameters:
  - Step Length
  - Cadence
  - Double Support Time
  - Trunk Forward Lean
  - Hip Flexion
  - Knee Flexion
  - Ankle Dorsiflexion
  - Walking Speed
- **Real-Time Visualization** – Annotated frame-by-frame analysis with metric overlays
- **Modular Design** – Easy integration into larger gait recognition or surveillance pipelines

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt

Ensure Python 3.8+ and PyTorch are installed.

2. Download the YOLOv8 Pose Model
Download yolov8n-pose.pt from Ultralytics GitHub or via CLI:

yolo download model=yolov8n-pose.pt
3. Run the Application
python gait_analysis.py
When prompted, select a video file. The system will run inference and stream annotated output in real-time.

📂 File Structure

gait-anomaly-detection/
├── gait_analysis.py       # Main execution script
├── requirements.txt       # All required libraries
├── README.md              # Project documentation
├── LICENSE                # MIT License (recommended)
⚙️ Architecture Overview

Input Video
    ↓
Frame-by-Frame Pose Estimation (YOLOv8n-Pose)
    ↓
Keypoint Filtering & Smoothing
    ↓
Gait Feature Extraction
    ↓
Metric Computation & Temporal Aggregation
    ↓
Real-Time Overlay & Display
📊 Benchmark Notes

Inference optimized for CPU (≈10–15 FPS with yolov8n on standard laptops)
Frame rate capped by I/O; GPU acceleration optional but supported via PyTorch
🔬 Use Cases

Behavioral surveillance (e.g., suspicious gait or loitering detection)
Gait health screening (e.g., fall risk, Parkinsonian gait patterns)
Human identification under occlusion/low resolution
Military or industrial monitoring
📖 References

Ultralytics YOLOv8 Documentation – https://docs.ultralytics.com
Pose Estimation Models (COCO Keypoints) – https://cocodataset.org/#keypoints-eval
📜 License

This project is distributed under the MIT License. See LICENSE for details.

👤 Author

Iroshan Pathirannahalage
Computer Vision & Gait-Based Anomaly Detection
GitHub: @iroshanpathirannahalage






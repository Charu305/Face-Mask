# 😷 Real-Time Face Mask Detection — Deep Learning

> **A two-stage Deep Learning pipeline** for real-time face mask detection — combining a state-of-the-art face detector (RetinaFace) with a lightweight MobileNetV2 classifier to identify whether each detected person is wearing a face mask or not.

---

## 📌 Project Overview

During and after the COVID-19 pandemic, automated mask compliance monitoring became a real-world need in public spaces, workplaces, and transportation hubs. This project builds a **production-ready, real-time face mask detection system** that:

1. **Detects faces** in images and live video using RetinaFace — a high-accuracy, single-stage face detection model
2. **Classifies each face** as `With Mask` ✅ or `Without Mask` ❌ using a fine-tuned MobileNetV2 classifier
3. **Runs in real time** via webcam feed with bounding box overlays

This is a computer vision project built with **PyTorch** (detection) and **TensorFlow/Keras** (classification), evaluated against industry-standard benchmarks.

---

## 🎯 Problem Statement

> *Given an image or live video frame, detect all faces present and determine whether each person is wearing a face mask — in real time.*

**Real-world applications:**
- Office / workplace entry compliance monitoring
- Airport and public transport safety checks
- Retail store and hospital entrance screening
- CCTV-based automated alerts

---

## 🏗️ System Architecture

```
Input (Image / Webcam Frame)
         │
         ▼
┌─────────────────────────┐
│   Stage 1: Face         │
│   Detection             │
│   RetinaFace (PyTorch)  │
│   Backbone: MobileNet   │
│   0.25 / ResNet-50      │
└─────────┬───────────────┘
          │  Face bounding boxes + landmarks
          ▼
┌─────────────────────────┐
│   Stage 2: Mask         │
│   Classification        │
│   MobileNetV2 (Keras)   │
│   With Mask / No Mask   │
└─────────┬───────────────┘
          │
          ▼
Output: Annotated frame with labels & bounding boxes
```

---

## 🗂️ Project Structure

```
Face-Mask/
│
├── data/                        # Dataset configs and WIDER FACE data structure
├── dataset_cropped/             # Cropped face images for mask classifier training
├── models/                      # RetinaFace model definitions (backbone + heads)
├── layers/                      # Custom PyTorch layers (PriorBox, MultiBoxLoss etc.)
├── utils/                       # Utility functions (NMS, box decoding, augmentation)
├── widerface_evaluate/          # WiderFace benchmark evaluation scripts
├── curve/                       # Evaluation curves and benchmark result plots
│
├── train.py                     # RetinaFace face detector training script
├── train_mobilenetv2_mask.py    # MobileNetV2 mask classifier fine-tuning
├── detect.py                    # Single image / batch inference
├── realtime_mask_detector.py    # Live webcam mask detection
├── test_mask_classifier.py      # Evaluate mask classifier on test set
├── test_widerface.py            # Evaluate face detector on WiderFace benchmark
├── test_fddb.py                 # Evaluate face detector on FDDB benchmark
├── convert_to_onnx.py           # Export trained model to ONNX format
│
├── mask_classifier_mobilenetv2.h5   # Trained mask classifier weights
└── mask_demo.mp4                    # Demo video output
```

---

## 🔬 Technical Deep Dive

### Stage 1 — Face Detection: RetinaFace (PyTorch)

RetinaFace is a single-stage, dense face localisation model published in [Deng et al., 2019 (arxiv)](https://arxiv.org/abs/1905.00641). It simultaneously predicts:
- **Face bounding boxes**
- **Five facial landmarks** (eyes, nose, mouth corners)

Two backbone options were used depending on the deployment context:

| Backbone | Model Size | Use Case |
|---|---|---|
| MobileNet0.25 | ~1.7 MB | Edge / real-time deployment |
| ResNet-50 | Larger | Higher accuracy scenarios |

**Training:** Fine-tuned on the [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) dataset — the standard benchmark for face detection research.

**WiderFace Benchmark Results (this implementation):**

| Backbone | Easy | Medium | Hard |
|---|---|---|---|
| ResNet-50 (same param as MxNet) | 94.82% | 93.84% | 89.60% |
| MobileNet0.25 | 90.70% | 88.16% | 73.82% |

**FDDB Benchmark Results:**

| Backbone | Performance |
|---|---|
| MobileNet0.25 | 98.64% |
| ResNet-50 | 99.22% |

### Stage 2 — Mask Classification: MobileNetV2

- Fine-tuned a pre-trained **MobileNetV2** (ImageNet weights) on a labelled dataset of cropped face images: `with_mask` and `without_mask` classes.
- MobileNetV2 was chosen for its **low latency and small footprint** — ideal for real-time classification on the output of Stage 1.
- Training script: `train_mobilenetv2_mask.py`
- Saved model: `mask_classifier_mobilenetv2.h5`

### Real-Time Pipeline (`realtime_mask_detector.py`)

- Captures frames from webcam using OpenCV
- Runs RetinaFace to get face bounding boxes per frame
- Crops each detected face and passes it through MobileNetV2
- Overlays bounding boxes with colour-coded labels (`With Mask` / `No Mask`) at live framerate

### ONNX Export (`convert_to_onnx.py`)

- Exported the trained model to **ONNX format** for cross-platform deployment (e.g., TensorRT, OpenVINO, mobile inference engines)
- Enables deployment beyond Python — to C++, mobile, and edge hardware

---

## 📊 Model Performance

**Face Detector (RetinaFace — MobileNet0.25 backbone):**
- WiderFace Easy: **90.70%** | Medium: **88.16%**
- FDDB: **98.64%**

**Mask Classifier (MobileNetV2):**
- Trained on balanced `with_mask` / `without_mask` cropped face dataset
- Evaluated with Accuracy, Precision, Recall, and F1 Score on held-out test set

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Face Detection | PyTorch, RetinaFace |
| Mask Classification | TensorFlow / Keras, MobileNetV2 |
| Real-Time Inference | OpenCV |
| Model Export | ONNX |
| Benchmarking | WiderFace Evaluation, FDDB |
| Dataset | WIDER FACE, custom cropped mask dataset |

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Charu305/Face-Mask.git
cd Face-Mask

# 2. Install dependencies
pip install torch torchvision tensorflow opencv-python numpy onnx

# 3. Download pretrained weights and place in ./weights/
#    mobilenet0.25_Final.pth  OR  Resnet50_Final.pth

# 4. Run real-time webcam detection
python realtime_mask_detector.py

# 5. Run detection on a single image
python detect.py --trained_model weights/mobilenet0.25_Final.pth \
                 --network mobile0.25 --image test.jpg

# 6. Export model to ONNX
python convert_to_onnx.py
```

---

## 💡 Key Learnings & Takeaways

- **Two-stage pipelines are powerful** — separating face detection from mask classification allows each model to be optimised, swapped, or improved independently.
- **MobileNet backbones enable real-time performance** — MobileNet0.25 keeps the model under 2 MB while achieving competitive benchmark scores, making it deployable on edge devices.
- **ONNX export is essential for production** — exporting to ONNX decouples the model from the Python/PyTorch/Keras ecosystem, enabling deployment to C++, mobile (Android/iOS), and hardware accelerators.
- **Benchmark datasets matter** — evaluating on WiderFace (Easy / Medium / Hard splits) and FDDB gives a much more honest picture of detector robustness than accuracy on a simple train/test split.
- **Transfer learning accelerates convergence** — using ImageNet-pretrained MobileNetV2 weights and fine-tuning only the classifier head allowed fast, stable training on a relatively small mask dataset.

---

## 📁 Datasets Used

| Dataset | Purpose |
|---|---|
| [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) | Face detector training & evaluation |
| [FDDB](http://vis-www.cs.umass.edu/fddb/) | Face detector benchmark evaluation |
| Custom cropped dataset (`dataset_cropped/`) | Mask classifier training (with\_mask / without\_mask) |

---

## 📄 References

- [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641) — Deng et al., 2019
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) — Sandler et al., 2018
- [Pytorch RetinaFace](https://github.com/biubug6/Pytorch_Retinaface) — base implementation reference

---

## 👩‍💻 Author

**Charunya** — Deep Learning & Computer Vision
🔗 [GitHub Profile](https://github.com/Charu305)

---

## 📄 License

MIT License — see [LICENSE.MIT](./LICENSE.MIT) for details.

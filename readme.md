# ASL ABEF Edge AI Demo

This repository contains a **real-time American Sign Language (ASL) A/B/E/F hand gesture recognition demo** optimized for **Edge AI** devices. The project demonstrates end-to-end inference using a lightweight model on local hardware, highlighting low latency and real-time performance, even on limited resources.

---

## Features

- **CSV Test Sample Recognition**  
  Load and display hand gestures from the SignMNIST CSV dataset. Press `n` to cycle through different test samples.

- **External Image Recognition**  
  Press `o` to open a file dialog and select any image (PNG, JPG, JPEG, BMP) for recognition. The image is automatically preprocessed to 28×28 grayscale for model inference.

- **Simulated Real-Time Stream**  
  Loop through CSV test images at a configurable interval (default 100ms) to simulate real-time edge inference.

- **Camera Input Mode**  
  Use a connected webcam to capture hand gestures in real-time and perform live inference.

- **Edge AI Demonstration**  
  Displays per-frame **inference time (ms)** and **FPS** in the window and terminal to emphasize low-latency, on-device processing.  

- **Lightweight Model**  
  The model is designed for edge deployment, with small file size and fast inference suitable for devices with limited resources.

---

## Requirements

- Python 3.10+  
- PyTorch  
- OpenCV (`opencv-python`)  
- NumPy  
- Pandas  
- Tkinter (for file selection dialog)  

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/<username>/asl_edge_ai_demo.git
cd asl_edge_ai_demo
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Linux / macOS
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Test CSV samples (manual mode)

```bash
python run_asl_abef_test_demo.py
```

- Press `n` to go to the next test sample.  
- Press `q` to quit.

### 2. External image recognition

- Run the demo normally, then press `o` to select an image from your file system.  
- The image will be resized to 28×28 grayscale and recognized by the model.

### 3. Simulated real-time stream

```bash
python run_asl_abef_test_demo.py --simulate-stream --stream-delay-ms 100
```

- Loops through CSV test samples every 100ms to simulate streaming inference.  
- Displays inference time and FPS.

### 4. Camera mode

```bash
python run_asl_abef_test_demo.py --use-camera
```

- Use a webcam to capture live hand gestures.  
- Displays predictions in real-time with inference time and FPS.

---

## Files in this repository

- `run_asl_abef_test_demo.py` – Main demo script  
- `asl_abef_model.py` – CNN model definition  
- `models/` – Pretrained model checkpoint and config  
- `sign_mnist_test.csv` – Test dataset (A/B/E/F hand gestures)  
- `README.md` – Project description and usage instructions  

---

## Edge AI Considerations

- All inference runs locally on the device.  
- Low-latency predictions suitable for real-time applications.  
- Lightweight and optimized for deployment on edge devices.  
- Performance metrics (ms and FPS) are displayed to highlight real-time capabilities.  

---

## References

- SignMNIST Dataset: [https://www.kaggle.com/datamunge/sign-language-mnist](https://www.kaggle.com/datamunge/sign-language-mnist)  
- PyTorch: [https://pytorch.org](https://pytorch.org)  
- OpenCV: [https://opencv.org](https://opencv.org)


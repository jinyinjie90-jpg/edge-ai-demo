# ASL ABEF Edge AI Project

这个版本用于 Sign Language MNIST 数据集，只训练 4 个字母：

```text
A, B, E, F
```

对应原始 Sign Language MNIST 标签：

```text
A = 0
B = 1
E = 4
F = 5
```

训练时会自动把它们重新映射成：

```text
A -> 0
B -> 1
E -> 2
F -> 3
```

这样模型输出就是 4 类，更适合你的 Edge AI 项目。

---

## 你需要准备的数据

把 Kaggle / Sign Language MNIST 的两个 CSV 文件放到本项目目录：

```text
sign_mnist_train.csv
sign_mnist_test.csv
```

目录应该类似：

```text
asl_abef_edge_ai_project/
├── sign_mnist_train.csv
├── sign_mnist_test.csv
├── asl_abef_model.py
├── train_asl_abef.py
├── run_asl_abef_test_demo.py
├── benchmark_asl_abef.py
└── webcam_asl_abef_demo.py
```

---

## 1. 安装依赖

```powershell
python -m pip install torch torchvision pandas numpy scikit-learn pillow opencv-python
```

如果你要控制 PPT：

```powershell
python -m pip install pyautogui
```

---

## 2. 训练 ABEF 四类模型

```powershell
python train_asl_abef.py --train-csv ".\sign_mnist_train.csv" --test-csv ".\sign_mnist_test.csv" --epochs 10 --batch-size 64
```

训练后会生成：

```text
models/asl_abef_cnn.pt
models/asl_abef_cnn.json
```

---

## 3. 测试集本地推理 Demo

这个最稳，推荐课堂展示用。

```powershell
python run_asl_abef_test_demo.py --model-path ".\models\asl_abef_cnn.pt" --config-path ".\models\asl_abef_cnn.json" --test-csv ".\sign_mnist_test.csv"
```

它会随机抽取测试集图片，显示：

```text
True Label
Predicted Label
Confidence
Inference Time
FPS estimate
```

按：

```text
n = 下一张
q = 退出
```

---

## 4. Edge AI Benchmark

```powershell
python benchmark_asl_abef.py --model-path ".\models\asl_abef_cnn.pt" --config-path ".\models\asl_abef_cnn.json"
```

输出：

```text
model_file_size_mb
average_inference_time_ms
estimated_fps
device
image_size
input_channels
```

这些数据可以直接放报告。

---

## 5. 摄像头可选 Demo

注意：Sign Language MNIST 是裁剪好的 28×28 灰度手势图，普通摄像头真实画面和训练集分布不同，所以摄像头版本不一定稳定。课堂展示建议优先使用测试集 demo。

```powershell
python webcam_asl_abef_demo.py --model-path ".\models\asl_abef_cnn.pt" --config-path ".\models\asl_abef_cnn.json"
```

如果要启用 PPT 控制：

```powershell
python webcam_asl_abef_demo.py --model-path ".\models\asl_abef_cnn.pt" --config-path ".\models\asl_abef_cnn.json" --enable-control
```

默认映射：

```text
A -> next_slide
B -> previous_slide
E -> start_pause
F -> confirm
```

---

## 推荐项目表述

This project uses a subset of the Sign Language MNIST dataset to build an Edge AI-based ASL letter recognition system. 
Four static ASL letters, A, B, E, and F, are selected for lightweight local inference. 
A compact CNN is trained on 28×28 grayscale images and deployed locally. 
The system is evaluated using accuracy, model size, inference latency, FPS, and local deployment feasibility.

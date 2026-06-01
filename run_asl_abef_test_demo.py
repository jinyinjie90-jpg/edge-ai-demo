import argparse
import json
import random
import time
from pathlib import Path
import os

import cv2
import numpy as np
import pandas as pd
import torch
from tkinter import Tk
from tkinter.filedialog import askopenfilename

from asl_abef_model import ASLABEFCNN

ORIGINAL_TO_NEW = {0: 0, 1: 1, 4: 2, 5: 3}
CLASS_NAMES = ["A", "B", "E", "F"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/asl_abef_cnn.pt")
    parser.add_argument("--config-path", default="models/asl_abef_cnn.json")
    parser.add_argument("--test-csv", default="sign_mnist_test.csv")
    parser.add_argument("--use-camera", action="store_true", help="Use camera for real-time recognition")
    parser.add_argument("--simulate-stream", action="store_true", help="Simulate real-time stream from test images")
    parser.add_argument("--stream-delay-ms", type=int, default=100, help="Delay between frames in simulated stream mode")
    return parser.parse_args()

def load_test_samples(test_csv: str):
    df = pd.read_csv(test_csv)
    df = df[df["label"].isin(ORIGINAL_TO_NEW.keys())].copy()
    labels_original = df["label"].astype(int).values
    labels_new = np.array([ORIGINAL_TO_NEW[int(x)] for x in labels_original], dtype=np.int64)
    pixels = df.drop(columns=["label"]).values.astype(np.float32)
    images = pixels.reshape(-1, 1, 28, 28) / 255.0
    return images, labels_new

def load_model(model_path: str, config_path: str, device):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    class_names = config.get("class_names", CLASS_NAMES)
    checkpoint = torch.load(model_path, map_location=device)
    model = ASLABEFCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, class_names

@torch.no_grad()
def predict_one(model, image_np, device):
    tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).to(device)
    start = time.perf_counter()
    logits = model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    probs = torch.softmax(logits, dim=1)[0]
    conf, pred_idx = torch.max(probs, dim=0)
    inference_ms = (end - start) * 1000
    return int(pred_idx.item()), float(conf.item()), inference_ms

def make_canvas(image, true_name, pred_name, confidence, inference_ms, fps_est, mode_text):
    img_28 = (image[0] * 255).astype(np.uint8)
    display = cv2.resize(img_28, (280,280), interpolation=cv2.INTER_NEAREST)
    display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
    canvas = np.zeros((420,520,3), dtype=np.uint8)
    canvas[20:300, 120:400, :] = display[:280,:280]
    cv2.putText(canvas, f"True: {true_name}", (20, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
    cv2.putText(canvas, f"Pred: {pred_name} ({confidence:.2f})", (20, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255),2)
    cv2.putText(canvas, f"{inference_ms:.2f} ms | {fps_est:.1f} FPS", (20,405), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255),2)
    cv2.putText(canvas, mode_text, (20,25), cv2.FONT_HERSHEY_SIMPLEX,0.6,(180,180,180),1)
    return canvas

def select_external_image():
    root = Tk()
    root.withdraw()
    file_path = askopenfilename(title="Select an image file", filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
    root.destroy()
    if not file_path:
        return None, None
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (28,28), interpolation=cv2.INTER_NEAREST)
    img_input = img_resized.astype(np.float32)/255.0
    img_input = img_input[np.newaxis,:,:]
    true_name = os.path.basename(file_path).split('.')[0]
    return img_input, true_name

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(args.model_path, args.config_path, device)

    images, labels = None, None
    if not args.use_camera and not args.simulate_stream:
        images, labels = load_test_samples(args.test_csv)

    window_name = "ASL ABEF Edge AI Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name,700,700)

    idx = 0
    while True:
        key = cv2.waitKey(1) & 0xFF

        # 按 'o' 弹出选择文件窗口识别图片
        if key == ord('o'):
            img_input, true_name = select_external_image()
            if img_input is not None:
                pred_label, confidence, inference_ms = predict_one(model, img_input, device)
                pred_name = class_names[pred_label]
                fps_est = 1000.0 / max(inference_ms, 1e-6)
                canvas = make_canvas(img_input, true_name, pred_name, confidence, inference_ms, fps_est, "User loaded image")
                cv2.imshow(window_name, canvas)

        # CSV 测试集手动切换
        if images is not None and key == ord('n'):
            img_input = images[idx]
            true_label = labels[idx]
            true_name = class_names[true_label]
            idx = (idx+1) % len(images)
            mode_text = "CSV sample mode"

            pred_label, confidence, inference_ms = predict_one(model,img_input,device)
            pred_name = class_names[pred_label]
            fps_est = 1000.0/max(inference_ms,1e-6)
            canvas = make_canvas(img_input,true_name,pred_name,confidence,inference_ms,fps_est,mode_text)
            cv2.imshow(window_name,canvas)

        # 按 'q' 退出
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__=="__main__":
    main()

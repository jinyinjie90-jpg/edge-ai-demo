import argparse
import json
import time
from pathlib import Path

import torch

from asl_abef_model import ASLABEFCNN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/asl_abef_cnn.pt")
    parser.add_argument("--config-path", default="models/asl_abef_cnn.json")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=300)
    return parser.parse_args()


def load_model(model_path: str, config_path: str, device):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    class_names = config.get("class_names", ["A", "B", "E", "F"])

    checkpoint = torch.load(model_path, map_location=device)
    model = ASLABEFCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    return model, class_names


@torch.no_grad()
def benchmark(model, device, warmup, runs):
    dummy = torch.randn(1, 1, 28, 28).to(device)

    for _ in range(warmup):
        _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(runs):
        _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    avg_ms = (end - start) * 1000.0 / runs
    fps = 1000.0 / avg_ms
    return avg_ms, fps


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(args.model_path, args.config_path, device)

    model_size_mb = Path(args.model_path).stat().st_size / (1024 * 1024)
    avg_ms, fps = benchmark(model, device, args.warmup, args.runs)

    print("========== Edge AI Benchmark ==========")
    print(f"model_path: {args.model_path}")
    print(f"num_classes: {len(class_names)}")
    print(f"class_names: {class_names}")
    print("image_size: 28")
    print("input_channels: 1")
    print(f"device: {device}")
    print(f"model_file_size_mb: {model_size_mb:.4f}")
    print(f"average_inference_time_ms: {avg_ms:.4f}")
    print(f"estimated_fps: {fps:.2f}")


if __name__ == "__main__":
    main()

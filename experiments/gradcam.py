"""Grad-CAM for a YOLO11 detection feature layer.

The scalar target is the maximum pre-NMS class probability from the detector.
This makes the target definition explicit and reproducible for object detection.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from agm_yolo.model import load_model


def letterbox(im: np.ndarray, size: int):
    h, w = im.shape[:2]
    gain = min(size / h, size / w)
    nw, nh = int(round(w * gain)), int(round(h * gain))
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, gain, left, top


def last_conv(module: nn.Module) -> nn.Conv2d:
    convs = [m for m in module.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("Selected layer contains no Conv2d")
    return convs[-1]


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--layer", type=int, default=22, help="YOLO model layer containing target Conv2d")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--out", default="results/gradcam.png")
    p.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay opacity")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() and torch.cuda.is_available() else args.device)
    wrapper = load_model(args.weights)
    net = wrapper.model.to(device).eval()
    target_module = last_conv(net.model[args.layer])
    state = {}

    def hook(_module, _inputs, output):
        state["activation"] = output
        output.retain_grad()

    handle = target_module.register_forward_hook(hook)
    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(args.image)
    canvas, _gain, _left, _top = letterbox(bgr, args.imgsz)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    net.zero_grad(set_to_none=True)
    output = net(x)
    pred = output[0] if isinstance(output, tuple) else output
    if pred.ndim != 3 or pred.shape[1] <= 4:
        raise RuntimeError(f"Unexpected detector output shape: {tuple(pred.shape)}")
    target = pred[:, 4:, :].max()
    target.backward()
    activation = state["activation"]
    gradient = activation.grad
    if gradient is None:
        raise RuntimeError("No gradient captured at target layer")
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(args.imgsz, args.imgsz), mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    cam -= cam.min(); cam /= max(cam.max(), 1e-12)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(canvas, 1.0 - args.alpha, heat, args.alpha, 0.0)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), overlay)
    handle.remove()
    print(out)


if __name__ == "__main__":
    main()

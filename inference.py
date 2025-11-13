# inference.py
from pathlib import Path
import argparse
from typing import List

import torch
from PIL import Image
from torchvision import transforms

from model import SimpleCNN


def load_image(path: Path, img_size=128):
    if not path.exists():
        raise FileNotFoundError(str(path))
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    return tf(Image.open(path).convert("RGB")).unsqueeze(0)


@torch.no_grad()
def predict(img_path: Path, weights_path: Path):
    payload = torch.load(weights_path, map_location="cpu")

    meta = payload["meta"]
    classes: List[str] = meta["classes"]
    num_classes = meta["num_classes"]
    img_size = meta["img_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    model = SimpleCNN(num_classes).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    x = load_image(img_path, img_size=img_size).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    top_prob, top_idx = probs.max(dim=1)

    pred = classes[top_idx.item()]
    prob = float(top_prob.item())
    print(f"[RESULT] Predict = {pred}, prob={prob:.3f}")
    return pred, prob


def main():
    proj_root = Path(__file__).resolve().parents[1]
    default_img = proj_root / "data" / "Project Frames" / "dusty_1" / "dusty_1_frame0.jpg"
    default_w = proj_root / "saved" / "model.pth"

    parser = argparse.ArgumentParser()
    parser.add_argument("--img", default=str(default_img))
    parser.add_argument("--weights", default=str(default_w))
    args = parser.parse_args()

    predict(Path(args.img), Path(args.weights))


if __name__ == "__main__":
    main()











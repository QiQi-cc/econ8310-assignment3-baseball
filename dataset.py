# dataset.py
from pathlib import Path
from typing import List, Tuple, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ProjectFramesDataset(Dataset):
    def __init__(self, root_dir: Path, img_size: int = 128):
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root_dir.resolve()}")

        classes = [p.name for p in sorted(self.root_dir.iterdir()) if p.is_dir()]
        if not classes:
            raise RuntimeError(f"No class subfolders found under: {self.root_dir}")

        self.classes: List[str] = classes
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}

        exts = {".jpg", ".jpeg", ".png"}
        self.samples: List[Tuple[Path, int]] = []
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            for p in sorted(cls_dir.rglob("*")):
                if p.suffix.lower() in exts:
                    self.samples.append((p, self.class_to_idx[cls]))

        if not self.samples:
            raise RuntimeError(f"No images found under: {self.root_dir}")

        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.tf(img)
        return x, label


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "data" / "Project Frames"
    ds = ProjectFramesDataset(root)
    print(len(ds), "frames")
    print("Classes:", ds.classes)



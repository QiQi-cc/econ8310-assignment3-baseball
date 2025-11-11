# dataset.py
from pathlib import Path
from typing import List, Tuple, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ProjectFramesDataset(Dataset):
    """
    Image classification dataset that treats each immediate subfolder under
    root_dir as one class. Exposes:
      - self.classes: List[str]
      - self.class_to_idx: Dict[str, int]
      - self.idx_to_class: Dict[int, str]
    Each sample returns (image_tensor, class_index).
    """

    def __init__(self, root_dir: Path, img_size: int = 128):
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root_dir.resolve()}")

        # Discover classes (subfolders)
        classes = [p.name for p in sorted(self.root_dir.iterdir()) if p.is_dir()]
        if not classes:
            raise RuntimeError(f"No class subfolders found under: {self.root_dir}")
        self.classes: List[str] = classes
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class: Dict[int, str] = {i: c for c, i in self.class_to_idx.items()}

        # Collect image paths
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.samples: List[Tuple[Path, int]] = []
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            for p in sorted(cls_dir.rglob("*")):
                if p.suffix.lower() in exts:
                    self.samples.append((p, self.class_to_idx[cls]))

        if not self.samples:
            raise RuntimeError(f"No images found under: {self.root_dir}")

        # Basic transforms only (simple student-style pipeline)
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # [0,1]
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.tf(img)
        return x, label


# Quick sanity check
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1] / "data" / "Project Frames"
    ds = ProjectFramesDataset(ROOT, img_size=128)
    print(f"Number of frames: {len(ds)}")
    x, y = ds[0]
    name = ds.idx_to_class[y]
    print("One sample:", x.shape, name)



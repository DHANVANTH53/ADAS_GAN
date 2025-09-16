# utils/dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class PairedWeatherDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=256):
        """
        expects folder structure:
        root_dir/train/rainy/*.png
        root_dir/train/clean/*.png
        same for val/test
        """
        self.rainy_dir = os.path.join(root_dir, split, 'rainy')
        self.clean_dir = os.path.join(root_dir, split, 'clean')
        if not os.path.isdir(self.rainy_dir) or not os.path.isdir(self.clean_dir):
            raise RuntimeError(f"Dataset directories not found: {self.rainy_dir} or {self.clean_dir}")
        self.rainy_files = sorted([os.path.join(self.rainy_dir, f) for f in os.listdir(self.rainy_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        self.clean_files = sorted([os.path.join(self.clean_dir, f) for f in os.listdir(self.clean_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        if len(self.rainy_files) != len(self.clean_files):
            raise RuntimeError("paired dataset lengths mismatch - ensure equal counts or provide mapping script")
        self.img_size = img_size
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])

    def __len__(self): return len(self.rainy_files)

    def __getitem__(self, idx):
        rainy = Image.open(self.rainy_files[idx]).convert('RGB')
        clean = Image.open(self.clean_files[idx]).convert('RGB')
        rainy = self.transform(rainy)
        clean = self.transform(clean)
        return rainy, clean

def get_dataloader(root_dir, split='train', img_size=256, batch_size=8, shuffle=True, num_workers=4, pin_memory=True):
    ds = PairedWeatherDataset(root_dir, split=split, img_size=img_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

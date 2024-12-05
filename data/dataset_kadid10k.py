
import pandas as pd
import re
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import ImageFilter
import random

# 왜곡 유형 매핑
distortion_types_mapping = {
    1: "gaussian_blur",
    2: "lens_blur",
    3: "motion_blur",
    4: "color_diffusion",
    5: "color_shift",
    6: "color_quantization",
    7: "color_saturation_1",
    8: "color_saturation_2",
    9: "jpeg2000",
    10: "jpeg",
    11: "white_noise",
    12: "white_noise_color_component",
    13: "impulse_noise",
    14: "multiplicative_noise",
    15: "denoise",
    16: "brighten",
    17: "darken",
    18: "mean_shift",
    19: "jitter",
    20: "non_eccentricity_patch",
    21: "pixelate",
    22: "quantization",
    23: "color_block",
    24: "high_sharpen",
    25: "contrast_change"
}

class KADID10KDataset(Dataset):
    def __init__(self, root: str, phase: str = "train", split_idx: int = 0, crop_size: int = 224):
        """
        Initialize the KADID10KDataset.
        
        Args:
            root (str): Path to the dataset directory or directly to the CSV file.
            phase (str): One of "train", "val", "test", or "all".
            split_idx (int): Index of the dataset split to use.
            crop_size (int): Size to which images will be cropped/resized.
        """
        super().__init__()
        self.root = Path(root)
        self.phase = phase
        self.crop_size = crop_size

        # Check if root is a file (CSV path) or directory
        if self.root.is_file():
            csv_path = self.root
            self.dataset_root = self.root.parent  # Parent directory of the CSV file
        else:
            csv_path = self.root / "kadid10k.csv"
            self.dataset_root = self.root

        # Load scores from CSV
        scores_csv = pd.read_csv(csv_path)
        scores_csv = scores_csv[["dist_img", "ref_img", "dmos"]]

        # Set image paths
        self.images = np.array([self.dataset_root / "images" / img for img in scores_csv["dist_img"].values])
        self.ref_images = np.array([self.dataset_root / "images" / img for img in scores_csv["ref_img"].values])
        self.mos = np.array(scores_csv["dmos"].values.tolist())

        self.distortion_types = []
        self.distortion_levels = []

        for img in self.images:
            # Extract distortion type and level from image name
            match = re.search(r'I\d+_(\d+)_(\d+)\.png$', str(img))
            if match:
                dist_type = distortion_types_mapping[int(match.group(1))]
                self.distortion_types.append(dist_type)
                self.distortion_levels.append(int(match.group(2)))

        self.distortion_types = np.array(self.distortion_types)
        self.distortion_levels = np.array(self.distortion_levels)

        # Handle train/val/test splits
        if self.phase != "all":
            split_file_path = self.dataset_root / "splits" / f"{self.phase}.npy"
            if not split_file_path.exists():
                raise FileNotFoundError(f"Split file not found: {split_file_path}")
            split_idxs = np.load(split_file_path)[split_idx]
            split_idxs = np.array(list(filter(lambda x: x != -1, split_idxs)))  # Remove padding indices
            self.images = self.images[split_idxs]
            self.ref_images = self.ref_images[split_idxs]
            self.mos = self.mos[split_idxs]
            self.distortion_types = self.distortion_types[split_idxs]
            self.distortion_levels = self.distortion_levels[split_idxs]


    def transform(self, image: Image) -> torch.Tensor:
        # Transform image to desired size and convert to tensor
        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])(image)
    
    def apply_random_distortions(image, num_distortions=4):
        distortions = [
            lambda img: img.filter(ImageFilter.GaussianBlur(radius=2)),
            lambda img: img.rotate(15),
            lambda img: img.filter(ImageFilter.UnsharpMask(radius=2, percent=150)),
            lambda img: img.resize((int(img.width * 0.8), int(img.height * 0.8))),
        ]
        for _ in range(num_distortions):
            distortion = random.choice(distortions)
            image = distortion(image)
        return image


    def apply_distortion(self, image: torch.Tensor) -> torch.Tensor:
        pil_image = transforms.ToPILImage()(image)
        distortions = [
            lambda img: img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2))),
            lambda img: img.filter(ImageFilter.UnsharpMask(radius=random.uniform(1, 2))),
            lambda img: img.filter(ImageFilter.DETAIL),
            lambda img: img.filter(ImageFilter.EDGE_ENHANCE),
        ]
        # 최대 4개의 왜곡 순차 적용
        num_distortions = random.randint(1, 4)
        for _ in range(num_distortions):
            distortion = random.choice(distortions)
            pil_image = distortion(pil_image)
        return transforms.ToTensor()(pil_image)



    def __getitem__(self, index: int):
        img_A_orig = Image.open(self.images[index]).convert("RGB")
        img_B_orig = Image.open(self.ref_images[index]).convert("RGB")

        # 경음성 쌍: 50% 축소
        img_A_cropped = img_A_orig.resize((img_A_orig.width // 2, img_A_orig.height // 2))
        img_B_cropped = img_B_orig.resize((img_B_orig.width // 2, img_B_orig.height // 2))

        img_A_orig = self.transform(img_A_orig)
        img_B_orig = self.transform(img_B_orig)
        img_A_cropped = self.transform(img_A_cropped)
        img_B_cropped = self.transform(img_B_cropped)

        return {
            "img_A": torch.stack([img_A_orig, img_A_cropped]),
            "img_B": torch.stack([img_B_orig, img_B_cropped]),
            "mos": self.mos[index],
        }


    def __len__(self):
        return len(self.images)

    def get_split_indices(self, split: int, phase: str) -> np.ndarray:
        split_file_path = self.root / "splits" / f"{phase}.npy"
        split_indices = np.load(split_file_path)[split]
        return split_indices
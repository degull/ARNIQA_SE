""" import pandas as pd
import re
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from pathlib import Path
from typing import List, Tuple, Optional

# Distortion types mapping
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
        self.root = Path(root)
        self.phase = phase
        self.crop_size = crop_size

        # is_synthetic 속성 추가 (KADID10K는 synthetic 데이터셋이므로 True)
        self.is_synthetic = True

        # Load scores from CSV
        scores_csv = pd.read_csv(self.root / "kadid10k.csv")
        scores_csv = scores_csv[["dist_img", "ref_img", "dmos"]]

        self.images = scores_csv["dist_img"].values.tolist()
        self.images = np.array([self.root / "images" / el for el in self.images])

        self.ref_images = scores_csv["ref_img"].values.tolist()
        self.ref_images = np.array([self.root / "images" / el for el in self.ref_images])

        self.mos = np.array(scores_csv["dmos"].values.tolist())

        if self.phase != "all":
            split_idxs = np.load(self.root / "splits" / f"{self.phase}.npy")[split_idx]
            split_idxs = np.array(list(filter(lambda x: x != -1, split_idxs)))
            self.images = self.images[split_idxs]
            self.ref_images = self.ref_images[split_idxs]
            self.mos = self.mos[split_idxs]

        self.distortion_types = []
        for image in self.images:
            match = re.search(r'I\d+_(\d+)_(\d+)\.png$', str(image))
            dist_type = distortion_types_mapping[int(match.group(1))] if match else None
            self.distortion_types.append(dist_type)

    def apply_distortion(self, image):
        if random.random() > 0.5:
            pil_image = transforms.ToPILImage()(image)
            distorted_image = pil_image.filter(ImageFilter.GaussianBlur(radius=2))
            return transforms.ToTensor()(distorted_image)
        return image

    def __getitem__(self, index: int) -> Optional[dict]:
        try:
            img_A_orig = Image.open(self.images[index]).convert("RGB")
            img_B_orig = Image.open(self.ref_images[index]).convert("RGB")
        except Exception as e:
            print(f"Error loading images at index {index}: {e}")
            return None  # None 반환으로 변경

        img_A_orig = self.transform(img_A_orig)
        img_B_orig = self.transform(img_B_orig)

        # Crops 생성 및 스택 이미지
        crops_A = [img_A_orig]
        crops_B = [img_B_orig]

        # 추가적인 crops 적용
        crops_A += [self.apply_distortion(img_A_orig) for _ in range(3)]
        crops_B += [self.apply_distortion(img_B_orig) for _ in range(3)]

        # Stack crops
        img_A = torch.stack(crops_A)
        img_B = torch.stack(crops_B)

        # Batch 차원을 추가하여 모델에 전달
        img_A = img_A.unsqueeze(0)
        img_B = img_B.unsqueeze(0)

        return {
            "img_A_orig": img_A,
            "img_B_orig": img_B,
            "img_A_ds": img_A,
            "img_B_ds": img_B,
            "mos": self.mos[index],
            "distortion_type": self.distortion_types[index],
        }
    def transform(self, image):
        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])(image)

    def __len__(self):
        return len(self.images)

    def get_split_indices(self, split: int, phase: str) -> np.ndarray:

        split_file_path = self.root / "splits" / f"{phase}.npy"
        split_indices = np.load(split_file_path)[split]
        return split_indices  """

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
        super().__init__()
        self.root = Path(root)
        self.phase = phase
        self.crop_size = crop_size

        # CSV 파일에서 점수 로드
        scores_csv = pd.read_csv(self.root / "kadid10k.csv")
        scores_csv = scores_csv[["dist_img", "ref_img", "dmos"]]

        # 이미지 경로 설정
        self.images = np.array([self.root / "images" / img for img in scores_csv["dist_img"].values])
        self.ref_images = np.array([self.root / "images" / img for img in scores_csv["ref_img"].values])
        self.mos = np.array(scores_csv["dmos"].values.tolist())

        self.distortion_types = []
        self.distortion_levels = []

        for img in self.images:
            # 이미지 이름에서 왜곡 유형과 레벨 추출
            match = re.search(r'I\d+_(\d+)_(\d+)\.png$', str(img))
            if match:
                dist_type = distortion_types_mapping[int(match.group(1))]
                self.distortion_types.append(dist_type)
                self.distortion_levels.append(int(match.group(2)))

        self.distortion_types = np.array(self.distortion_types)
        self.distortion_levels = np.array(self.distortion_levels)

        if self.phase != "all":
            split_idxs = np.load(self.root / "splits" / f"{self.phase}.npy")[split_idx]
            split_idxs = np.array(list(filter(lambda x: x != -1, split_idxs)))  # 패딩 제거
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
    
    def apply_distortion(self, image: torch.Tensor) -> torch.Tensor:
        # Convert tensor to PIL image
        pil_image = transforms.ToPILImage()(image)
        
        # Apply a random distortion (e.g., Gaussian Blur)
        if random.random() > 0.5:
            return transforms.ToTensor()(pil_image.filter(ImageFilter.GaussianBlur(radius=2)))
        
        return image  # Return original image if no distortion applied


    def __getitem__(self, index: int) -> dict:
        img_A_orig = Image.open(self.images[index]).convert("RGB")
        img_B_orig = Image.open(self.ref_images[index]).convert("RGB")

        img_A_orig = self.transform(img_A_orig)
        img_B_orig = self.transform(img_B_orig)

        # Create crops and stack images
        crops_A = [img_A_orig]
        crops_B = [img_B_orig]

        # Apply additional crops
        crops_A += [self.apply_distortion(img_A_orig) for _ in range(3)]
        crops_B += [self.apply_distortion(img_B_orig) for _ in range(3)]

        # Stack crops
        img_A = torch.stack(crops_A)  # Shape: [num_crops, 3, crop_size, crop_size]
        img_B = torch.stack(crops_B)  # Shape: [num_crops, 3, crop_size, crop_size]

        # Reshape to [1, num_crops, 3, crop_size, crop_size]
        img_A = img_A.unsqueeze(0)
        img_B = img_B.unsqueeze(0)

        return {
            "img_A_orig": img_A,
            "img_B_orig": img_B,
            "img_A_ds": img_A,  # img_A_ds를 추가합니다
            "img_B_ds": img_B,  # img_B_ds를 추가합니다
            "mos": self.mos[index],
        }

    def __len__(self):
        return len(self.images)

    def get_split_indices(self, split: int, phase: str) -> np.ndarray:
        split_file_path = self.root / "splits" / f"{phase}.npy"
        split_indices = np.load(split_file_path)[split]
        return split_indices
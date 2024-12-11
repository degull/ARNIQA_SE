""" 
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
        return split_indices """


# 왜곡 그룹화: distortion_groups 딕셔너리를 사용하여 7개의 그룹을 정의 -> 각 그룹에 여러 개의 왜곡을 포함시키고, 각 그룹에서 하나 이상의 왜곡을 선택하도록
# 왜곡 적용: apply_random_distortions 함수에서 각 그룹에서 랜덤으로 왜곡을 선택하고, 그 왜곡을 이미지에 동시에 적용
# 이미지 왜곡 적용: apply_random_distortions 함수 내에서 다양한 왜곡이 적용되며, 최대 4개의 왜곡이 동시에 적용


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

# 왜곡 그룹 정의 (7개 그룹)
distortion_groups = {
    'blur': [1, 2, 3],  # gaussian_blur, lens_blur, motion_blur
    'color': [4, 5, 6, 7, 8],  # color_diffusion, color_shift, color_quantization, color_saturation_1, color_saturation_2
    'compression': [9, 10],  # jpeg2000, jpeg
    'noise': [11, 12, 13, 14],  # white_noise, white_noise_color_component, impulse_noise, multiplicative_noise
    'enhancement': [15, 16, 17],  # denoise, brighten, darken
    'shift': [18, 19],  # mean_shift, jitter
    'others': [20, 21, 22, 23, 24, 25]  # non_eccentricity_patch, pixelate, quantization, color_block, high_sharpen, contrast_change
}

# 각 왜곡에 대한 강도 레벨 정의 (5개 강도)
distortion_levels = {
    'gaussian_blur': [0.5, 1.0, 1.5, 2.0, 2.5],
    'lens_blur': [1, 2, 3, 4, 5],
    'motion_blur': [1, 2, 3, 4, 5],
    'color_diffusion': [0.05, 0.1, 0.2, 0.3, 0.4],
    'color_shift': [-30, -20, -10, 10, 20],
    'color_quantization': [8, 16, 32, 64, 128],
    'color_saturation_1': [0.5, 0.6, 0.7, 0.8, 1.0],
    'color_saturation_2': [0.5, 0.6, 0.7, 0.8, 1.0],
    'jpeg2000': [0.1, 0.2, 0.3, 0.4, 0.5],
    'jpeg': [0.1, 0.2, 0.3, 0.4, 0.5],
    'white_noise': [5, 10, 15, 20, 25],
    'impulse_noise': [0.05, 0.1, 0.2, 0.3, 0.4],
    'multiplicative_noise': [0.1, 0.2, 0.3, 0.4, 0.5],
    'denoise': [0.5, 0.6, 0.7, 0.8, 1.0],
    'brighten': [10, 20, 30, 40, 50],
    'darken': [10, 20, 30, 40, 50],
    'mean_shift': [5, 10, 15, 20, 25],
    'jitter': [5, 10, 15, 20, 25],
    'non_eccentricity_patch': [0.5, 1.0, 1.5, 2.0, 2.5],
    'pixelate': [5, 10, 15, 20, 25],
    'quantization': [2, 4, 8, 16, 32],
    'color_block': [10, 20, 30, 40, 50],
    'high_sharpen': [1, 2, 3, 4, 5],
    'contrast_change': [0.5, 0.6, 0.7, 0.8, 1.0]
}

class KADID10KDataset(Dataset):
    def __init__(self, root: str, phase: str = "train", split_idx: int = 0, crop_size: int = 224):
        super().__init__()
        self.root = Path(root)
        self.phase = phase
        self.crop_size = crop_size

        if self.root.is_file():
            csv_path = self.root
            self.dataset_root = self.root.parent
        else:
            csv_path = self.root / "kadid10k.csv"
            self.dataset_root = self.root

        scores_csv = pd.read_csv(csv_path)
        scores_csv = scores_csv[["dist_img", "ref_img", "dmos"]]

        self.images = np.array([self.dataset_root / "images" / img for img in scores_csv["dist_img"].values])
        self.ref_images = np.array([self.dataset_root / "images" / img for img in scores_csv["ref_img"].values])
        self.mos = np.array(scores_csv["dmos"].values.tolist())

        self.distortion_types = []
        self.distortion_levels = []

        for img in self.images:
            match = re.search(r'I\d+_(\d+)_(\d+)\.png$', str(img))
            if match:
                dist_type = distortion_types_mapping[int(match.group(1))]
                self.distortion_types.append(dist_type)
                self.distortion_levels.append(int(match.group(2)))

        self.distortion_types = np.array(self.distortion_types)
        self.distortion_levels = np.array(self.distortion_levels)

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
        return transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor(),
        ])(image)


    def apply_random_distortions(self, image, num_distortions=4):
        selected_distortions = []
        
        # Define distortion levels for each distortion
        distortion_levels = {
            'gaussian_blur': [1, 2, 3, 4, 5],  # Example level ranges, adjust as needed
            'lens_blur': [1, 2, 3, 4, 5],
            'motion_blur': [1, 2, 3, 4, 5],
            'color_diffusion': [1, 2, 3, 4, 5],
            'color_shift': [1, 2, 3, 4, 5],
            'color_quantization': [1, 2, 3, 4, 5],
            'color_saturation_1': [1, 2, 3, 4, 5],
            'color_saturation_2': [1, 2, 3, 4, 5],
            'jpeg2000': [1, 2, 3, 4, 5],
            'jpeg': [1, 2, 3, 4, 5],
            'white_noise': [1, 2, 3, 4, 5],
            'impulse_noise': [1, 2, 3, 4, 5],
            # Add any other distortions you need to handle with levels
        }
        
        # Ensure distortion_groups have corresponding levels
        for group in distortion_groups.values():
            for distortion in group:
                distortion_name = distortion_types_mapping[distortion]
                
                if distortion_name in distortion_levels:
                    # Randomly pick a distortion from the group and a random level for it
                    level = random.choice(distortion_levels[distortion_name])
                    selected_distortions.append((distortion_name, level))

        # Apply selected distortions with chosen levels
        for distortion, level in selected_distortions:
            if distortion == "gaussian_blur":
                radius = level  # Example: Level could affect the blur radius
                image = image.filter(ImageFilter.GaussianBlur(radius=radius))
            elif distortion == "lens_blur":
                radius = level  # Example: Level could affect the blur radius
                image = image.filter(ImageFilter.GaussianBlur(radius=radius))
            elif distortion == "motion_blur":
                radius = level  # Example: Level could affect the blur intensity
                image = image.filter(ImageFilter.BoxBlur(radius))
            elif distortion == "color_diffusion":
                intensity = level * 5  # Example: Increase diffusion based on level
                image = image.convert("RGB")
                diffused = np.array(image)
                diffused = diffused + np.random.uniform(-intensity, intensity, diffused.shape)
                image = Image.fromarray(np.clip(diffused, 0, 255).astype(np.uint8))
            elif distortion == "color_shift":
                shift_amount = level * 10  # Example: Shift color based on level
                image = image.convert("RGB")
                shifted = np.array(image)
                shifted = shifted + np.random.randint(-shift_amount, shift_amount, shifted.shape)
                image = Image.fromarray(np.clip(shifted, 0, 255).astype(np.uint8))
            elif distortion == "jpeg2000":
                image = image.convert("RGB")
                image = image.resize((image.width // 2, image.height // 2))  # Simulate compression by resizing
            elif distortion == "white_noise":
                noise = np.random.normal(0, level * 5, (image.height, image.width, 3))  # Vary noise intensity based on level
                noisy_image = np.array(image) + noise
                image = Image.fromarray(np.clip(noisy_image, 0, 255).astype(np.uint8))
            elif distortion == "impulse_noise":
                image = image.convert("RGB")
                image = np.array(image)
                prob = level * 0.05  # Probability of noise
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        if random.random() < prob:
                            image[i][j] = np.random.choice([0, 255], size=3)
                image = Image.fromarray(image)
        
        return image
    

    
    def apply_distortion(image, distortion, level):
        if distortion == "gaussian_blur":
            image = image.filter(ImageFilter.GaussianBlur(radius=level))
        elif distortion == "lens_blur":
            image = image.filter(ImageFilter.GaussianBlur(radius=level))
        elif distortion == "motion_blur":
            image = image.filter(ImageFilter.BoxBlur(level))
        elif distortion == "color_diffusion":
            diffused = np.array(image)
            diffused += np.random.uniform(-level, level, diffused.shape)
            image = Image.fromarray(np.clip(diffused, 0, 255).astype(np.uint8))
        elif distortion == "color_shift":
            shifted = np.array(image)
            shifted += np.random.randint(-level, level, shifted.shape)
            image = Image.fromarray(np.clip(shifted, 0, 255).astype(np.uint8))
        elif distortion == "jpeg2000":
            image = image.resize((image.width // 2, image.height // 2))
        elif distortion == "white_noise":
            noise = np.random.normal(0, level, (image.height, image.width, 3))
            noisy_image = np.array(image) + noise
            image = Image.fromarray(np.clip(noisy_image, 0, 255).astype(np.uint8))
        elif distortion == "impulse_noise":
            image = np.array(image)
            prob = 0.1
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if random.random() < prob:
                        image[i][j] = np.random.choice([0, 255], size=3)
            image = Image.fromarray(image)

        return image

    def __getitem__(self, index: int):
        # 원본 이미지 로드
        img_A_orig = Image.open(self.images[index]).convert("RGB")
        img_B_orig = Image.open(self.ref_images[index]).convert("RGB")

        # 두 이미지에 대해 동일한 왜곡을 적용하여 positive pair 생성
        img_A_distorted = self.apply_random_distortions(img_A_orig)
        img_B_distorted = self.apply_random_distortions(img_B_orig)

        # 이미지를 텐서로 변환
        img_A_orig = self.transform(img_A_orig)
        img_B_orig = self.transform(img_B_orig)
        img_A_distorted = self.transform(img_A_distorted)
        img_B_distorted = self.transform(img_B_distorted)

        # 반환: 원본 이미지와 왜곡된 이미지를 각각 쌍으로 묶어서 반환
        return {
            "img_A": torch.stack([img_A_orig, img_A_distorted]),
            "img_B": torch.stack([img_B_orig, img_B_distorted]),
            "mos": self.mos[index],  # MOS (Mean Opinion Score)
        }


    def __len__(self):
        return len(self.images)

    def get_split_indices(self, split: int, phase: str) -> np.ndarray:
        split_file_path = self.root / "splits" / f"{phase}.npy"
        split_indices = np.load(split_file_path)[split]
        return split_indices

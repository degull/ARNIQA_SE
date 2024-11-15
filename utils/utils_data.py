import sys
import os
print(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.distortions import *


from random import randrange
import torchvision.transforms.functional as TF
from typing import List, Callable, Union
from PIL.Image import Image as PILImage
from typing import Tuple
from utils.distortions import *


distortion_groups = {
    "blur": ["gaublur", "lensblur", "motionblur"],
    "color_distortion": ["colordiff", "colorshift", "colorsat1", "colorsat2"],
    "jpeg": ["jpeg2000", "jpeg"],
    "noise": ["whitenoise", "whitenoiseCC", "impulsenoise", "multnoise"],
    "brightness_change": ["brighten", "darken", "meanshift"],
    "spatial_distortion": ["jitter", "noneccpatch", "pixelate", "quantization", "colorblock"],
    "sharpness_contrast": ["highsharpen", "lincontrchange", "nonlincontrchange"],
}

distortion_groups_mapping = {
    "gaublur": "blur",
    "lensblur": "blur",
    "motionblur": "blur",
    "colordiff": "color_distortion",
    "colorshift": "color_distortion",
    "colorsat1": "color_distortion",
    "colorsat2": "color_distortion",
    "jpeg2000": "jpeg",
    "jpeg": "jpeg",
    "whitenoise": "noise",
    "whitenoiseCC": "noise",
    "impulsenoise": "noise",
    "multnoise": "noise",
    "brighten": "brightness_change",
    "darken": "brightness_change",
    "meanshift": "brightness_change",
    "jitter": "spatial_distortion",
    "noneccpatch": "spatial_distortion",
    "pixelate": "spatial_distortion",
    "quantization": "spatial_distortion",
    "colorblock": "spatial_distortion",
    "highsharpen": "sharpness_contrast",
    "lincontrchange": "sharpness_contrast",
    "nonlincontrchange": "sharpness_contrast",
}

distortion_range = {
    "gaublur": [0.1, 0.5, 1, 2, 5],
    "lensblur": [1, 2, 4, 6, 8],
    "motionblur": [1, 2, 4, 6, 10],
    "colordiff": [1, 3, 6, 8, 12],
    "colorshift": [1, 3, 6, 8, 12],
    "colorsat1": [0.4, 0.2, 0.1, 0, -0.4],
    "colorsat2": [1, 2, 3, 6, 9],
    "jpeg2000": [16, 32, 45, 120, 170],
    "jpeg": [43, 36, 24, 7, 4],
    "whitenoise": [0.001, 0.002, 0.003, 0.005, 0.01],
    "whitenoiseCC": [0.0001, 0.0005, 0.001, 0.002, 0.003],
    "impulsenoise": [0.001, 0.005, 0.01, 0.02, 0.03],
    "multnoise": [0.001, 0.005, 0.01, 0.02, 0.05],
    "brighten": [0.1, 0.2, 0.4, 0.7, 1.1],
    "darken": [0.05, 0.1, 0.2, 0.4, 0.8],
    "meanshift": [0, 0.08, -0.08, 0.15, -0.15],
    "jitter": [0.05, 0.1, 0.2, 0.5, 1],
    "noneccpatch": [20, 40, 60, 80, 100],
    "pixelate": [0.01, 0.05, 0.1, 0.2, 0.5],
    "quantization": [20, 16, 13, 10, 7],
    "colorblock": [2, 4, 6, 8, 10],
    "highsharpen": [1, 2, 3, 6, 12],
    "lincontrchange": [0., 0.15, -0.4, 0.3, -0.6],
    "nonlincontrchange": [0.4, 0.3, 0.2, 0.1, 0.05],
}

distortion_functions = {
    "gaublur": gaussian_blur,
    "lensblur": lens_blur,
    "motionblur": motion_blur,
    "colordiff": color_diffusion,
    "colorshift": color_shift,
    "colorsat1": color_saturation1,
    "colorsat2": color_saturation2,
    "jpeg2000": jpeg2000,
    "jpeg": jpeg,
    "whitenoise": white_noise,
    "whitenoiseCC": white_noise_cc,
    "impulsenoise": impulse_noise,
    "multnoise": multiplicative_noise,
    "brighten": brighten,
    "darken": darken,
    "meanshift": mean_shift,
    "jitter": jitter,
    "noneccpatch": non_eccentricity_patch,
    "pixelate": pixelate,
    "quantization": quantization,
    "colorblock": color_block,
    "highsharpen": high_sharpen,
    "lincontrchange": linear_contrast_change,
    "nonlincontrchange": non_linear_contrast_change,
}


def distort_images(image: torch.Tensor, distort_functions: list = None, distort_values: list = None,
                   max_distortions: int = 4, num_levels: int = 5) -> torch.Tensor:

    if distort_functions is None or distort_values is None:
        distort_functions, distort_values = get_distortions_composition(max_distortions, num_levels)

    for distortion, value in zip(distort_functions, distort_values):
        image = distortion(image, value)
        image = image.to(torch.float32)
        image = torch.clip(image, 0, 1)

    return image, distort_functions, distort_values


def get_distortions_composition(max_distortions: int = 7, num_levels: int = 5) -> Tuple[List[Callable], List[Union[int, float]]]:
    #MEAN과 STD는 Gaussian 분포를 생성하기 위한 평균과 표준 편차 값
    # 이 값들은 왜곡 강도를 무작위로 선택하는 데 필요한 확률 분포를 만듭니다.
    MEAN = 0
    STD = 2.5

    num_distortions = random.randint(1, max_distortions)    # 1에서 max_distortions(기본값 7) 사이의 랜덤 숫자를 선택하여, 몇 개의 왜곡을 조합할지 결정
    groups = random.sample(list(distortion_groups.keys()), num_distortions) # distortion_groups에 정의된 왜곡 그룹 중에서 무작위로 선택된 그룹 이름들의 리스트
    distortions = [random.choice(distortion_groups[group]) for group in groups]     # 각 선택된 그룹 내에서 하나의 특정 왜곡을 무작위로 선택하여 distortions 리스트에 추가
    distort_functions = [distortion_functions[dist] for dist in distortions]    # distortion_functions 딕셔너리를 통해 각 왜곡 함수에 해당하는 실제 함수 객체를 가져와 distort_functions 리스트에 추가

    probabilities = [1 / (STD * np.sqrt(2 * np.pi)) * np.exp(-((i - MEAN) ** 2) / (2 * STD ** 2))
                     for i in range(num_levels)]  # probabilities according to a gaussian distribution
    normalized_probabilities = [prob / sum(probabilities)
                                for prob in probabilities]  # normalize probabilities
    distort_values = [np.random.choice(distortion_range[dist][:num_levels], p=normalized_probabilities) for dist
                      in distortions]

    return distort_functions, distort_values


def resize_crop(img: PILImage, crop_size: int = 224, downscale_factor: int = 1) -> PILImage:
    w, h = img.size
    if downscale_factor > 1:
        img = img.resize((w // downscale_factor, h // downscale_factor))
        w, h = img.size

    if crop_size is not None:
        top = randrange(0, max(1, h - crop_size))
        left = randrange(0, max(1, w - crop_size))
        img = TF.crop(img, top, left, crop_size, crop_size)     # Automatically pad with zeros if the crop is out of bounds

    return img


def center_corners_crop(img: PILImage, crop_size: int = 224) -> List[PILImage]:

    width, height = img.size

    # Calculate the coordinates for the center crop and the four corners
    cx = width // 2
    cy = height // 2
    crops = [
        TF.crop(img, cy - crop_size // 2, cx - crop_size // 2, crop_size, crop_size),  # Center
        TF.crop(img, 0, 0, crop_size, crop_size),  # Top-left corner
        TF.crop(img, height - crop_size, 0, crop_size, crop_size),  # Bottom-left corner
        TF.crop(img, 0, width - crop_size, crop_size, crop_size),  # Top-right corner
        TF.crop(img, height - crop_size, width - crop_size, crop_size, crop_size)  # Bottom-right corner
    ]

    return crops

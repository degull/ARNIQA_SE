import math
import numpy as np
from typing import Union, Tuple
import torch
from torch.nn import functional as F
import scipy
import torch.nn as nn

def sign(x: float) -> int:
    return 1 if x >= 0 else -1


def mapmm(x: torch.Tensor) -> torch.Tensor:
    minx = torch.min(x)
    maxx = torch.max(x)
    if minx < maxx:
        x = (x - minx) / (maxx - minx)
    return x


def fspecial(filter_type: str, p2: Union[int, Tuple[int, int]], p3: Union[int, float] = None) -> np.ndarray:
    if filter_type == 'gaussian':
        m, n = [(ss - 1.) / 2. for ss in p2]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * p3 * p3))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh

        return h

    elif filter_type == 'disk':
        rad = p2
        crad = math.ceil(rad - 0.5)

        x, y = np.ogrid[-rad: rad + 1, -rad: rad + 1]
        y = np.tile(y.transpose(), y.shape[1])
        x = np.tile(x, x.shape[0]).transpose()

        y = np.abs(y)
        x = np.abs(x)

        maxxy = np.maximum(x, y)
        minxy = np.minimum(x, y)

        r1 = (rad ** 2 - (maxxy + 0.5) ** 2)
        r2 = (rad ** 2 - (minxy - 0.5) ** 2)

        if (r1 > 0).all():
            warn_m1 = r1 ** 0.5
        else:
            warn_m1 = 0
        if (r2 > 0).all():
            warn_m2 = r2 ** 0.5
        else:
            warn_m2 = 0

        m1 = (rad ** 2 < (maxxy + 0.5) ** 2 + (minxy - 0.5) ** 2) * (minxy - 0.5) + (
                rad ** 2 >= (maxxy + 0.5) ** 2 + (minxy - 0.5) ** 2) * warn_m1
        m2 = (rad ** 2 > (maxxy - 0.5) ** 2 + (minxy + 0.5) ** 2) * (minxy + 0.5) + (
                rad ** 2 <= (maxxy - 0.5) ** 2 + (minxy + 0.5) ** 2) * warn_m2

        sgrid = (rad ** 2 * (0.5 * (np.arcsin(m2 / rad) - np.arcsin(m1 / rad)) +
                             0.25 * (np.sin(2 * np.arcsin(m2 / rad)) - np.sin(2 * np.arcsin(m1 / rad)))) - (
                         maxxy - 0.5) * (m2 - m1) + (m1 - minxy + 0.5)) * np.logical_or(
            np.logical_and((rad ** 2 < (maxxy + 0.5) ** 2 + (minxy + 0.5) ** 2),
                           (rad ** 2 > (maxxy - 0.5) ** 2 + (minxy - 0.5) ** 2)),
            np.logical_and(np.logical_and(minxy == 0, maxxy - 0.5 < rad), maxxy + 0.5 >= rad))

        sgrid = sgrid + ((maxxy + 0.5) ** 2 + (minxy + 0.5) ** 2 < rad ** 2)
        sgrid[crad, crad] = np.minimum(math.pi * rad ** 2, math.pi / 2)
        if (crad > 0) and (rad > crad - 0.5) and (rad ** 2 < (crad - 0.5) ** 2 + 0.25):
            m1 = np.sqrt(rad ** 2 - (crad - 0.5) ** 2)
            m1n = m1 / rad
            sg0 = 2 * (rad ** 2 * (0.5 * np.arcsin(m1n) + 0.25 * np.sin(2 * np.arcsin(m1n))) - m1 * (crad - 0.5))
            sgrid[2 * crad, crad] = sg0
            sgrid[crad, 2 * crad] = sg0
            sgrid[crad, 0] = sg0
            sgrid[0, crad] = sg0
            sgrid[2 * crad, crad] = sgrid[2 * crad, crad] - sg0
            sgrid[crad, 2 * crad] = sgrid[crad, 2 * crad] - sg0
            sgrid[crad, 2] = sgrid[crad, 2] - sg0
            sgrid[2, crad] = sgrid[2, crad + 1] - sg0

        sgrid[crad, crad] = np.minimum(sgrid[crad, crad], 1)
        h = sgrid / np.sum(sgrid)
        return h
    elif filter_type == 'motion':

        eps = 2.2204e-16
        length = max(1, p2)
        half_len = (length - 1) / 2.
        phi = (p3 % 180) / 180 * math.pi

        cosphi = math.cos(phi)
        sinphi = math.sin(phi)
        xsign = sign(cosphi)
        linewdt = 1

        sx = int(half_len * cosphi + linewdt * xsign - length * eps)
        sy = int(half_len * sinphi + linewdt - length * eps)
        x, y = np.mgrid[0:sx + (1 * xsign):xsign, 0:sy + 1]
        x = x.transpose()
        y = y.transpose()

        dist2line = (y * cosphi - x * sinphi)
        rad = (x ** 2 + y ** 2) ** 0.5

        lastpix = np.where(np.logical_and((rad >= half_len), (abs(dist2line) <= linewdt)))
        x2lastpix = half_len - np.abs((x[lastpix] + dist2line[lastpix] * sinphi) / cosphi);

        dist2line[lastpix] = np.sqrt(dist2line[lastpix] ** 2 + x2lastpix ** 2)
        dist2line = linewdt + eps - np.abs(dist2line)
        dist2line[dist2line < 0] = 0

        h = np.rot90(dist2line, 2)
        tmp_h = np.zeros((h.shape[0] * 2 - 1, h.shape[1] * 2 - 1))
        tmp_h[0:h.shape[0], 0:h.shape[1]] = h
        tmp_h[(h.shape[0]) - 1:, h.shape[1] - 1:] = dist2line
        h = tmp_h

        h /= np.sum(h) + eps * length * length

        if cosphi > 0:
            h = np.flipud(h)

        return h

    else:
        raise NotImplementedError(f"Filter type {filter_type} not implemented")


def filter2D(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """PyTorch version of cv2.filter2D
    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    img = img.float()
    k1 = kernel.size(-2)
    k2 = kernel.size(-1)

    b, c, h, w = img.size()
    if k1 % 2 == 1 or k2 % 2 == 1:
        img = F.pad(img, (k2 // 2, k2 // 2, k1 // 2, k1 // 2), mode='replicate')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k1, k2)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k1, k2).repeat(1, c, 1, 1).view(b * c, 1, k1, k2)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


def curves(xx: torch.Tensor, coef: float) -> torch.Tensor:
    if type(coef) == list:
        coef = [[0.3, 0.5, 0.7],
                [coef[0], 0.5, coef[1]]]
    else:
        coef = [[0.5], [coef]]

    x = np.array([0] + [p for p in coef[0]] + [1])
    y = np.array([0] + [p for p in coef[1]] + [1])

    cs = spline(x, y)

    yy = ppval(cs, xx)

    yy = torch.clamp(yy, 0, 1)

    return yy


def spline(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    dd = 1
    dx = np.diff(x)
    divdif = np.diff(y) / dx

    if n == 3:
        y[1:3] = divdif
        y[2] = np.diff(divdif.T).T / (x[2] - x[0])
        y[1] -= y[2] * dx[0]
        dlk = y[[2, 1, 0]].shape[0]
        l = x[[0, 2]].shape[0] - 1
        dl = np.prod(dd) * l
        k = np.fix(dlk / dl + 100 * 2.2204e-16)

        pp = (x[[0, 2]], y[[2, 1, 0]], l, int(k), dd)

    elif n > 3:
        b = np.zeros(n)
        b[1:n - 1] = 3 * (dx[1:n] * divdif[0:n - 2] + dx[0:n - 2] * divdif[1:n])

        x31 = x[2] - x[0]
        xn = x[n - 1] - x[n - 3]

        b[0] = ((dx[0] + 2 * x31) * dx[1] * divdif[0] + dx[0] ** 2 * divdif[1]) / x31
        b[n - 1] = (dx[n - 2] ** 2 * divdif[n - 3] + (2 * xn + dx[n - 2]) * dx[n - 3] * divdif[n - 2]) / xn;

        dxt = dx.T
        c = np.zeros((3, 5))
        c[0, :] = [x31] + list(dxt[0:n - 2]) + [0]
        c[1, :] = [dxt[1]] + list(2 * (dxt[1:n - 1] + dxt[0:n - 2])) + [dxt[n - 3]]
        c[2, :] = [0] + list(dxt[1:n - 1]) + [xn]

        c = scipy.sparse.dia_matrix((c, [-1, 0, 1]), shape=(5, 5))
        c = scipy.sparse.csc_matrix(c)
        ic = scipy.sparse.linalg.inv(c)
        s = b * ic

        n = x.shape[0]
        d = 1
        dxd = dx

        dzzdx = (divdif - s[0:n - 1]) / dxd
        dzdxdx = (s[1:n] - divdif) / dxd

        coefs = np.vstack(((dzdxdx - dzzdx) / dxd, 2 * dzzdx - dzdxdx, s[0:n - 1], y[0:n - 1])).T

        pp = (x, coefs, x.shape[0], x.shape[0], d)
    else:
        raise ValueError('x.shape[0] must be >= 3')

    return pp


def ppval(pp: np.ndarray, xx: torch.Tensor) -> torch.Tensor:
    lx = torch.numel(xx)
    xs = xx.reshape(1, lx)
    b, c, l, k, dd = pp
    b = torch.as_tensor(b, device=xx.device)
    ranges = b.clone()
    ranges[0] = -torch.inf
    ranges[-1] = torch.inf
    index = histc(xs, ranges)

    xs = xs - b[index]

    c = torch.as_tensor(c, device=xx.device)

    if len(c.shape) == 1:
        v = c[0]
        for i in range(1, k):
            v = xs * v + c[i]
    else:
        v = c[index, 0]

        for i in range(1, k - 1):
            v = xs * v + c[index, i]
    v = v.view(xx.shape)
    return v


def histc(x: torch.Tensor, binranges: torch.Tensor) -> torch.Tensor:
    indices = torch.bucketize(x, binranges)
    return torch.remainder(indices, len(binranges)) - 1


def imscatter(x: torch.Tensor, amount: float = 0.05, iterations=1) -> torch.Tensor:
    """
    Apply scatter distortion to the image.

    Args:
        x (torch.Tensor): Input image tensor with shape (B, C, H, W) or (C, H, W).
        amount (float): Scatter intensity.
        iterations (int): Number of scatter iterations.

    Returns:
        torch.Tensor: Scattered image tensor with shape (B, C, H, W) or (C, H, W).
    """
    if x.dim() == 4:  # Batched input (B, C, H, W)
        batch_size, channels, height, width = x.size()
    elif x.dim() == 3:  # Single image input (C, H, W)
        batch_size = 1
        channels, height, width = 1, *x.size()[-2:]
        x = x.unsqueeze(0)  # Add batch dimension
    else:
        raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D tensor.")

    device = x.device
    y = x.clone()

    for _ in range(iterations):
        shift_map_y = torch.randn((batch_size, 1, height, width), device=device) * amount
        shift_map_x = torch.randn((batch_size, 1, height, width), device=device) * amount

        # Pass all required arguments to generate_scatter_grid
        grid = generate_scatter_grid(batch_size, height, width, shift_map_y, shift_map_x, device)
        y = F.grid_sample(y, grid, mode='bilinear', padding_mode='reflection', align_corners=False)

    return y.squeeze(0) if batch_size == 1 else y


def bilinear_interpolate_torch(im: torch.Tensor, x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    dtype_long = torch.LongTensor

    x0 = torch.floor(x).type(dtype_long).to(im.device)
    x1 = x0 + 1

    y0 = torch.floor(y).type(dtype_long).to(im.device)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    R1 = Ia * (x1 - x) / (x1 - x0 + eps) + Ic * (x - x0) / (x1 - x0 + eps)
    R2 = Ib * (x1 - x) / (x1 - x0 + eps) + Id * (x - x0) / (x1 - x0 + eps)
    P = R1 * (y1 - y) / (y1 - y0 + eps) + R2 * (y - y0) / (y1 - y0 + eps)
    return P


import random

def apply_random_distortions(images, shared_distortion=None, return_info=False):
    """
    주어진 이미지에 랜덤 왜곡을 적용합니다.
    
    Args:
        images (torch.Tensor): 입력 이미지 텐서 [batch_size, C, H, W].
        shared_distortion (str, optional): 동일한 왜곡을 적용할 경우 지정.
        return_info (bool): 적용된 왜곡 정보를 반환할지 여부.
    
    Returns:
        torch.Tensor: 왜곡된 이미지.
        list (optional): 적용된 왜곡의 목록 (return_info가 True일 경우).
    """
    distortions = ["blur", "noise", "color_shift", "jpeg_compression"]
    applied_distortions = []

    for i in range(images.size(0)):  # 배치 크기만큼 반복
        distortion = shared_distortion if shared_distortion else random.choice(distortions)
        applied_distortions.append(distortion)

        if distortion == "blur":
            images[i] = images[i].unsqueeze(0).float()  # Blur 적용 코드 추가
        elif distortion == "noise":
            images[i] += torch.randn_like(images[i]) * 0.1  # Noise 추가
        elif distortion == "color_shift":
            images[i] = images[i] * random.uniform(0.8, 1.2)  # 색 변화 추가
        elif distortion == "jpeg_compression":
            # JPEG 압축 왜곡 적용 코드
            pass

    if return_info:
        return images, applied_distortions
    return images




def generate_hard_negatives(img, scale_factor=0.5):
    """
    Generate hard negatives by downscaling the input image.

    Args:
        img (torch.Tensor): Input image tensor of shape [batch_size, num_crops, C, H, W].
        scale_factor (float): Scaling factor for downscaling.

    Returns:
        torch.Tensor: Downscaled image tensor of shape [batch_size, num_crops, C, new_H, new_W].
    """
    batch_size, num_crops, C, H, W = img.size()
    new_H, new_W = int(H * scale_factor), int(W * scale_factor)

    # View 전에 contiguous() 호출
    img = img.contiguous().view(-1, C, H, W)  # Flatten (batch_size * num_crops, C, H, W)

    # Resize images
    downscaled_imgs = nn.functional.interpolate(img, size=(new_H, new_W), mode='bilinear', align_corners=False)

    # Reshape back to original dimensions
    downscaled_imgs = downscaled_imgs.view(batch_size, num_crops, C, new_H, new_W)

    return downscaled_imgs




def imscatter(img: torch.Tensor, amount: float = 0.05, iterations: int = 1) -> torch.Tensor:
    """
    Apply scatter distortion to the image.
    """
    if img.dim() == 3:  # 3D 입력일 경우 (C, H, W)
        img = img.unsqueeze(0)  # 배치 차원 추가
    if img.dim() != 4:  # 4D 텐서가 아닌 경우 오류 처리
        raise ValueError(f"Expected 4D input (B, C, H, W), but got {img.dim()}D tensor.")

    device = img.device
    batch_size, _, height, width = img.size()
    y = img.clone()

    for _ in range(iterations):
        sy = torch.randn((batch_size, 1, height, width), device=device) * amount
        sx = torch.randn((batch_size, 1, height, width), device=device) * amount
        grid = generate_scatter_grid(batch_size, height, width, sy, sx, device)
        y = F.grid_sample(y, grid, mode='bilinear', padding_mode='reflection', align_corners=False)

    return y.squeeze(0) if y.size(0) == 1 else y


def add_gaussian_noise(img: torch.Tensor, std: float = 0.1) -> torch.Tensor:
    """
    Add Gaussian noise to the image.
    
    Args:
        img (torch.Tensor): Input image tensor with shape (C, H, W).
        std (float): Standard deviation of the noise.
    
    Returns:
        torch.Tensor: Noisy image tensor with shape (C, H, W).
    """
    noise = torch.randn_like(img) * std
    return torch.clamp(img + noise, 0, 1)


def apply_motion_blur(img: torch.Tensor, kernel_size: int = 5, angle: float = 45.0) -> torch.Tensor:
    """
    Apply motion blur to the image using a predefined kernel.

    Args:
        img (torch.Tensor): Input image tensor of shape (B, C, H, W) or (B, num_crops, C, H, W).
        kernel_size (int): Size of the motion blur kernel.
        angle (float): Angle of motion blur in degrees.

    Returns:
        torch.Tensor: Blurred image tensor with the same shape as input.
    """
    if img.dim() == 4:  # 4D input: [B, C, H, W]
        batch_size, channels, height, width = img.size()
    elif img.dim() == 3:  # 3D input: [C, H, W]
        batch_size = 1
        channels, height, width = img.shape
        img = img.unsqueeze(0)  # Add batch dimension
    else:
        raise ValueError(f"Expected 3D or 4D input, got {img.dim()}D tensor.")

    # Create motion blur kernel
    kernel = create_motion_blur_kernel(kernel_size, angle)
    kernel = torch.from_numpy(kernel).float().to(img.device)
    kernel = kernel.expand(channels, 1, kernel_size, kernel_size)  # Match kernel to number of channels

    # Apply motion blur
    blurred = F.conv2d(img, kernel, padding=kernel_size // 2, groups=channels)

    if batch_size == 1:
        blurred = blurred.squeeze(0)  # Remove batch dimension if added

    return blurred


def create_motion_blur_kernel(kernel_size: int, angle: float) -> np.ndarray:
    """
    Create a motion blur kernel.
    
    Args:
        kernel_size (int): Size of the kernel.
        angle (float): Angle of the motion blur in degrees.
    
    Returns:
        np.ndarray: Motion blur kernel.
    """
    kernel = np.zeros((kernel_size, kernel_size))
    mid = kernel_size // 2
    for i in range(kernel_size):
        offset = round(mid - i * math.tan(math.radians(angle)))
        if 0 <= mid + offset < kernel_size:
            kernel[i, mid + offset] = 1
    return kernel / kernel.sum()


def generate_scatter_grid(batch_size: int, height: int, width: int, sy: torch.Tensor, sx: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Generate scatter grid for imscatter.

    Args:
        batch_size (int): Batch size of the input images.
        height (int): Height of the input images.
        width (int): Width of the input images.
        sy (torch.Tensor): Scatter map for y-axis.
        sx (torch.Tensor): Scatter map for x-axis.
        device (torch.device): Device to generate the grid.

    Returns:
        torch.Tensor: Grid tensor of shape (B, H, W, 2).
    """
    xx, yy = torch.meshgrid(
        torch.arange(0, width, device=device, dtype=torch.float32),
        torch.arange(0, height, device=device, dtype=torch.float32),
        indexing="xy"
    )
    xx = xx.unsqueeze(0).expand(batch_size, -1, -1)
    yy = yy.unsqueeze(0).expand(batch_size, -1, -1)

    grid = torch.stack((xx + sx.squeeze(1), yy + sy.squeeze(1)), dim=-1)
    return grid

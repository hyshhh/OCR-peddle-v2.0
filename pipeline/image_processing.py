"""
图像预处理算法模块

提供小波变换去噪、卷积锐化等方法，供 locator 在 OCR 前对 crop 做预处理。
每个方法接收 BGR 图像，返回处理后的 BGR 图像。
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def wavelet_denoise(
    crop: np.ndarray,
    method: str = "haar",
    level: int = 1,
    threshold: float = 10.0,
) -> np.ndarray:
    """
    小波变换去噪。

    Args:
        crop: BGR 图像。
        method: 小波基函数，如 "haar", "db1", "db2"。
        level: 分解层数。
        threshold: 去噪阈值（越大去噪越强，细节丢失越多）。
    """
    try:
        import pywt

        h, w = crop.shape[:2]

        if crop.ndim == 3:
            # 逐通道处理
            result = crop.copy()
            for c in range(3):
                channel = crop[:, :, c].astype(np.float64)
                coeffs = pywt.wavedec2(channel, method, level=level)
                # 对高频系数做软阈值
                denoised_coeffs = [coeffs[0]]
                for detail in coeffs[1:]:
                    denoised_detail = tuple(
                        pywt.threshold(d, threshold, mode="soft") for d in detail
                    )
                    denoised_coeffs.append(denoised_detail)
                denoised = pywt.waverec2(denoised_coeffs, method)
                # 裁剪到原始尺寸（小波重建可能产生微小尺寸差异）
                denoised = denoised[:h, :w]
                result[:, :, c] = np.clip(denoised, 0, 255).astype(np.uint8)
            return result
        else:
            coeffs = pywt.wavedec2(crop, method, level=level)
            denoised_coeffs = [coeffs[0]]
            for detail in coeffs[1:]:
                denoised_detail = tuple(
                    pywt.threshold(d, threshold, mode="soft") for d in detail
                )
                denoised_coeffs.append(denoised_detail)
            denoised = pywt.waverec2(denoised_coeffs, method)
            denoised = denoised[:h, :w]
            return np.clip(denoised, 0, 255).astype(np.uint8)
    except ImportError:
        logger.warning("pywt 未安装，跳过去噪。安装: pip install PyWavelets")
        return crop
    except Exception as e:
        logger.warning("小波去噪异常: %s", e)
        return crop


def convolution_sharpen(
    crop: np.ndarray,
    strength: float = 1.5,
    kernel_size: int = 3,
) -> np.ndarray:
    """
    卷积锐化。

    使用 Unsharp Mask 原理：先模糊再与原图做差，增强边缘。
    核心公式：输出 = 原图 + strength × (原图 - 模糊图)

    Args:
        crop: BGR 图像。
        strength: 锐化强度（0=不锐化，1.0=标准锐化，越大越锐）。
        kernel_size: 高斯模糊核大小（必须为奇数）。
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    # 高斯模糊
    blurred = cv2.GaussianBlur(crop, (kernel_size, kernel_size), 0)

    # Unsharp Mask：输出 = 原图 + strength × (原图 - 模糊图)
    # = (1 + strength) × 原图 - strength × 模糊图
    sharpened = cv2.addWeighted(crop, 1.0 + strength, blurred, -strength, 0)

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def clahe_enhance(
    crop: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: int = 8,
) -> np.ndarray:
    """
    CLAHE 对比度增强（自适应直方图均衡化）。

    适合光照不均匀的场景。

    Args:
        crop: BGR 图像。
        clip_limit: 对比度限制。
        grid_size: 网格大小。
    """
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def gaussian_blur_denoise(
    crop: np.ndarray,
    ksize: int = 3,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    高斯模糊去噪。

    Args:
        crop: BGR 图像。
        ksize: 核大小（必须为奇数）。
        sigma: 高斯标准差。
    """
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(crop, (ksize, ksize), sigma)


def median_blur_denoise(
    crop: np.ndarray,
    ksize: int = 3,
) -> np.ndarray:
    """
    中值滤波去噪（对椒盐噪声效果好）。

    Args:
        crop: BGR 图像。
        ksize: 核大小（必须为奇数）。
    """
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(crop, ksize)


def bilateral_denoise(
    crop: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """
    双边滤波去噪（保边去噪）。

    Args:
        crop: BGR 图像。
        d: 像素邻域直径。
        sigma_color: 颜色空间标准差。
        sigma_space: 坐标空间标准差。
    """
    return cv2.bilateralFilter(crop, d, sigma_color, sigma_space)


def unsharp_mask(
    crop: np.ndarray,
    sigma: float = 1.0,
    strength: float = 1.5,
) -> np.ndarray:
    """
    USM 锐化（Unsharp Mask）。

    先模糊再与原图做差，增强边缘。

    Args:
        crop: BGR 图像。
        sigma: 高斯模糊标准差（越大锐化范围越广）。
        strength: 锐化强度。
    """
    blurred = cv2.GaussianBlur(crop, (0, 0), sigma)
    sharpened = cv2.addWeighted(crop, 1 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def grayscale_convert(crop: np.ndarray) -> np.ndarray:
    """转灰度图（某些 OCR 对灰度效果更好）。"""
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return crop


def adaptive_threshold(crop: np.ndarray, block_size: int = 11, c: int = 2) -> np.ndarray:
    """
    自适应阈值二值化。

    适合文字与背景对比度低的场景。

    Args:
        crop: BGR 图像。
        block_size: 邻域大小（必须为奇数）。
        c: 常数偏移。
    """
    if block_size % 2 == 0:
        block_size += 1
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block_size, c)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


# ── 算法注册表 ──────────────────────────────────

ALGORITHMS: dict[str, callable] = {
    "wavelet_denoise": wavelet_denoise,
    "convolution_sharpen": convolution_sharpen,
    "clahe_enhance": clahe_enhance,
    "gaussian_blur": gaussian_blur_denoise,
    "median_blur": median_blur_denoise,
    "bilateral_denoise": bilateral_denoise,
    "unsharp_mask": unsharp_mask,
    "grayscale": grayscale_convert,
    "adaptive_threshold": adaptive_threshold,
}


def build_pipeline(steps: list[dict]) -> list[tuple[callable, dict]]:
    """
    根据配置构建预处理流水线。

    Args:
        steps: 支持两种格式：
            - 简单格式：["wavelet_denoise", "convolution_sharpen"]
            - 详细格式：[{"name": "wavelet_denoise", "params": {"level": 2}}]

    Returns:
        [(func, params), ...]
    """
    pipeline = []
    for step in steps:
        if isinstance(step, str):
            name = step
            params = {}
        elif isinstance(step, dict):
            name = step.get("name", "")
            params = step.get("params", {})
        else:
            continue

        if name not in ALGORITHMS:
            logger.warning("未知的图像处理算法: %s，跳过", name)
            continue
        pipeline.append((ALGORITHMS[name], params))
    return pipeline


def apply_pipeline(crop: np.ndarray, pipeline: list[tuple[callable, dict]]) -> np.ndarray:
    """
    按顺序执行预处理流水线。

    Args:
        crop: BGR 图像。
        pipeline: [(func, params), ...]

    Returns:
        处理后的图像。
    """
    for func, params in pipeline:
        try:
            crop = func(crop, **params)
        except Exception as e:
            logger.warning("%s 执行异常: %s", func.__name__, e)
    return crop

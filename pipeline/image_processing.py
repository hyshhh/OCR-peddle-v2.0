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


def gamma_correction(crop: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Gamma 校正（亮度调整）。

    gamma < 1: 变亮（暗区细节增强）
    gamma > 1: 变暗（亮区细节增强）
    gamma = 1: 不变

    Args:
        crop: BGR 图像。
        gamma: Gamma 值。
    """
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255 for i in range(256)
    ]).astype(np.uint8)
    return cv2.LUT(crop, table)


def contrast_stretching(crop: np.ndarray, low_percent: float = 2.0, high_percent: float = 98.0) -> np.ndarray:
    """
    对比度拉伸（线性归一化）。

    将像素值从 [low, high] 拉伸到 [0, 255]。

    Args:
        crop: BGR 图像。
        low_percent: 低百分位（截断下限）。
        high_percent: 高百分位（截断上限）。
    """
    if crop.ndim == 3:
        # 逐通道处理
        result = crop.copy()
        for c in range(3):
            channel = crop[:, :, c]
            low = np.percentile(channel, low_percent)
            high = np.percentile(channel, high_percent)
            if high - low < 1:
                continue
            result[:, :, c] = np.clip(
                (channel - low) / (high - low) * 255, 0, 255
            ).astype(np.uint8)
        return result
    else:
        low = np.percentile(crop, low_percent)
        high = np.percentile(crop, high_percent)
        if high - low < 1:
            return crop
        return np.clip(
            (crop - low) / (high - low) * 255, 0, 255
        ).astype(np.uint8)


def retinex_enhance(crop: np.ndarray, sigma: int = 30) -> np.ndarray:
    """
    Retinex 增强（单尺度 SSR）。

    分离光照和反射，去除光照影响。适合去雾、去阴影。

    Args:
        crop: BGR 图像。
        sigma: 高斯模糊标准差（越大去除的光照范围越广）。
    """
    if crop.ndim == 3:
        # 逐通道处理
        result = crop.copy()
        for c in range(3):
            channel = crop[:, :, c].astype(np.float64) + 1.0  # +1 避免 log(0)
            blur = cv2.GaussianBlur(channel, (0, 0), sigma)
            retinex = np.log1p(channel) - np.log1p(blur)
            # 归一化到 [0, 255]
            retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min() + 1e-6) * 255
            result[:, :, c] = np.clip(retinex, 0, 255).astype(np.uint8)
        return result
    else:
        channel = crop.astype(np.float64) + 1.0
        blur = cv2.GaussianBlur(channel, (0, 0), sigma)
        retinex = np.log1p(channel) - np.log1p(blur)
        retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min() + 1e-6) * 255
        return np.clip(retinex, 0, 255).astype(np.uint8)


def edge_enhance(crop: np.ndarray, strength: float = 0.7) -> np.ndarray:
    """
    拉普拉斯边缘增强。

    Args:
        crop: BGR 图像。
        strength: 增强强度（越大边缘越明显）。
    """
    laplacian = cv2.Laplacian(crop, cv2.CV_64F)
    enhanced = crop.astype(np.float64) - strength * laplacian
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def tophat_enhance(crop: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """
    顶帽变换（形态学增强）。

    提取亮细节，适合文字增强。

    Args:
        crop: BGR 图像。
        kernel_size: 结构元素大小。
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if crop.ndim == 3:
        # 逐通道处理
        result = crop.copy()
        for c in range(3):
            tophat = cv2.morphologyEx(crop[:, :, c], cv2.MORPH_TOPHAT, kernel)
            result[:, :, c] = cv2.add(crop[:, :, c], tophat)
        return result
    else:
        tophat = cv2.morphologyEx(crop, cv2.MORPH_TOPHAT, kernel)
        return cv2.add(crop, tophat)


def shadow_adjust(crop: np.ndarray, shadow_amount: float = 50.0) -> np.ndarray:
    """
    阴影调节 — 提亮暗区，保留亮区。

    使用 LAB 色彩空间，只调整 L 通道的暗区部分。

    Args:
        crop: BGR 图像。
        shadow_amount: 提亮程度 (0-100)。0=不调整，100=最大程度提亮。
    """
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 归一化到 [0, 1]
    l_norm = l.astype(np.float64) / 255.0

    # 只提亮暗区（像素值 < 0.5 的部分）
    # 使用 sigmoid 曲线：暗区提亮多，亮区几乎不变
    shadow_factor = shadow_amount / 100.0
    # 对暗区应用提亮：越暗提亮越多
    mask = np.clip(1.0 - l_norm, 0, 1)  # 暗区 mask 值大
    adjustment = mask * shadow_factor * 0.5  # 最大提亮 0.5（约 128 级）
    l_enhanced = np.clip(l_norm + adjustment, 0, 1)

    l_result = (l_enhanced * 255).astype(np.uint8)
    enhanced = cv2.merge([l_result, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def exposure_adjust(crop: np.ndarray, exposure: float = 0.0) -> np.ndarray:
    """
    曝光调节 — 整体亮度调整。

    Args:
        crop: BGR 图像。
        exposure: 曝光值 (-100 到 100)。
                  负值变暗，正值变亮，0=不变。
    """
    # 将 -100~100 映射到实际调整值
    # 100 对应 +1.0（最大提亮），-100 对应 -1.0（最大变暗）
    factor = exposure / 100.0

    if factor >= 0:
        # 提亮：向 255 靠近
        result = crop.astype(np.float64) + (255 - crop.astype(np.float64)) * factor
    else:
        # 变暗：向 0 靠近
        result = crop.astype(np.float64) * (1 + factor)

    return np.clip(result, 0, 255).astype(np.uint8)


def contrast_adjust(crop: np.ndarray, contrast: float = 0.0) -> np.ndarray:
    """
    对比度调节 — 线性对比度调整。

    Args:
        crop: BGR 图像。
        contrast: 对比度值 (-100 到 100)。
                  负值降低对比度，正值增强对比度，0=不变。
    """
    # 将 -100~100 映射到系数
    # 100 -> 系数 2.0（对比度翻倍），-100 -> 系数 0.0（全灰）
    factor = (contrast + 100) / 100.0

    # 计算均值
    mean = np.mean(crop)

    # 线性对比度调整：output = mean + factor * (input - mean)
    result = mean + factor * (crop.astype(np.float64) - mean)

    return np.clip(result, 0, 255).astype(np.uint8)


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
    "gamma_correction": gamma_correction,
    "contrast_stretching": contrast_stretching,
    "retinex": retinex_enhance,
    "edge_enhance": edge_enhance,
    "tophat": tophat_enhance,
    "shadow_adjust": shadow_adjust,
    "exposure_adjust": exposure_adjust,
    "contrast_adjust": contrast_adjust,
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

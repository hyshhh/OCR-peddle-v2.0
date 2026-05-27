"""
HullNumberLocator — 基于 PaddleOCR TextDetection 的弦号定位

在 YOLO crop 上运行文字检测，返回检测到的文字区域框（已转换为原始帧坐标）。

可选 UVDoc 文字矫正预处理：crop → UVDoc 矫正 → TextDetection 检测。
支持保存矫正前后的 crop 图像用于调试。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 我们自己的参数（从 config 中提取后不传给 PaddleOCR）
_OWN_KEYS = frozenset({
    "enabled", "score_threshold", "min_area",
    "unwarp_enabled", "unwarp_model_name", "unwarp_model_dir", "unwarp_device",
    "save_crops", "crop_save_dir", "crop_save_interval",
    "preprocess_steps",
})


@dataclass
class TextRegion:
    """单个文字检测区域。"""
    bbox_frame: tuple[int, int, int, int]  # (x1, y1, x2, y2) 在原始帧中的坐标
    confidence: float = 0.0
    polygon: np.ndarray | None = None      # 原始多边形点（帧坐标）


class HullNumberLocator:
    """
    弦号定位器 — PaddleOCR TextDetection 检测文字区域。

    流程（可选 UVDoc 矫正）：
    1. crop → UVDoc 矫正（若启用）→ TextDetection 检测
    2. 后处理过滤（score_threshold / min_area）
    3. 坐标转换：crop 坐标 → 原始帧坐标
    """

    def __init__(self, locator_cfg: dict):
        self._score_threshold: float = locator_cfg.get("score_threshold", 0.5)
        self._min_area: int = locator_cfg.get("min_area", 100)

        # UVDoc 矫正配置
        self._unwarp_enabled: bool = bool(locator_cfg.get("unwarp_enabled", False))
        self._unwarp_model_name: str | None = locator_cfg.get("unwarp_model_name")
        self._unwarp_model_dir: str | None = locator_cfg.get("unwarp_model_dir")
        self._unwarp_device: str | None = locator_cfg.get("unwarp_device")

        # Crop 保存配置
        self._save_crops: bool = bool(locator_cfg.get("save_crops", False))
        self._crop_save_dir: str = locator_cfg.get("crop_save_dir", "./crops")
        self._crop_save_interval: float = locator_cfg.get("crop_save_interval", 10)
        self._last_crop_save: float = 0.0

        # 图像预处理流水线
        from pipeline.image_processing import build_pipeline
        self._preprocess_pipeline = build_pipeline(locator_cfg.get("preprocess_steps", []))

        # TextDetection 参数（去掉所有我们自己的 key）
        self._paddle_kwargs: dict = {
            k: v for k, v in locator_cfg.items()
            if k not in _OWN_KEYS and v is not None
        }

        self._det_model = None
        self._unwarp_model = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        try:
            from paddleocr import TextDetection

            logger.info("TextDetection 参数: %s", self._paddle_kwargs)
            self._det_model = TextDetection(**self._paddle_kwargs)

            if self._unwarp_enabled:
                from paddleocr import TextImageUnwarping
                unwarp_kwargs = {}
                if self._unwarp_model_name:
                    unwarp_kwargs["model_name"] = self._unwarp_model_name
                if self._unwarp_model_dir:
                    unwarp_kwargs["model_dir"] = self._unwarp_model_dir
                if self._unwarp_device:
                    unwarp_kwargs["device"] = self._unwarp_device
                self._unwarp_model = TextImageUnwarping(**unwarp_kwargs)
                logger.info("UVDoc 矫正模型加载成功")

            self._initialized = True
            logger.info("后处理: score_threshold=%.3f, min_area=%d, unwarp=%s, save_crops=%s",
                        self._score_threshold, self._min_area,
                        self._unwarp_enabled, self._save_crops)
        except ImportError:
            logger.error(
                "PaddleOCR 未安装。请安装: pip install paddleocr>=3.5 paddlepaddle-gpu>=3.3"
            )
            raise
        except Exception as e:
            logger.error("PaddleOCR 模型加载失败: %s", e)
            raise

    def _unwarp_crop(self, crop: np.ndarray) -> np.ndarray:
        """UVDoc 矫正：BGR → RGB → 矫正 → BGR。"""
        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            output = self._unwarp_model.predict(rgb)
            result = output[0]
            if isinstance(result, dict):
                corrected = result.get("doctr_img", result.get("res", None))
            else:
                corrected = getattr(result, "doctr_img", None)
            if corrected is not None:
                if isinstance(corrected, np.ndarray):
                    if corrected.ndim == 3 and corrected.shape[2] == 3:
                        return cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)
                    return corrected
            logger.debug("UVDoc 返回空结果，使用原图")
            return crop
        except Exception as e:
            logger.warning("UVDoc 矫正异常，使用原图: %s", e)
            return crop

    def _maybe_save_crops(self, crop_raw: np.ndarray, crop_unwarped: np.ndarray, track_id: int) -> None:
        """每隔 crop_save_interval 秒保存矫正前后的 crop。"""
        if not self._save_crops:
            return

        now = time.monotonic()
        if now - self._last_crop_save < self._crop_save_interval:
            return
        self._last_crop_save = now

        try:
            save_dir = Path(self._crop_save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            ts = int(time.time())
            prefix = save_dir / f"track{track_id}_{ts}"
            cv2.imwrite(str(prefix) + "_before.jpg", crop_raw)
            cv2.imwrite(str(prefix) + "_after.jpg", crop_unwarped)
            logger.info("Crop 已保存: %s_before.jpg / %s_after.jpg", prefix, prefix)
        except Exception as e:
            logger.warning("保存 crop 失败: %s", e)

    def locate(
        self,
        crop: np.ndarray,
        offset_x: int = 0,
        offset_y: int = 0,
        track_id: int = 0,
    ) -> list[TextRegion]:
        self._ensure_initialized()

        if crop is None or crop.size == 0:
            return []

        try:
            # UVDoc 矫正（可选）
            if self._unwarp_enabled:
                det_input = self._unwarp_crop(crop)
                self._maybe_save_crops(crop, det_input, track_id)
            else:
                det_input = crop

            # 图像预处理（小波去噪、卷积锐化等）
            if self._preprocess_pipeline:
                from pipeline.image_processing import apply_pipeline
                det_input = apply_pipeline(det_input, self._preprocess_pipeline)

            output = self._det_model.predict(det_input)

            regions: list[TextRegion] = []

            for res in output:
                if isinstance(res, dict):
                    boxes = res.get('dt_polys', None)
                    scores = res.get('dt_scores', None)
                elif hasattr(res, 'dt_polys'):
                    boxes = res.dt_polys
                    scores = getattr(res, 'dt_scores', None)
                else:
                    continue

                if boxes is None:
                    continue

                for idx, box in enumerate(boxes):
                    box_array = np.array(box, dtype=np.int32)

                    score = 0.0
                    if scores is not None and idx < len(scores):
                        score = float(scores[idx])

                    if score < self._score_threshold:
                        continue

                    area = cv2.contourArea(box_array)
                    if area < self._min_area:
                        continue

                    # UVDoc 矫正后图像尺寸可能变化，需要缩放坐标回原始 crop
                    if self._unwarp_enabled and det_input.shape[:2] != crop.shape[:2]:
                        h_scale = crop.shape[1] / det_input.shape[1]
                        v_scale = crop.shape[0] / det_input.shape[0]
                        box_array[:, 0] = (box_array[:, 0] * h_scale).astype(np.int32)
                        box_array[:, 1] = (box_array[:, 1] * v_scale).astype(np.int32)

                    # 坐标转换：crop 坐标 → 帧坐标
                    box_frame = box_array.copy()
                    box_frame[:, 0] += offset_x
                    box_frame[:, 1] += offset_y

                    x_coords = box_frame[:, 0]
                    y_coords = box_frame[:, 1]
                    x1 = int(np.min(x_coords))
                    y1 = int(np.min(y_coords))
                    x2 = int(np.max(x_coords))
                    y2 = int(np.max(y_coords))

                    regions.append(TextRegion(
                        bbox_frame=(x1, y1, x2, y2),
                        confidence=score,
                        polygon=box_frame,
                    ))

            return regions

        except Exception as e:
            logger.warning("弦号定位异常: %s", e)
            return []

    def cleanup(self) -> None:
        self._det_model = None
        self._unwarp_model = None
        self._initialized = False

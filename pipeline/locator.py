"""
HullNumberLocator — 基于 PaddleOCR PaddleOCR 的弦号定位

流程：crop → PaddleOCR(检测+识别) → 返回文字区域+OCR文本
支持保存各阶段 crop 用于调试。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 我们自己的参数（不传给 PaddleOCR）
_OWN_KEYS = frozenset({
    "enabled", "score_threshold", "min_area",
    "save_crops", "crop_save_dir", "crop_save_interval",
    "use_doc_orientation_classify", "use_doc_unwarping", "use_textline_orientation",
})


@dataclass
class TextRegion:
    """单个文字检测区域。"""
    bbox_frame: tuple[int, int, int, int]  # (x1, y1, x2, y2) 在原始帧中的坐标
    confidence: float = 0.0
    polygon: np.ndarray | None = None
    ocr_text: str = ""  # OCR 识别到的文字


class HullNumberLocator:
    """
    弦号定位器 — PaddleOCR 三步流水线。

    流程：
    1. crop → PaddleOCR（检测 + 识别）
    2. 返回文字区域坐标 + OCR 文本
    """

    def __init__(self, locator_cfg: dict):
        self._score_threshold: float = locator_cfg.get("score_threshold", 0.5)
        self._min_area: int = locator_cfg.get("min_area", 100)

        # Crop 保存
        self._save_crops: bool = bool(locator_cfg.get("save_crops", False))
        self._crop_save_dir: str = locator_cfg.get("crop_save_dir", "./crops")
        self._crop_save_interval: float = locator_cfg.get("crop_save_interval", 10)
        self._last_crop_save: float = 0.0

        # PaddleOCR 功能开关（用于判断是否需要单独跑矫正保存）
        self._use_doc_unwarping: bool = bool(locator_cfg.get("use_doc_unwarping", False))
        self._use_textline_orientation: bool = bool(locator_cfg.get("use_textline_orientation", False))

        # PaddleOCR 参数（过滤掉我们自己的参数）
        self._paddle_kwargs: dict = {
            k: v for k, v in locator_cfg.items()
            if k not in _OWN_KEYS and v is not None
        }

        self._ocr_model = None
        self._unwarp_model = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        try:
            from paddleocr import PaddleOCR

            # PaddleOCR 默认关闭文档预处理，由我们自己的参数控制
            kwargs = {
                "use_doc_orientation_classify": bool(
                    self._paddle_kwargs.pop("use_doc_orientation_classify", False)
                ),
                "use_doc_unwarping": bool(
                    self._paddle_kwargs.pop("use_doc_unwarping", False)
                ),
                "use_textline_orientation": bool(
                    self._paddle_kwargs.pop("use_textline_orientation", False)
                ),
                "engine": self._paddle_kwargs.pop("engine", "paddle"),
            }
            # 合并剩余的 paddle 参数（limit_side_len, thresh 等）
            kwargs.update(self._paddle_kwargs)

            logger.info("PaddleOCR 参数: %s", kwargs)
            self._ocr_model = PaddleOCR(**kwargs)

            # 如果开启了矫正且需要保存 crop，单独加载 TextImageUnwarping 用于获取矫正后的图
            if self._use_doc_unwarping and self._save_crops:
                from paddleocr import TextImageUnwarping
                self._unwarp_model = TextImageUnwarping()
                logger.info("TextImageUnwarping 加载成功（用于 crop 保存）")

            self._initialized = True
            logger.info("定位流水线初始化完成: save_crops=%s", self._save_crops)
        except ImportError:
            logger.error(
                "PaddleOCR 未安装。请安装: pip install paddleocr>=3.5 paddlepaddle-gpu>=3.3"
            )
            raise
        except Exception as e:
            logger.error("PaddleOCR 模型加载失败: %s", e)
            raise

    def _maybe_save_crops(self, crop_raw: np.ndarray, track_id: int) -> None:
        """每隔 crop_save_interval 秒保存 crop（矫正前）。"""
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
            logger.info("Crop 已保存: %s_before.jpg", prefix)
        except Exception as e:
            logger.warning("保存 crop 失败: %s", e)

    def _maybe_save_after_crop(self, crop_raw: np.ndarray, track_id: int) -> None:
        """保存矫正后的 crop（每隔 crop_save_interval 秒）。"""
        if not self._save_crops or self._unwarp_model is None:
            return

        try:
            rgb = cv2.cvtColor(crop_raw, cv2.COLOR_BGR2RGB)
            output = self._unwarp_model.predict(rgb)
            result = output[0]
            if isinstance(result, dict):
                corrected = result.get("doctr_img", result.get("res", None))
            else:
                corrected = getattr(result, "doctr_img", None)

            if corrected is not None and isinstance(corrected, np.ndarray):
                if corrected.ndim == 3 and corrected.shape[2] == 3:
                    corrected = cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)

                save_dir = Path(self._crop_save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                ts = int(time.time())
                prefix = save_dir / f"track{track_id}_{ts}"

                cv2.imwrite(str(prefix) + "_after.jpg", corrected)
                logger.info("Crop 已保存: %s_after.jpg", prefix)
        except Exception as e:
            logger.warning("保存矫正后 crop 失败: %s", e)

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
            # 保存原始 crop（矫正前）
            self._maybe_save_crops(crop, track_id)

            # PaddleOCR 检测 + 识别
            output = self._ocr_model.predict(crop)

            # 保存矫正后的 crop（如果开启了 UVDoc 矫正）
            if self._use_doc_unwarping and self._unwarp_model is not None:
                self._maybe_save_after_crop(crop, track_id)

            regions: list[TextRegion] = []

            for res in output:
                if not isinstance(res, dict):
                    continue

                # 提取检测结果
                rec_texts = res.get("rec_texts", [])
                rec_scores = res.get("rec_scores", [])
                rec_polys = res.get("rec_polys", None)
                dt_polys = res.get("dt_polys", None)

                # 使用 rec_polys（识别结果对应的多边形）作为主要框
                # 如果没有 rec_polys 则用 dt_polys
                boxes = rec_polys if rec_polys is not None else dt_polys

                if boxes is None:
                    continue

                for idx, box in enumerate(boxes):
                    box_array = np.array(box, dtype=np.int32)

                    # 获取置信度
                    score = 0.0
                    if rec_scores is not None and idx < len(rec_scores):
                        score = float(rec_scores[idx])

                    # 获取 OCR 文字
                    ocr_text = ""
                    if rec_texts is not None and idx < len(rec_texts):
                        ocr_text = str(rec_texts[idx])

                    # 置信度过滤
                    if score < self._score_threshold:
                        continue

                    # 面积过滤
                    area = cv2.contourArea(box_array)
                    if area < self._min_area:
                        continue

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
                        ocr_text=ocr_text,
                    ))

            return regions

        except Exception as e:
            logger.warning("弦号定位异常: %s", e)
            return []

    def cleanup(self) -> None:
        self._ocr_model = None
        self._initialized = False

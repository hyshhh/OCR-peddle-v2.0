"""
HullNumberLocator — 弦号定位器

支持两种检测器：
  - PaddleOCR TextDetection：文字检测，适合通用 OCR 场景
  - YOLO：目标检测，适合训练好的弦号检测模型

可选图像预处理流水线。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 我们自己的参数（从 config 中提取后不传给检测器）
_OWN_KEYS = frozenset({
    "enabled", "detector_type",
    "score_threshold", "min_area",
    "save_crops", "crop_save_dir", "crop_save_interval",
    "preprocess_steps",
    "yolo_model", "yolo_model_name", "yolo_conf", "yolo_iou", "yolo_imgsz", "yolo_device", "yolo_classes",
})


@dataclass
class TextRegion:
    """单个文字检测区域。"""
    bbox_frame: tuple[int, int, int, int]  # (x1, y1, x2, y2) 在原始帧中的坐标
    confidence: float = 0.0
    polygon: np.ndarray | None = None      # 原始多边形点（帧坐标）
    label: str = ""                        # 检测标签（YOLO 类别名）


class HullNumberLocator:
    """
    弦号定位器 — 支持 PaddleOCR 和 YOLO 两种检测器。

    流程：
    1. crop → 图像预处理（可选）
    2. 检测器推理
    3. 后处理过滤（score_threshold / min_area）
    4. 坐标转换：crop 坐标 → 原始帧坐标
    """

    def __init__(self, locator_cfg: dict):
        self._detector_type: str = locator_cfg.get("detector_type", "paddle")
        self._score_threshold: float = locator_cfg.get("score_threshold", 0.5)
        self._min_area: int = locator_cfg.get("min_area", 100)

        # Crop 保存配置
        self._save_crops: bool = bool(locator_cfg.get("save_crops", False))
        self._crop_save_dir: str = locator_cfg.get("crop_save_dir", "./crops")
        self._crop_save_interval: float = locator_cfg.get("crop_save_interval", 10)
        self._last_crop_save: float = 0.0

        # 图像预处理流水线
        from pipeline.image_processing import build_pipeline
        self._preprocess_pipeline = build_pipeline(locator_cfg.get("preprocess_steps", []))

        # 检测器参数
        self._det_model = None
        self._initialized = False

        if self._detector_type == "yolo":
            self._yolo_model_path: str = locator_cfg.get("yolo_model", "best.pt")
            self._yolo_conf: float = locator_cfg.get("yolo_conf", 0.25)
            self._yolo_iou: float = locator_cfg.get("yolo_iou", 0.45)
            self._yolo_imgsz: int = locator_cfg.get("yolo_imgsz", 640)
            self._yolo_device: str = locator_cfg.get("yolo_device", "")
            self._yolo_classes: list[int] | None = locator_cfg.get("yolo_classes")
        else:
            # PaddleOCR 参数（去掉所有我们自己的 key）
            self._paddle_kwargs: dict = {
                k: v for k, v in locator_cfg.items()
                if k not in _OWN_KEYS and v is not None
            }

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        try:
            if self._detector_type == "yolo":
                self._init_yolo()
            else:
                self._init_paddle()

            self._initialized = True
            logger.info(
                "弦号定位器初始化: detector=%s, score_threshold=%.3f, min_area=%d, save_crops=%s",
                self._detector_type, self._score_threshold, self._min_area, self._save_crops,
            )
        except Exception as e:
            logger.error("弦号定位器初始化失败: %s", e)
            raise

    def _init_paddle(self) -> None:
        """初始化 PaddleOCR TextDetection。"""
        from paddleocr import TextDetection

        logger.info("TextDetection 参数: %s", self._paddle_kwargs)
        self._det_model = TextDetection(**self._paddle_kwargs)

    def _init_yolo(self) -> None:
        """初始化 YOLO 检测器。"""
        from ultralytics import YOLO

        logger.info("加载弦号检测 YOLO 模型: %s", self._yolo_model_path)
        self._det_model = YOLO(self._yolo_model_path)

        # 打印模型类别信息
        if hasattr(self._det_model, "names"):
            logger.info("YOLO 模型类别: %s", self._det_model.names)

    def _maybe_save_crops(self, crop: np.ndarray, track_id: int) -> None:
        """每隔 crop_save_interval 秒保存 crop。"""
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
            cv2.imwrite(str(prefix) + "_crop.jpg", crop)
            logger.info("Crop 已保存: %s_crop.jpg", prefix)
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
            # 保存 crop（调试用）
            self._maybe_save_crops(crop, track_id)

            # 图像预处理（小波去噪、卷积锐化等）
            det_input = crop
            if self._preprocess_pipeline:
                from pipeline.image_processing import apply_pipeline
                det_input = apply_pipeline(det_input, self._preprocess_pipeline)

            # 检测器推理
            if self._detector_type == "yolo":
                regions = self._detect_yolo(det_input, offset_x, offset_y)
            else:
                regions = self._detect_paddle(det_input, offset_x, offset_y)

            return regions

        except Exception as e:
            logger.warning("弦号定位异常: %s", e)
            return []

    def _detect_paddle(
        self,
        det_input: np.ndarray,
        offset_x: int,
        offset_y: int,
    ) -> list[TextRegion]:
        """PaddleOCR TextDetection 检测。"""
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

    def _detect_yolo(
        self,
        det_input: np.ndarray,
        offset_x: int,
        offset_y: int,
    ) -> list[TextRegion]:
        """YOLO 目标检测。"""
        results = self._det_model.predict(
            source=det_input,
            conf=self._yolo_conf,
            iou=self._yolo_iou,
            imgsz=self._yolo_imgsz,
            device=self._yolo_device or None,
            classes=self._yolo_classes,
            verbose=False,
        )

        regions: list[TextRegion] = []

        if not results or results[0].boxes is None:
            return regions

        boxes = results[0].boxes
        names = results[0].names if hasattr(results[0], "names") else {}

        for i in range(len(boxes)):
            # 获取 bbox (xyxy 格式)
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

            # 获取置信度
            score = float(boxes.conf[i].item())

            # 获取类别
            cls_id = int(boxes.cls[i].item()) if boxes.cls is not None else -1
            label = names.get(cls_id, str(cls_id)) if names else str(cls_id)

            # 置信度过滤
            if score < self._score_threshold:
                continue

            # 面积过滤
            area = (x2 - x1) * (y2 - y1)
            if area < self._min_area:
                continue

            # 坐标转换：crop 坐标 → 帧坐标
            regions.append(TextRegion(
                bbox_frame=(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
                confidence=score,
                label=label,
            ))

        return regions

    def cleanup(self) -> None:
        self._det_model = None
        self._initialized = False

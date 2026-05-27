"""
HullNumberLocator — 基于 PaddleOCR TextDetection 的弦号定位

在 YOLO crop 上运行文字检测，返回检测到的文字区域框（已转换为原始帧坐标）。

支持 PaddleOCR 3.5 TextDetection 参数（无前缀版本），
可通过 config.yaml 灵活调整检测参数。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 我们自己的参数（从 config 中提取后不传给 TextDetection）
_OWN_KEYS = frozenset({"enabled", "score_threshold", "min_area"})


@dataclass
class TextRegion:
    """单个文字检测区域。"""
    bbox_frame: tuple[int, int, int, int]  # (x1, y1, x2, y2) 在原始帧中的坐标
    confidence: float = 0.0
    polygon: np.ndarray | None = None      # 原始多边形点（帧坐标）


class HullNumberLocator:
    """
    弦号定位器 — 使用 PaddleOCR TextDetection 在 crop 中检测文字区域。

    接受完整的 locator_cfg 字典，提取自己的参数（score_threshold / min_area），
    其余所有 det_* 参数直接透传给 PaddleOCR TextDetection。

    坐标转换流程：
    1. PaddleOCR 返回 crop 内的多边形坐标
    2. 加上 crop 在原始帧中的偏移量 (offset_x, offset_y)
    3. 得到原始帧中的绝对坐标
    """

    def __init__(self, locator_cfg: dict):
        """
        Args:
            locator_cfg: hull_locator 配置字典。包含 enabled / score_threshold /
                min_area 以及所有 PaddleOCR TextDetection 参数。
        """
        self._score_threshold: float = locator_cfg.get("score_threshold", 0.5)
        self._min_area: int = locator_cfg.get("min_area", 100)

        # 提取透传给 TextDetection 的参数（去掉我们自己的 key）
        self._paddle_kwargs: dict = {
            k: v for k, v in locator_cfg.items()
            if k not in _OWN_KEYS and v is not None
        }

        self._model = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """延迟初始化 PaddleOCR 模型（首次调用时加载）。"""
        if self._initialized:
            return

        try:
            from paddleocr import TextDetection
            logger.info("TextDetection 参数: %s", self._paddle_kwargs)
            self._model = TextDetection(**self._paddle_kwargs)
            self._initialized = True
            logger.info("PaddleOCR TextDetection 加载成功")
        except ImportError:
            logger.error(
                "PaddleOCR 未安装。请安装: pip install paddleocr>=3.5 paddlepaddle-gpu>=3.3"
            )
            raise
        except Exception as e:
            logger.error("PaddleOCR 模型加载失败: %s", e)
            raise

    def locate(
        self,
        crop: np.ndarray,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> list[TextRegion]:
        """
        在 crop 图像中定位文字区域。

        Args:
            crop: YOLO 裁剪的原始船只图像 (BGR)，未经 resize。
            offset_x: crop 在原始帧中的左上角 x 偏移量。
            offset_y: crop 在原始帧中的左上角 y 偏移量。

        Returns:
            检测到的文字区域列表，坐标已转换为原始帧坐标系。
        """
        self._ensure_initialized()

        if crop is None or crop.size == 0:
            return []

        try:
            output = self._model.predict(crop)
            logger.debug("predict 返回 %d 条结果, 类型: %s", len(output), type(output[0]) if output else "空")

            regions: list[TextRegion] = []

            for res in output:
                boxes = None
                scores = None

                # 兼容不同版本的返回格式
                if hasattr(res, 'dt_polys'):
                    boxes = res.dt_polys
                    scores = getattr(res, 'dt_scores', None)
                elif isinstance(res, dict):
                    boxes = res.get('dt_polys', res.get('boxes', None))
                    scores = res.get('dt_scores', res.get('scores', None))
                elif hasattr(res, '__dict__'):
                    obj_dict = res.__dict__
                    boxes = obj_dict.get('dt_polys', obj_dict.get('boxes', None))
                    scores = obj_dict.get('dt_scores', obj_dict.get('scores', None))

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

        except Exception as e:
            logger.warning("弦号定位异常: %s", e)
            return []

    def cleanup(self) -> None:
        """释放模型资源。"""
        self._model = None
        self._initialized = False

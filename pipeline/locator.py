"""
HullNumberLocator — 基于 PaddleOCR TextDetection 的弦号定位

在 YOLO crop 上运行文字检测，返回检测到的文字区域框（已转换为原始帧坐标）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TextRegion:
    """单个文字检测区域。"""
    bbox_frame: tuple[int, int, int, int]  # (x1, y1, x2, y2) 在原始帧中的坐标
    confidence: float = 0.0
    polygon: np.ndarray | None = None      # 原始多边形点（帧坐标）


class HullNumberLocator:
    """
    弦号定位器 — 使用 PaddleOCR TextDetection 在 crop 中检测文字区域。

    坐标转换流程：
    1. PaddleOCR 返回 crop 内的多边形坐标
    2. 加上 crop 在原始帧中的偏移量 (offset_x, offset_y)
    3. 得到原始帧中的绝对坐标
    """

    def __init__(
        self,
        score_threshold: float = 0.5,
        min_area: int = 100,
        text_det_thresh: float = 0.3,
        text_det_box_thresh: float = 0.5,
        text_det_unclip_ratio: float = 1.5,
        text_det_max_side_limit: int = 960,
    ):
        """
        Args:
            score_threshold: 文字检测置信度阈值（后处理过滤）。
            min_area: 最小文字区域面积（像素²），过滤噪声。
            text_det_thresh: 检测阈值，概率图中超过此值的像素视为文字。
            text_det_box_thresh: 检测框阈值，候选框平均分数低于此值被过滤。
            text_det_unclip_ratio: 膨胀系数，扩大检测框区域。
            text_det_max_side_limit: 输入图像最大边长限制（像素）。
        """
        self._score_threshold = score_threshold
        self._min_area = min_area
        self._text_det_thresh = text_det_thresh
        self._text_det_box_thresh = text_det_box_thresh
        self._text_det_unclip_ratio = text_det_unclip_ratio
        self._text_det_max_side_limit = text_det_max_side_limit
        self._model = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """延迟初始化 PaddleOCR 模型（首次调用时加载）。"""
        if self._initialized:
            return

        try:
            from paddleocr import TextDetection
            self._model = TextDetection(
                text_det_thresh=self._text_det_thresh,
                text_det_box_thresh=self._text_det_box_thresh,
                text_det_unclip_ratio=self._text_det_unclip_ratio,
                text_det_max_side_limit=self._text_det_max_side_limit,
            )
            self._initialized = True
            logger.info(
                "PaddleOCR TextDetection 模型加载成功: thresh=%.2f, box_thresh=%.2f, unclip=%.2f",
                self._text_det_thresh, self._text_det_box_thresh, self._text_det_unclip_ratio,
            )
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
            # 直接传 numpy 数组给 PaddleOCR，避免 JPEG 编解码损失
            output = self._model.predict(crop)

            regions: list[TextRegion] = []

            for res in output:
                # 提取检测结果
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
                    # box 是多边形点数组，shape: (N, 2)
                    box_array = np.array(box, dtype=np.int32)

                    # 获取置信度
                    score = 0.0
                    if scores is not None and idx < len(scores):
                        score = float(scores[idx])

                    # 过滤低置信度
                    if score < self._score_threshold:
                        continue

                    # 计算多边形面积
                    area = cv2.contourArea(box_array)
                    if area < self._min_area:
                        continue

                    # 坐标转换：crop 坐标 → 帧坐标
                    box_frame = box_array.copy()
                    box_frame[:, 0] += offset_x
                    box_frame[:, 1] += offset_y

                    # 计算外接矩形（帧坐标）
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

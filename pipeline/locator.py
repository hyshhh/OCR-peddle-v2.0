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
    ):
        """
        Args:
            score_threshold: 文字检测置信度阈值。
            min_area: 最小文字区域面积（像素²），过滤噪声。
        """
        self._score_threshold = score_threshold
        self._min_area = min_area
        self._model = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """延迟初始化 PaddleOCR 模型（首次调用时加载）。"""
        if self._initialized:
            return

        try:
            from paddleocr import TextDetection
            self._model = TextDetection()
            self._initialized = True
            logger.info("PaddleOCR TextDetection 模型加载成功")
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
            crop: YOLO 裁剪的船只图像 (BGR)。
            offset_x: crop 在原始帧中的左上角 x 偏移量。
            offset_y: crop 在原始帧中的左上角 y 偏移量。

        Returns:
            检测到的文字区域列表，坐标已转换为原始帧坐标系。
        """
        self._ensure_initialized()

        if crop is None or crop.size == 0:
            return []

        # 保存 crop 到临时文件（PaddleOCR TextDetection 接受文件路径）
        import tempfile
        import os

        tmp_file = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            cv2.imwrite(tmp_path, crop)
            tmp_file = tmp_path

            # 运行 PaddleOCR TextDetection
            # 保存 crop 用于调试
            import os as _os
            debug_dir = _os.path.join(_os.getcwd(), 'debug_crops')
            _os.makedirs(debug_dir, exist_ok=True)
            debug_path = _os.path.join(debug_dir, f'crop_{_os.path.basename(tmp_path)}')
            _os.system(f'cp {tmp_path} {debug_path}')
            logger.info("Crop 保存到: %s (size=%s)", debug_path, crop.shape if hasattr(crop, 'shape') else 'unknown')

            output = self._model.predict(tmp_path)
            logger.info("PaddleOCR 原始输出: %s", output)
            logger.info("PaddleOCR 输出类型: %s", type(output))
            if output:
                for i, res in enumerate(output):
                    logger.info("  结果[%d] 类型=%s", i, type(res))
                    if hasattr(res, '__dict__'):
                        logger.info("  结果[%d] attrs=%s", i, list(res.__dict__.keys()))

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

        except Exception as e:
            logger.warning("弦号定位异常: %s", e)
            return []

    def cleanup(self) -> None:
        """释放模型资源。"""
        self._model = None
        self._initialized = False

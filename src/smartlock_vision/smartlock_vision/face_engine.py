#!/usr/bin/env python3
"""臉部識別引擎核心類別

封裝 InsightFace 模型的臉部檢測、特徵提取功能
"""

import os
import numpy as np
import insightface
from pathlib import Path
from typing import List, Optional, Any
from dataclasses import dataclass

from smartlock_vision.vision_utils import get_default_logger


@dataclass
class FaceResult:
    """臉部檢測結果資料類別"""

    bbox: np.ndarray
    confidence: float
    embedding: np.ndarray
    liveness_score: Optional[float] = None


class FaceEngine:
    """臉部識別引擎 - 封裝 InsightFace 模型與特徵處理"""

    def __init__(
        self,
        model_name: str = "buffalo_sc",
        ctx_id: int = 0,
        det_thresh: float = 0.5,
        enable_gpu: bool = True,
        logger: Optional[Any] = None,
    ):
        """初始化臉部引擎

        Args:
            model_name: InsightFace 模型名稱 (buffalo_sc, buffalo_m, buffalo_l 等)
            ctx_id: GPU 設備 ID (0 為第一個 GPU，-1 為 CPU)
            det_thresh: 臉部偵測信心閾值 (0.0-1.0)
            enable_gpu: 若可用，啟用 GPU 加速
            logger: 日誌記錄器，若無則使用預設記錄器
        """
        self.model_name = model_name
        self.det_thresh = det_thresh
        self.enable_gpu = enable_gpu
        self.ctx_id = ctx_id if enable_gpu else -1
        self.logger = logger or get_default_logger(__name__)

        self.face_model: Optional[insightface.app.FaceAnalysis] = None
        self._init_model()

    def _init_model(self) -> None:
        """初始化 InsightFace FaceAnalysis 模型"""
        try:
            os.environ["INSIGHTFACE_HOME"] = str(Path.home() / ".insightface")
            providers = self._get_providers()

            self.face_model = insightface.app.FaceAnalysis(
                name=self.model_name,
                providers=providers,
            )

            self.face_model.prepare(ctx_id=self.ctx_id, det_thresh=self.det_thresh)

            self.logger.info(f"✓ InsightFace {self.model_name} 模型載入成功")
        except Exception as e:
            raise RuntimeError(f"InsightFace 初始化失敗: {e}")

    def _get_providers(self) -> List[str]:
        """取得 ONNX Runtime 的可用執行提供者

        Returns:
            List[str]: 按優先順序的提供者清單
        """
        providers = []

        if self.enable_gpu:
            try:
                import onnxruntime

                available_providers = onnxruntime.get_available_providers()

                if "CUDAExecutionProvider" in available_providers:
                    providers.append("CUDAExecutionProvider")
                    self.logger.info("✓ GPU 加速 (CUDA) 可用")
                else:
                    self.logger.warning("⚠ GPU 加速 (CUDA) 不可用，將使用 CPU")
            except ImportError:
                self.logger.warning("⚠ onnxruntime 未找到，將使用 CPU")

        providers.append("CPUExecutionProvider")
        return providers

    def detect_and_extract(self, image: np.ndarray) -> Optional[FaceResult]:
        """偵測影像中的臉部並提取特徵

        Args:
            image: BGR 格式的輸入影像 (OpenCV 格式)

        Returns:
            Optional[FaceResult]: 提取到的臉部結果，若沒有提取到則返回 None
        """
        if image is None or image.size == 0:
            return None

        try:
            if self.face_model is None:
                self.logger.error("臉部模型未初始化")
                return None

            faces = self.face_model.get(image, max_num=1)

            if len(faces) == 0:
                self.logger.warning("無法從臉部區域提取特徵")
                return None

            face = FaceResult(
                bbox=faces[0].bbox.astype(int),
                confidence=float(faces[0].det_score),
                embedding=faces[0].embedding.astype(np.float32),
            )
        except Exception as e:
            self.logger.error(f"臉部偵測並提取錯誤: {e}")
            return None

        return face

    def _detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        """偵測影像中的臉部位置

        Args:
            image: BGR 格式的輸入影像 (OpenCV 格式)

        Returns:
            Optional[np.ndarray]: 偵測到的臉部邊界框，若沒有偵測到則返回 None
        """
        if image is None or image.size == 0:
            return None

        try:
            if self.face_model is None:
                self.logger.error("臉部模型未初始化")
                return None

            bboxes, keypoints = self.face_model.det_model.detect(image, max_num=1)

            if bboxes.shape[0] == 0:
                return None

            bbox = bboxes[0, 0:4].astype(int)
            bbox = np.maximum(bbox, 0)

        except Exception as e:
            self.logger.error(f"臉部偵測錯誤: {e}")
            return None

        return bbox

    def _extract_face_region(
        self, image: np.ndarray, bbox: np.ndarray, expand_ratio: float = 0.2
    ) -> Optional[np.ndarray]:
        """從影像中提取臉部區域並做擴展

        Args:
            image: BGR 格式的輸入影像
            bbox: 邊界框座標 [x1, y1, x2, y2]
            expand_ratio: 邊界框擴展比例 (0.2 = 20% 擴展)

        Returns:
            Optional[np.ndarray]: 裁剪的臉部影像，若提取失敗則返回 None
        """
        try:
            h, w = image.shape[:2]
            x1, y1, x2, y2 = bbox

            if x2 <= x1 or y2 <= y1:
                self.logger.debug("無效的邊界框座標")
                return None

            width = x2 - x1
            height = y2 - y1
            expand_x = int(width * expand_ratio / 2)
            expand_y = int(height * expand_ratio / 2)

            x1 = max(0, x1 - expand_x)
            y1 = max(0, y1 - expand_y)
            x2 = min(w, x2 + expand_x)
            y2 = min(h, y2 + expand_y)

            face_region = image[y1:y2, x1:x2]

            if face_region.size == 0:
                return None

            return face_region
        except Exception as e:
            self.logger.debug(f"提取臉部區域錯誤: {e}")
            return None

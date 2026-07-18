"""
Robust YOLO + OCR based bar chart recognizer.
Production-ready implementation with configuration management, type hints,
error handling, and logging.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from numpy.typing import NDArray
from paddleocr import PaddleOCR
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

from .config import BarDetectorConfig, load_config


# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class TextElement:
    """Normalized text element with geometry."""
    text: str
    cx: float
    cy: float
    w: float
    h: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class AxisInfo:
    """Detected axis information."""
    elements: list[TextElement]
    coordinate: float  # x-coordinate for y-axis, y-coordinate for x-axis
    is_numeric: bool


@dataclass
class ChartComponents:
    """Detected chart components."""
    title: Optional[TextElement]
    x_axis: Optional[AxisInfo]
    y_axis: Optional[AxisInfo]
    direction: str  # 'vertical' or 'horizontal'
    rows: list[list[TextElement]]
    columns: list[list[TextElement]]


@dataclass
class BarDataPoint:
    """Single bar data point."""
    label: str
    value: float
    pixel_coordinate: float


@dataclass
class ChartData:
    """Final chart recognition result."""
    title: Optional[str]
    direction: str
    data: dict[str, float]
    y_axis_scale: tuple[float, float]  # (slope, intercept) for pixel->value conversion


class OCRError(Exception):
    """OCR processing error."""
    pass


class YOLOError(Exception):
    """YOLO model error."""
    pass


class ChartAnalysisError(Exception):
    """Chart analysis error."""
    pass


class BarDetector:
    """
    Robust YOLO + OCR based bar chart recognizer.
    
    Uses YOLO for bar/text region detection and PaddleOCR for text recognition,
    then applies geometric clustering and heuristic analysis to reconstruct
    the chart data structure.
    """
    
    def __init__(self, config: Optional[BarDetectorConfig] = None, config_path: Optional[str] = None):
        """
        Initialize BarDetector.
        
        Args:
            config: Optional BarDetectorConfig instance. If not provided, loads from config_path or defaults.
            config_path: Optional path to YAML config file.
        """
        self.config = config or load_config(config_path)
        self._init_models()
        self._class_map: dict[int, list[dict[str, Any]]] = {}
        logger.info("BarDetector initialized with config: %s", self.config.yolo.model_path)
    
    def _init_models(self) -> None:
        """Initialize YOLO and OCR models with error handling."""
        # Initialize YOLO
        try:
            model_path = Path(self.config.yolo.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"YOLO model not found: {model_path}")
            self.yolo = YOLO(str(model_path))
            self.yolo.to(self.config.yolo.device)
            logger.info("YOLO model loaded on %s", self.config.yolo.device)
        except Exception as e:
            logger.error("Failed to load YOLO model: %s", e)
            raise YOLOError(f"Failed to load YOLO model: {e}") from e
        
        # Initialize OCR
        try:
            ocr_cfg = self.config.ocr
            self.ocr = PaddleOCR(
                lang=ocr_cfg.lang,
                use_doc_orientation_classify=ocr_cfg.use_doc_orientation_classify,
                use_doc_unwarping=ocr_cfg.use_doc_unwarping,
                use_textline_orientation=ocr_cfg.use_textline_orientation,
                device=ocr_cfg.device,
                enable_mkldnn=ocr_cfg.enable_mkldnn,
            )
            logger.info("PaddleOCR initialized on %s", ocr_cfg.device)
        except Exception as e:
            logger.error("Failed to initialize PaddleOCR: %s", e)
            raise OCRError(f"Failed to initialize PaddleOCR: {e}") from e
    
    # ---------------------------------------------------------------------
    # Geometry utilities
    # ---------------------------------------------------------------------
    
    @staticmethod
    def _xyxy_to_geometry(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
        """Convert bbox (x1, y1, x2, y2) to (cx, cy, w, h)."""
        return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    
    @staticmethod
    def _apply_padding(x1: float, y1: float, x2: float, y2: float, pad: int) -> tuple[float, float, float, float]:
        """Apply padding to bounding box."""
        return x1 + pad, y1 + pad, x2 - pad, y2 - pad
    
    # ---------------------------------------------------------------------
    # OCR result normalization
    # ---------------------------------------------------------------------
    
    def normalize_ocr_result(self, ocr_result: list[dict[str, Any]]) -> list[TextElement]:
        """
        Convert raw OCR result to structured TextElement list.
        
        Args:
            ocr_result: List of dicts with 'text' and 'box' (x1, y1, x2, y2).
            
        Returns:
            List of TextElement objects.
        """
        elements = []
        for item in ocr_result:
            try:
                x1, y1, x2, y2 = item['box']
                cx, cy, w, h = self._xyxy_to_geometry(x1, y1, x2, y2)
                elements.append(TextElement(
                    text=item['text'].strip(),
                    cx=cx, cy=cy, w=w, h=h,
                    x1=x1, y1=y1, x2=x2, y2=y2
                ))
            except (KeyError, ValueError, TypeError) as e:
                logger.warning("Skipping invalid OCR item %s: %s", item, e)
                continue
        return elements
    
    # ---------------------------------------------------------------------
    # Clustering
    # ---------------------------------------------------------------------
    
    def cluster_rows(self, texts: list[TextElement]) -> list[list[TextElement]]:
        """Cluster text elements into rows (by y-coordinate)."""
        if not texts:
            return []
        
        ys = np.array([[t.cy] for t in texts], dtype=np.float32)
        eps = self.config.clustering.row_eps
        min_samples = self.config.clustering.min_samples
        
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(ys)
        
        rows: dict[int, list[TextElement]] = {}
        for label, text in zip(labels, texts):
            rows.setdefault(label, []).append(text)
        
        # Sort each row left-to-right
        output = []
        for row in rows.values():
            row.sort(key=lambda t: t.cx)
            output.append(row)
        
        # Sort rows top-to-bottom
        output.sort(key=lambda r: np.mean([t.cy for t in r]))
        return output
    
    def cluster_columns(self, texts: list[TextElement]) -> list[list[TextElement]]:
        """Cluster text elements into columns (by x-coordinate)."""
        if not texts:
            return []
        
        xs = np.array([[t.cx] for t in texts], dtype=np.float32)
        eps = self.config.clustering.column_eps
        min_samples = self.config.clustering.min_samples
        
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xs)
        
        cols: dict[int, list[TextElement]] = {}
        for label, text in zip(labels, texts):
            cols.setdefault(label, []).append(text)
        
        output = list(cols.values())
        # Sort columns left-to-right
        output.sort(key=lambda c: np.mean([t.cx for t in c]))
        return output
    
    # ---------------------------------------------------------------------
    # Text classification utilities
    # ---------------------------------------------------------------------
    
    @staticmethod
    def is_number(text: str) -> bool:
        """Check if text represents a number."""
        if not text:
            return False
        try:
            float(text)
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def try_int(value: str) -> int | str:
        """Convert to int if possible, otherwise return original string."""
        if isinstance(value, str) and value.lstrip('+-').isdigit():
            return int(value)
        return value
    
    # ---------------------------------------------------------------------
    # Chart component detection
    # ---------------------------------------------------------------------
    
    def detect_title(self, texts: list[TextElement]) -> Optional[TextElement]:
        """Detect chart title (top-most non-numeric text with sufficient length)."""
        candidates = [
            t for t in texts
            if len(t.text) >= self.config.title_detection.min_title_length
            and not self.is_number(t.text)
        ]
        
        if not candidates:
            logger.debug("No title candidates found")
            return None
        
        # Return top-most candidate
        candidates.sort(key=lambda t: t.cy)
        title = candidates[0]
        logger.debug("Detected title: '%s' at y=%.1f", title.text, title.cy)
        return title
    
    def detect_x_axis(self, rows: list[list[TextElement]], img_height: int) -> Optional[AxisInfo]:
        """
        Detect X-axis: bottom row with multiple non-numeric (for vertical bars) 
        or numeric (for horizontal bars) elements.
        """
        if not rows:
            return None
        
        threshold_y = img_height * self.config.axis_detection.x_axis_height_fraction
        min_elements = self.config.axis_detection.min_axis_elements
        
        # Check from bottom up
        for row in reversed(rows):
            avg_y = np.mean([t.cy for t in row])
            if avg_y > threshold_y and len(row) >= min_elements:
                is_numeric = all(self.is_number(t.text) for t in row)
                logger.debug("Detected X-axis at y=%.1f (numeric=%s, count=%d)", 
                           avg_y, is_numeric, len(row))
                return AxisInfo(
                    elements=row,
                    coordinate=avg_y,
                    is_numeric=is_numeric
                )
        
        logger.debug("No X-axis detected")
        return None
    
    def detect_y_axis(self, columns: list[list[TextElement]], img_width: int) -> Optional[AxisInfo]:
        """
        Detect Y-axis: leftmost column with multiple numeric elements.
        """
        if not columns:
            return None
        
        threshold_x = img_width * self.config.axis_detection.y_axis_width_fraction
        min_elements = self.config.axis_detection.min_axis_elements
        
        for col in columns:
            avg_x = np.mean([t.cx for t in col])
            if avg_x < threshold_x and len(col) >= min_elements:
                is_numeric = all(self.is_number(t.text) for t in col)
                logger.debug("Detected Y-axis at x=%.1f (numeric=%s, count=%d)",
                           avg_x, is_numeric, len(col))
                return AxisInfo(
                    elements=col,
                    coordinate=avg_x,
                    is_numeric=is_numeric
                )
        
        logger.debug("No Y-axis detected")
        return None
    
    # ---------------------------------------------------------------------
    # Full chart analysis
    # ---------------------------------------------------------------------
    
    def analyze_chart(self, ocr_result: list[dict[str, Any]], img_width: int, img_height: int) -> ChartComponents:
        """
        Analyze OCR results to extract chart structure.
        
        Args:
            ocr_result: Raw OCR output.
            img_width: Image width in pixels.
            img_height: Image height in pixels.
            
        Returns:
            ChartComponents with detected structure.
        """
        texts = self.normalize_ocr_result(ocr_result)
        
        if not texts:
            logger.warning("No text elements found in OCR result")
            return ChartComponents(
                title=None, x_axis=None, y_axis=None,
                direction='vertical', rows=[], columns=[]
            )
        
        rows = self.cluster_rows(texts)
        columns = self.cluster_columns(texts)
        title = self.detect_title(texts)
        x_axis = self.detect_x_axis(rows, img_height)
        y_axis = self.detect_y_axis(columns, img_width)
        
        # Determine chart direction
        if x_axis and x_axis.is_numeric:
            direction = 'horizontal'
        else:
            direction = 'vertical'
        
        logger.info("Chart analysis: direction=%s, title=%s, x_axis=%s, y_axis=%s",
                   direction, title.text if title else None,
                   'detected' if x_axis else 'none', 'detected' if y_axis else 'none')
        
        return ChartComponents(
            title=title,
            x_axis=x_axis,
            y_axis=y_axis,
            direction=direction,
            rows=rows,
            columns=columns
        )
    
    # ---------------------------------------------------------------------
    # Bar edge refinement
    # ---------------------------------------------------------------------
    
    def refine_top_y(self, crop: NDArray[np.uint8]) -> int:
        """
        Refine top Y coordinate of a vertical bar using horizontal projection profile.
        
        Args:
            crop: Bar crop (BGR image).
            
        Returns:
            Y offset from crop top.
        """
        if crop.size == 0:
            return 0
        
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        h, w = binary.shape
        cfg = self.config.bar_refinement
        
        # Central ROI
        left = int(w * cfg.vertical_roi_left_frac)
        right = int(w * cfg.vertical_roi_right_frac)
        roi = binary[:, left:right]
        
        # Horizontal projection
        profile = np.count_nonzero(roi, axis=1)
        threshold = roi.shape[1] * cfg.vertical_top_threshold_frac
        
        for y in range(h):
            if profile[y] >= threshold:
                return y
        
        return 0
    
    def refine_right_x(self, crop: NDArray[np.uint8]) -> int:
        """
        Refine right X coordinate of a horizontal bar by rotating and reusing refine_top_y.
        
        Args:
            crop: Bar crop (BGR image).
            
        Returns:
            X coordinate from crop left (0 = left edge, width = right edge).
        """
        if crop.size == 0:
            return crop.shape[1]
        
        # Rotate 90° CCW: right edge becomes top edge
        rot = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        top = self.refine_top_y(rot)
        
        # Convert back: top in rotated = right in original
        return crop.shape[1] - top
    
    # ---------------------------------------------------------------------
    # YOLO result processing
    # ---------------------------------------------------------------------
    
    def _populate_class_map(self, yolo_result) -> None:
        """Populate internal class map from YOLO result."""
        self._class_map.clear()
        boxes = yolo_result.boxes
        
        for box in boxes:
            try:
                cls_id = int(box.cls[0])
                xyxy = box.xyxy[0].tolist()
                
                # Apply padding
                x1, y1, x2, y2 = self._apply_padding(
                    xyxy[0], xyxy[1], xyxy[2], xyxy[3], self.config.box_padding
                )
                
                conf = float(box.conf[0])
                
                if conf < self.config.yolo.confidence_threshold:
                    continue
                
                self._class_map.setdefault(cls_id, []).append({
                    'bbox': [x1, y1, x2, y2],
                    'conf': conf
                })
            except (IndexError, ValueError) as e:
                logger.warning("Skipping invalid YOLO box: %s", e)
                continue
        
        # Sort by confidence descending
        for cls_id in self._class_map:
            self._class_map[cls_id].sort(key=lambda x: x['conf'], reverse=True)
        
        logger.debug("YOLO classes detected: %s", {k: len(v) for k, v in self._class_map.items()})
    
    # ---------------------------------------------------------------------
    # Main pipeline
    # ---------------------------------------------------------------------
    
    def run(self, image_path: str | Path) -> ChartData:
        """
        Run full bar chart recognition pipeline.
        
        Args:
            image_path: Path to chart image.
            
        Returns:
            ChartData with recognized chart data.
            
        Raises:
            FileNotFoundError: If image not found.
            ChartAnalysisError: If chart analysis fails.
            YOLOError: If YOLO inference fails.
            OCRError: If OCR fails.
        """
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ChartAnalysisError(f"Failed to load image: {img_path}")
        
        img_h, img_w = img.shape[:2]
        logger.info("Processing image: %s (%dx%d)", img_path.name, img_w, img_h)
        
        # YOLO inference
        try:
            yolo_results = self.yolo(str(img_path))
            self._populate_class_map(yolo_results[0])
        except Exception as e:
            logger.error("YOLO inference failed: %s", e)
            raise YOLOError(f"YOLO inference failed: {e}") from e
        
        # Check required classes
        bar_class = self.config.bar_class_id
        text_class = self.config.text_class_id
        
        if bar_class not in self._class_map or not self._class_map[bar_class]:
            raise ChartAnalysisError("No bars detected")
        if text_class not in self._class_map or not self._class_map[text_class]:
            raise ChartAnalysisError("No text regions detected")
        
        # Crop text regions and run OCR
        crops = []
        poses = []
        for box in self._class_map[text_class]:
            x1, y1, x2, y2 = map(int, box['bbox'])
            # Validate crop coordinates
            x1 = max(0, min(x1, img_w - 1))
            x2 = max(0, min(x2, img_w))
            y1 = max(0, min(y1, img_h - 1))
            y2 = max(0, min(y2, img_h))
            
            if x2 <= x1 or y2 <= y1:
                logger.warning("Invalid crop coordinates: (%d,%d,%d,%d)", x1, y1, x2, y2)
                continue
            
            crops.append(img[y1:y2, x1:x2])
            poses.append((x1, x2, y1, y2))
        
        if not crops:
            raise ChartAnalysisError("No valid text crops extracted")
        
        # Run OCR
        try:
            ocr_results_raw = self.ocr.predict(crops)
            texts = []
            for r in ocr_results_raw:
                rec_texts = r.get('rec_texts', [])
                texts.append(rec_texts[0] if rec_texts else '')
        except Exception as e:
            logger.error("OCR prediction failed: %s", e)
            raise OCRError(f"OCR prediction failed: {e}") from e
        
        # Build OCR results with positions
        ocr_results = []
        for text, (x1, x2, y1, y2) in zip(texts, poses):
            ocr_results.append({'text': text, 'box': (x1, y1, x2, y2)})
        
        # Analyze chart structure
        components = self.analyze_chart(ocr_results, img_w, img_h)
        
        # Extract axis data for scale computation
        x_axis = components.x_axis
        y_axis = components.y_axis
        
        if not x_axis or not y_axis:
            raise ChartAnalysisError("Failed to detect both axes")
        
        # Build pixel-to-value mapping from Y-axis (value axis)
        if components.direction == 'vertical':
            # Y-axis has values, X-axis has labels
            value_axis = y_axis
            label_axis = x_axis
        else:
            # Horizontal bars: X-axis has values, Y-axis has labels
            value_axis = x_axis
            label_axis = y_axis
        
        if not value_axis.is_numeric:
            raise ChartAnalysisError("Value axis does not contain numeric labels")
        
        # Compute pixel->value linear mapping
        pixels = [elem.cy if components.direction == 'vertical' else elem.cx 
                  for elem in value_axis.elements]
        values = [float(elem.text) for elem in value_axis.elements]
        
        if len(pixels) < 2:
            raise ChartAnalysisError("Insufficient value axis points for scale computation")
        
        try:
            coef_a, coef_b = np.polyfit(pixels, values, 1)
        except np.linalg.LinAlgError as e:
            raise ChartAnalysisError(f"Failed to compute scale: {e}") from e
        
        logger.info("Scale computed: value = %.4f * pixel + %.4f", coef_a, coef_b)
        
        # Process each bar
        data: dict[str, float] = {}
        
        for bar in self._class_map[bar_class]:
            x1, y1, x2, y2 = map(int, bar['bbox'])
            
            # Validate crop
            if x2 <= x1 or y2 <= y1:
                continue
            
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            if components.direction == 'vertical':
                # Refine top of vertical bar
                local_top = self.refine_top_y(crop)
                value_coord = y1 + local_top
                
                # Match to X-axis label (center of bar)
                label_coord = (x1 + x2) / 2
            else:
                # Refine right edge of horizontal bar
                local_right = self.refine_right_x(crop)
                value_coord = x1 + local_right
                
                # Match to Y-axis label (center of bar)
                label_coord = (y1 + y2) / 2
            
            # Find closest label
            label_elem = min(
                label_axis.elements,
                key=lambda t: abs((t.cx if components.direction == 'vertical' else t.cy) - label_coord)
            )
            label = label_elem.text
            
            # Convert pixel coordinate to value
            value = coef_a * value_coord + coef_b
            data[label] = float(value)
            
            logger.debug("Bar: label='%s', pixel=%.1f, value=%.2f", label, value_coord, value)
        
        return ChartData(
            title=components.title.text if components.title else None,
            direction=components.direction,
            data=data,
            y_axis_scale=(float(coef_a), float(coef_b))
        )


# Convenience function for quick usage
def detect_bar_chart(image_path: str | Path, config_path: Optional[str] = None) -> ChartData:
    """
    Convenience function for single-shot bar chart detection.
    
    Args:
        image_path: Path to chart image.
        config_path: Optional config file path.
        
    Returns:
        ChartData with recognized chart data.
    """
    detector = BarDetector(config_path=config_path)
    return detector.run(image_path)
"""
Configuration for BarDetector.
All tunable parameters are defined here to avoid magic numbers in the code.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class OCRConfig:
    """PaddleOCR configuration."""
    lang: str = "en"
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = False
    device: str = "cpu"  # "cpu" or "gpu"
    enable_mkldnn: bool = False


@dataclass
class YOLOConfig:
    """YOLO model configuration."""
    model_path: str = "src/bar_text_yolov8n_640_32.pt"
    device: str = "cuda"  # "cuda" or "cpu"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45


@dataclass
class ClusteringConfig:
    """DBSCAN clustering configuration for row/column detection."""
    row_eps: float = 20.0
    column_eps: float = 20.0
    min_samples: int = 1


@dataclass
class AxisDetectionConfig:
    """Axis detection heuristics configuration."""
    # X-axis: bottom portion of image (fraction of height)
    x_axis_height_fraction: float = 0.75
    # Y-axis: left portion of image (fraction of width)
    y_axis_width_fraction: float = 0.25
    # Minimum number of text elements to consider as axis
    min_axis_elements: int = 2


@dataclass
class TitleDetectionConfig:
    """Title detection configuration."""
    min_title_length: int = 3


@dataclass
class BarRefinementConfig:
    """Bar edge refinement configuration."""
    # For vertical bars: central ROI fraction (width)
    vertical_roi_left_frac: float = 0.3
    vertical_roi_right_frac: float = 0.7
    # Threshold for detecting bar top (fraction of ROI width)
    vertical_top_threshold_frac: float = 0.4
    # For horizontal bars: same parameters applied after rotation
    horizontal_roi_top_frac: float = 0.3
    horizontal_roi_bottom_frac: float = 0.7
    horizontal_right_threshold_frac: float = 0.4


@dataclass
class BarDetectorConfig:
    """Main configuration for BarDetector."""
    # Padding applied to YOLO bounding boxes
    box_padding: int = 1
    
    # Sub-configurations
    ocr: OCRConfig = field(default_factory=OCRConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    axis_detection: AxisDetectionConfig = field(default_factory=AxisDetectionConfig)
    title_detection: TitleDetectionConfig = field(default_factory=TitleDetectionConfig)
    bar_refinement: BarRefinementConfig = field(default_factory=BarRefinementConfig)
    
    # Class IDs (from YOLO model)
    bar_class_id: int = 0
    text_class_id: int = 1


def load_config(config_path: Optional[str] = None) -> BarDetectorConfig:
    """
    Load configuration from a YAML file if provided, otherwise return defaults.
    
    Args:
        config_path: Optional path to YAML config file.
        
    Returns:
        BarDetectorConfig instance.
    """
    if config_path and Path(config_path).exists():
        import yaml
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        return _dict_to_config(data)
    return BarDetectorConfig()


def _dict_to_config(data: dict) -> BarDetectorConfig:
    """Convert dictionary to BarDetectorConfig."""
    return BarDetectorConfig(
        box_padding=data.get('box_padding', 1),
        ocr=OCRConfig(**data.get('ocr', {})),
        yolo=YOLOConfig(**data.get('yolo', {})),
        clustering=ClusteringConfig(**data.get('clustering', {})),
        axis_detection=AxisDetectionConfig(**data.get('axis_detection', {})),
        title_detection=TitleDetectionConfig(**data.get('title_detection', {})),
        bar_refinement=BarRefinementConfig(**data.get('bar_refinement', {})),
        bar_class_id=data.get('bar_class_id', 0),
        text_class_id=data.get('text_class_id', 1),
    )
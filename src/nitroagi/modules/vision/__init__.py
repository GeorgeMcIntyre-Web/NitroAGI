"""
Vision Module for NitroAGI NEXUS
Provides computer vision capabilities including object detection, scene analysis, and OCR
"""

from .vision_module import VisionModule
from .processors import (
    ImageProcessor,
    ObjectDetector,
    SceneAnalyzer,
    OCRProcessor
)

__all__ = [
    "VisionModule",
    "ImageProcessor",
    "ObjectDetector",
    "SceneAnalyzer",
    "OCRProcessor"
]
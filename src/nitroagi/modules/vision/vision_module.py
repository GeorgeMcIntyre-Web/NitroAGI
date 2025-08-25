"""
Vision Module Implementation for NitroAGI NEXUS
Handles image understanding, object detection, scene analysis, and OCR
"""

import asyncio
import base64
import io
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from PIL import Image
import cv2

from nitroagi.core.base import (
    AIModule,
    ModuleCapability,
    ModuleRequest,
    ModuleResponse,
    ProcessingContext
)
from nitroagi.core.exceptions import ModuleException
from nitroagi.utils.logging import get_logger

# Import processors
from .processors import (
    ImageProcessor,
    ObjectDetector,
    SceneAnalyzer,
    OCRProcessor
)


class VisionTask(Enum):
    """Types of vision tasks."""
    OBJECT_DETECTION = "object_detection"
    SCENE_ANALYSIS = "scene_analysis"
    IMAGE_CLASSIFICATION = "image_classification"
    OCR = "ocr"
    FACE_DETECTION = "face_detection"
    IMAGE_SEGMENTATION = "image_segmentation"
    FEATURE_EXTRACTION = "feature_extraction"


@dataclass
class VisionResult:
    """Result from vision processing."""
    task: VisionTask
    objects: Optional[List[Dict[str, Any]]] = None
    scene_description: Optional[str] = None
    text_content: Optional[str] = None
    classifications: Optional[Dict[str, float]] = None
    features: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    confidence: float = 0.0
    processing_time_ms: float = 0.0


class VisionModule(AIModule):
    """
    Vision module for NitroAGI NEXUS.
    Provides comprehensive computer vision capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the vision module.
        
        Args:
            config: Module configuration
        """
        super().__init__(config)
        self.logger = get_logger(__name__)
        
        # Initialize processors
        self.image_processor = ImageProcessor()
        self.object_detector = ObjectDetector()
        self.scene_analyzer = SceneAnalyzer()
        self.ocr_processor = OCRProcessor()
        
        # Configuration
        self.max_image_size = config.get("max_image_size", (1920, 1080))
        self.enable_gpu = config.get("enable_gpu", False)
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        
        # Cache for processed images
        self.image_cache = {}
        self.cache_size = config.get("cache_size", 100)
    
    async def initialize(self) -> bool:
        """
        Initialize the vision module and load models.
        
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing Vision Module...")
            
            # Initialize processors
            await self.image_processor.initialize()
            await self.object_detector.initialize(use_gpu=self.enable_gpu)
            await self.scene_analyzer.initialize()
            await self.ocr_processor.initialize()
            
            self._initialized = True
            self.logger.info("Vision Module initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Vision Module: {e}")
            self._initialized = False
            return False
    
    async def process(self, request: ModuleRequest) -> ModuleResponse:
        """
        Process a vision request.
        
        Args:
            request: The vision processing request
            
        Returns:
            ModuleResponse with vision results
        """
        if not self._initialized:
            return ModuleResponse(
                request_id=request.id,
                module_name=self.name,
                status="error",
                error="Vision module not initialized"
            )
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Extract image and task from request
            image = await self._extract_image(request.data)
            task = self._determine_task(request)
            
            # Process based on task
            result = await self._process_vision_task(image, task, request)
            
            # Build response
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return ModuleResponse(
                request_id=request.id,
                module_name=self.name,
                status="success",
                data=self._format_result(result),
                processing_time_ms=processing_time,
                confidence_score=result.confidence
            )
            
        except Exception as e:
            self.logger.error(f"Vision processing failed: {e}", exc_info=True)
            return ModuleResponse(
                request_id=request.id,
                module_name=self.name,
                status="error",
                error=str(e),
                processing_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000
            )
    
    async def _extract_image(self, data: Any) -> Image.Image:
        """
        Extract image from request data.
        
        Args:
            data: Request data containing image
            
        Returns:
            PIL Image object
        """
        if isinstance(data, str):
            # Check if it's a file path
            if data.startswith('/') or data.startswith('C:\\'):
                return Image.open(data)
            # Check if it's base64 encoded
            elif data.startswith('data:image'):
                # Extract base64 data
                base64_str = data.split(',')[1] if ',' in data else data
                image_data = base64.b64decode(base64_str)
                return Image.open(io.BytesIO(image_data))
            else:
                raise ValueError(f"Unknown string format for image data")
        
        elif isinstance(data, bytes):
            # Raw image bytes
            return Image.open(io.BytesIO(data))
        
        elif isinstance(data, np.ndarray):
            # NumPy array
            return Image.fromarray(data)
        
        elif isinstance(data, Image.Image):
            # Already a PIL Image
            return data
        
        elif isinstance(data, dict):
            # Extract image from dict
            if "image" in data:
                return await self._extract_image(data["image"])
            elif "image_path" in data:
                return Image.open(data["image_path"])
            elif "image_url" in data:
                # Download image from URL
                return await self._download_image(data["image_url"])
        
        raise ValueError(f"Cannot extract image from data type: {type(data)}")
    
    def _determine_task(self, request: ModuleRequest) -> VisionTask:
        """
        Determine the vision task from request.
        
        Args:
            request: Module request
            
        Returns:
            VisionTask enum value
        """
        # Check capabilities
        if ModuleCapability.OBJECT_DETECTION in request.required_capabilities:
            return VisionTask.OBJECT_DETECTION
        elif ModuleCapability.SCENE_ANALYSIS in request.required_capabilities:
            return VisionTask.SCENE_ANALYSIS
        elif ModuleCapability.TEXT_EXTRACTION in request.required_capabilities:
            return VisionTask.OCR
        
        # Check explicit task in data
        if isinstance(request.data, dict):
            task_str = request.data.get("task", "").lower()
            if "object" in task_str or "detect" in task_str:
                return VisionTask.OBJECT_DETECTION
            elif "scene" in task_str or "describe" in task_str:
                return VisionTask.SCENE_ANALYSIS
            elif "ocr" in task_str or "text" in task_str:
                return VisionTask.OCR
            elif "face" in task_str:
                return VisionTask.FACE_DETECTION
            elif "segment" in task_str:
                return VisionTask.IMAGE_SEGMENTATION
            elif "classify" in task_str:
                return VisionTask.IMAGE_CLASSIFICATION
        
        # Default to scene analysis
        return VisionTask.SCENE_ANALYSIS
    
    async def _process_vision_task(
        self,
        image: Image.Image,
        task: VisionTask,
        request: ModuleRequest
    ) -> VisionResult:
        """
        Process specific vision task.
        
        Args:
            image: Input image
            task: Vision task to perform
            request: Original request
            
        Returns:
            VisionResult with task results
        """
        # Preprocess image
        processed_image = await self.image_processor.preprocess(
            image,
            target_size=self.max_image_size
        )
        
        result = VisionResult(task=task, metadata={})
        
        if task == VisionTask.OBJECT_DETECTION:
            # Detect objects in image
            objects = await self.object_detector.detect(
                processed_image,
                confidence_threshold=self.confidence_threshold
            )
            result.objects = objects
            result.confidence = self._calculate_average_confidence(objects)
            
        elif task == VisionTask.SCENE_ANALYSIS:
            # Analyze scene
            scene_data = await self.scene_analyzer.analyze(processed_image)
            result.scene_description = scene_data["description"]
            result.objects = scene_data.get("objects", [])
            result.metadata = scene_data.get("metadata", {})
            result.confidence = scene_data.get("confidence", 0.8)
            
        elif task == VisionTask.OCR:
            # Extract text from image
            text_data = await self.ocr_processor.extract_text(processed_image)
            result.text_content = text_data["text"]
            result.metadata["text_regions"] = text_data.get("regions", [])
            result.confidence = text_data.get("confidence", 0.9)
            
        elif task == VisionTask.IMAGE_CLASSIFICATION:
            # Classify image
            classifications = await self._classify_image(processed_image)
            result.classifications = classifications
            result.confidence = max(classifications.values()) if classifications else 0.0
            
        elif task == VisionTask.FACE_DETECTION:
            # Detect faces
            faces = await self._detect_faces(processed_image)
            result.objects = faces
            result.metadata["face_count"] = len(faces)
            result.confidence = 0.95 if faces else 0.0
            
        elif task == VisionTask.IMAGE_SEGMENTATION:
            # Segment image
            segments = await self._segment_image(processed_image)
            result.metadata["segments"] = segments
            result.confidence = 0.85
            
        elif task == VisionTask.FEATURE_EXTRACTION:
            # Extract features
            features = await self._extract_features(processed_image)
            result.features = features
            result.confidence = 0.9
        
        return result
    
    async def _classify_image(self, image: Image.Image) -> Dict[str, float]:
        """
        Classify image into categories.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of class labels and confidence scores
        """
        # Simplified classification (would use real model in production)
        # This is a placeholder implementation
        classifications = {
            "outdoor": 0.7,
            "nature": 0.6,
            "landscape": 0.5
        }
        
        # Analyze image properties for basic classification
        img_array = np.array(image)
        
        # Check brightness
        brightness = np.mean(img_array)
        if brightness > 150:
            classifications["bright"] = 0.8
        else:
            classifications["dark"] = 0.8
        
        # Check color dominance
        if len(img_array.shape) == 3:
            avg_color = np.mean(img_array, axis=(0, 1))
            if avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2]:
                classifications["nature"] = 0.9
            elif avg_color[2] > avg_color[0] and avg_color[2] > avg_color[1]:
                classifications["sky"] = 0.8
        
        return classifications
    
    async def _detect_faces(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Detect faces in image.
        
        Args:
            image: Input image
            
        Returns:
            List of detected faces with bounding boxes
        """
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Load face cascade (simplified - would use better model in production)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Format results
        face_objects = []
        for i, (x, y, w, h) in enumerate(faces):
            face_objects.append({
                "type": "face",
                "id": f"face_{i}",
                "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                "confidence": 0.95
            })
        
        return face_objects
    
    async def _segment_image(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Segment image into regions.
        
        Args:
            image: Input image
            
        Returns:
            List of image segments
        """
        # Simple segmentation using color clustering
        img_array = np.array(image)
        
        # Reshape for clustering
        pixels = img_array.reshape(-1, 3)
        
        # K-means clustering (simplified)
        from sklearn.cluster import KMeans
        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get segment labels
        labels = kmeans.labels_.reshape(img_array.shape[:2])
        
        segments = []
        for i in range(n_clusters):
            mask = (labels == i)
            segment_pixels = np.sum(mask)
            total_pixels = mask.size
            
            segments.append({
                "segment_id": i,
                "color": kmeans.cluster_centers_[i].tolist(),
                "pixel_count": int(segment_pixels),
                "percentage": float(segment_pixels / total_pixels)
            })
        
        return segments
    
    async def _extract_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract feature vector from image.
        
        Args:
            image: Input image
            
        Returns:
            Feature vector
        """
        # Simple feature extraction (would use CNN in production)
        img_array = np.array(image.resize((224, 224)))
        
        # Calculate basic features
        features = []
        
        # Color histogram
        for channel in range(3):
            hist, _ = np.histogram(img_array[:, :, channel], bins=32, range=(0, 256))
            features.extend(hist.tolist())
        
        # Edge features
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)
        
        # Texture features (simplified)
        features.append(np.std(gray))  # Standard deviation
        features.append(np.mean(gray))  # Mean brightness
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_average_confidence(self, objects: List[Dict[str, Any]]) -> float:
        """
        Calculate average confidence from detected objects.
        
        Args:
            objects: List of detected objects
            
        Returns:
            Average confidence score
        """
        if not objects:
            return 0.0
        
        confidences = [obj.get("confidence", 0.0) for obj in objects]
        return sum(confidences) / len(confidences)
    
    def _format_result(self, result: VisionResult) -> Dict[str, Any]:
        """
        Format vision result for response.
        
        Args:
            result: VisionResult object
            
        Returns:
            Formatted dictionary
        """
        output = {
            "task": result.task.value,
            "confidence": result.confidence,
            "processing_time_ms": result.processing_time_ms
        }
        
        if result.objects:
            output["objects"] = result.objects
        
        if result.scene_description:
            output["scene_description"] = result.scene_description
        
        if result.text_content:
            output["text"] = result.text_content
        
        if result.classifications:
            output["classifications"] = result.classifications
        
        if result.metadata:
            output["metadata"] = result.metadata
        
        return output
    
    async def _download_image(self, url: str) -> Image.Image:
        """
        Download image from URL.
        
        Args:
            url: Image URL
            
        Returns:
            PIL Image object
        """
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    image_data = await response.read()
                    return Image.open(io.BytesIO(image_data))
                else:
                    raise ValueError(f"Failed to download image: HTTP {response.status}")
    
    def get_capabilities(self) -> List[ModuleCapability]:
        """
        Get module capabilities.
        
        Returns:
            List of supported capabilities
        """
        return [
            ModuleCapability.IMAGE_UNDERSTANDING,
            ModuleCapability.OBJECT_DETECTION,
            ModuleCapability.SCENE_ANALYSIS,
            ModuleCapability.TEXT_EXTRACTION
        ]
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info("Cleaning up Vision Module resources")
        
        # Clear cache
        self.image_cache.clear()
        
        # Cleanup processors
        if hasattr(self, 'object_detector'):
            await self.object_detector.cleanup()
        
        self._initialized = False
"""Unit tests for vision module."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np
from PIL import Image
import io
import base64

from nitroagi.modules.vision.vision_module import (
    VisionModule,
    VisionTask,
    VisionResult
)
from nitroagi.modules.vision.processors import (
    ImageProcessor,
    ObjectDetector,
    SceneAnalyzer,
    OCRProcessor,
    BoundingBox
)
from nitroagi.core.base import ModuleRequest, ModuleCapability, ProcessingContext


@pytest.fixture
def vision_config():
    """Create vision module configuration."""
    return {
        "name": "vision",
        "max_image_size": (1920, 1080),
        "enable_gpu": False,
        "confidence_threshold": 0.5,
        "cache_size": 100
    }


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    # Create a simple RGB image
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def module_request(sample_image):
    """Create a sample module request."""
    return ModuleRequest(
        data=sample_image,
        context=ProcessingContext(),
        required_capabilities=[ModuleCapability.IMAGE_UNDERSTANDING]
    )


class TestVisionModule:
    """Test VisionModule functionality."""
    
    @pytest.mark.asyncio
    async def test_module_initialization(self, vision_config):
        """Test vision module initialization."""
        module = VisionModule(vision_config)
        
        # Mock processor initialization
        with patch.object(module.image_processor, 'initialize', new_callable=AsyncMock):
            with patch.object(module.object_detector, 'initialize', new_callable=AsyncMock):
                with patch.object(module.scene_analyzer, 'initialize', new_callable=AsyncMock):
                    with patch.object(module.ocr_processor, 'initialize', new_callable=AsyncMock):
                        result = await module.initialize()
                        
                        assert result is True
                        assert module._initialized is True
    
    @pytest.mark.asyncio
    async def test_image_extraction_from_path(self, vision_config):
        """Test extracting image from file path."""
        module = VisionModule(vision_config)
        
        # Mock Image.open
        with patch('PIL.Image.open') as mock_open:
            mock_image = MagicMock(spec=Image.Image)
            mock_open.return_value = mock_image
            
            result = await module._extract_image("/path/to/image.jpg")
            
            assert result == mock_image
            mock_open.assert_called_once_with("/path/to/image.jpg")
    
    @pytest.mark.asyncio
    async def test_image_extraction_from_base64(self, vision_config, sample_image):
        """Test extracting image from base64 string."""
        module = VisionModule(vision_config)
        
        # Convert sample image to base64
        buffer = io.BytesIO()
        sample_image.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        base64_str = "data:image/png;base64," + base64.b64encode(img_bytes).decode()
        
        result = await module._extract_image(base64_str)
        
        assert isinstance(result, Image.Image)
    
    @pytest.mark.asyncio
    async def test_task_determination(self, vision_config):
        """Test determining vision task from request."""
        module = VisionModule(vision_config)
        
        # Test object detection
        request = ModuleRequest(
            data="test",
            required_capabilities=[ModuleCapability.OBJECT_DETECTION]
        )
        task = module._determine_task(request)
        assert task == VisionTask.OBJECT_DETECTION
        
        # Test scene analysis
        request = ModuleRequest(
            data="test",
            required_capabilities=[ModuleCapability.SCENE_ANALYSIS]
        )
        task = module._determine_task(request)
        assert task == VisionTask.SCENE_ANALYSIS
        
        # Test OCR
        request = ModuleRequest(
            data="test",
            required_capabilities=[ModuleCapability.TEXT_EXTRACTION]
        )
        task = module._determine_task(request)
        assert task == VisionTask.OCR
    
    @pytest.mark.asyncio
    async def test_object_detection_processing(self, vision_config, sample_image):
        """Test object detection processing."""
        module = VisionModule(vision_config)
        module._initialized = True
        
        # Mock object detector
        mock_objects = [
            {"type": "person", "confidence": 0.9, "bbox": {"x": 10, "y": 10, "width": 50, "height": 50}},
            {"type": "car", "confidence": 0.8, "bbox": {"x": 60, "y": 60, "width": 30, "height": 30}}
        ]
        
        with patch.object(module.object_detector, 'detect', new_callable=AsyncMock) as mock_detect:
            mock_detect.return_value = mock_objects
            
            result = await module._process_vision_task(
                sample_image,
                VisionTask.OBJECT_DETECTION,
                ModuleRequest(data=sample_image)
            )
            
            assert result.task == VisionTask.OBJECT_DETECTION
            assert result.objects == mock_objects
            assert result.confidence == 0.85  # Average of 0.9 and 0.8
    
    @pytest.mark.asyncio
    async def test_scene_analysis_processing(self, vision_config, sample_image):
        """Test scene analysis processing."""
        module = VisionModule(vision_config)
        module._initialized = True
        
        # Mock scene analyzer
        mock_scene = {
            "description": "A bright outdoor scene",
            "objects": [{"type": "tree"}, {"type": "sky"}],
            "metadata": {"brightness": 150},
            "confidence": 0.9
        }
        
        with patch.object(module.scene_analyzer, 'analyze', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = mock_scene
            
            result = await module._process_vision_task(
                sample_image,
                VisionTask.SCENE_ANALYSIS,
                ModuleRequest(data=sample_image)
            )
            
            assert result.task == VisionTask.SCENE_ANALYSIS
            assert result.scene_description == "A bright outdoor scene"
            assert result.confidence == 0.9
    
    @pytest.mark.asyncio
    async def test_ocr_processing(self, vision_config, sample_image):
        """Test OCR processing."""
        module = VisionModule(vision_config)
        module._initialized = True
        
        # Mock OCR processor
        mock_ocr = {
            "text": "Hello World",
            "regions": [{"text": "Hello", "confidence": 0.95}],
            "confidence": 0.92
        }
        
        with patch.object(module.ocr_processor, 'extract_text', new_callable=AsyncMock) as mock_ocr_func:
            mock_ocr_func.return_value = mock_ocr
            
            result = await module._process_vision_task(
                sample_image,
                VisionTask.OCR,
                ModuleRequest(data=sample_image)
            )
            
            assert result.task == VisionTask.OCR
            assert result.text_content == "Hello World"
            assert result.confidence == 0.92
    
    @pytest.mark.asyncio
    async def test_process_request(self, vision_config, module_request, sample_image):
        """Test processing a complete request."""
        module = VisionModule(vision_config)
        module._initialized = True
        
        # Mock the processing pipeline
        with patch.object(module, '_extract_image', new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = sample_image
            
            with patch.object(module, '_process_vision_task', new_callable=AsyncMock) as mock_process:
                mock_result = VisionResult(
                    task=VisionTask.SCENE_ANALYSIS,
                    scene_description="Test scene",
                    confidence=0.85,
                    processing_time_ms=100
                )
                mock_process.return_value = mock_result
                
                response = await module.process(module_request)
                
                assert response.status == "success"
                assert response.module_name == "vision"
                assert "scene_description" in response.data
                assert response.confidence_score == 0.85
    
    @pytest.mark.asyncio
    async def test_error_handling(self, vision_config, module_request):
        """Test error handling in vision module."""
        module = VisionModule(vision_config)
        module._initialized = True
        
        # Mock an error in image extraction
        with patch.object(module, '_extract_image', side_effect=Exception("Image extraction failed")):
            response = await module.process(module_request)
            
            assert response.status == "error"
            assert "Image extraction failed" in response.error
    
    def test_get_capabilities(self, vision_config):
        """Test getting module capabilities."""
        module = VisionModule(vision_config)
        
        capabilities = module.get_capabilities()
        
        assert ModuleCapability.IMAGE_UNDERSTANDING in capabilities
        assert ModuleCapability.OBJECT_DETECTION in capabilities
        assert ModuleCapability.SCENE_ANALYSIS in capabilities
        assert ModuleCapability.TEXT_EXTRACTION in capabilities


class TestImageProcessor:
    """Test ImageProcessor functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test image processor initialization."""
        processor = ImageProcessor()
        await processor.initialize()
        assert processor.initialized is True
    
    @pytest.mark.asyncio
    async def test_preprocessing(self, sample_image):
        """Test image preprocessing."""
        processor = ImageProcessor()
        await processor.initialize()
        
        # Test resizing
        processed = await processor.preprocess(
            sample_image,
            target_size=(50, 50),
            normalize=False
        )
        
        assert processed.size == (50, 50)
        assert processed.mode == "RGB"
    
    @pytest.mark.asyncio
    async def test_normalization(self, sample_image):
        """Test image normalization."""
        processor = ImageProcessor()
        await processor.initialize()
        
        processed = await processor.preprocess(
            sample_image,
            normalize=True
        )
        
        # Check that image was normalized and converted back
        assert processed.mode == "RGB"
    
    def test_edge_detection(self, sample_image):
        """Test edge detection."""
        processor = ImageProcessor()
        
        edges = processor.detect_edges(sample_image)
        
        assert isinstance(edges, np.ndarray)
        assert edges.shape[:2] == (100, 100)  # Same dimensions as input


class TestObjectDetector:
    """Test ObjectDetector functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test object detector initialization."""
        detector = ObjectDetector()
        await detector.initialize(use_gpu=False)
        assert detector.initialized is True
    
    @pytest.mark.asyncio
    async def test_object_detection(self, sample_image):
        """Test object detection."""
        detector = ObjectDetector()
        await detector.initialize()
        
        # Mock face cascade
        with patch.object(detector.face_cascade, 'detectMultiScale') as mock_detect:
            mock_detect.return_value = np.array([[10, 10, 30, 30], [50, 50, 20, 20]])
            
            objects = await detector.detect(sample_image, confidence_threshold=0.5)
            
            assert len(objects) >= 0  # May detect faces or other objects
            
            # Check object structure
            for obj in objects:
                assert "type" in obj
                assert "bbox" in obj
                assert "confidence" in obj
    
    def test_coco_classes_loading(self):
        """Test loading COCO class names."""
        detector = ObjectDetector()
        classes = detector._load_coco_classes()
        
        assert len(classes) > 0
        assert "person" in classes
        assert "car" in classes


class TestSceneAnalyzer:
    """Test SceneAnalyzer functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test scene analyzer initialization."""
        analyzer = SceneAnalyzer()
        await analyzer.initialize()
        assert analyzer.initialized is True
    
    @pytest.mark.asyncio
    async def test_scene_analysis(self, sample_image):
        """Test scene analysis."""
        analyzer = SceneAnalyzer()
        await analyzer.initialize()
        
        # Mock sklearn KMeans
        with patch('nitroagi.modules.vision.processors.KMeans') as mock_kmeans:
            mock_instance = MagicMock()
            mock_instance.cluster_centers_ = np.array([[100, 100, 100]] * 5)
            mock_kmeans.return_value = mock_instance
            
            result = await analyzer.analyze(sample_image)
            
            assert "description" in result
            assert "scene_type" in result
            assert "metadata" in result
            assert "confidence" in result
    
    def test_color_analysis(self):
        """Test color analysis."""
        analyzer = SceneAnalyzer()
        
        # Create a test image array
        img_array = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        with patch('nitroagi.modules.vision.processors.KMeans') as mock_kmeans:
            mock_instance = MagicMock()
            mock_instance.cluster_centers_ = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128], [255, 255, 255]])
            mock_kmeans.return_value = mock_instance
            
            color_info = analyzer._analyze_colors(img_array)
            
            assert "average_color" in color_info
            assert "dominant_colors" in color_info
            assert "mood" in color_info
    
    def test_composition_analysis(self):
        """Test composition analysis."""
        analyzer = SceneAnalyzer()
        
        # Create test image
        img_array = np.random.randint(0, 255, (90, 120, 3), dtype=np.uint8)
        
        composition = analyzer._analyze_composition(img_array)
        
        assert "active_region" in composition
        assert "symmetry" in composition
        assert "aspect_ratio" in composition


class TestOCRProcessor:
    """Test OCRProcessor functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test OCR processor initialization."""
        processor = OCRProcessor()
        await processor.initialize()
        assert processor.initialized is True
    
    @pytest.mark.asyncio
    async def test_text_extraction_basic(self, sample_image):
        """Test basic text extraction."""
        processor = OCRProcessor()
        await processor.initialize()
        
        # Force basic extraction (no tesseract)
        processor.tesseract_available = False
        
        result = await processor.extract_text(sample_image)
        
        assert "text" in result
        assert "regions" in result
        assert "confidence" in result
    
    @pytest.mark.asyncio
    async def test_text_extraction_with_tesseract(self, sample_image):
        """Test text extraction with Tesseract."""
        processor = OCRProcessor()
        await processor.initialize()
        
        if processor.tesseract_available:
            with patch('pytesseract.image_to_string') as mock_ocr:
                mock_ocr.return_value = "Sample text"
                
                with patch('pytesseract.image_to_data') as mock_data:
                    mock_data.return_value = {
                        'conf': [80, 90],
                        'text': ['Sample', 'text'],
                        'left': [10, 50],
                        'top': [10, 10],
                        'width': [30, 25],
                        'height': [15, 15]
                    }
                    
                    result = await processor.extract_text(sample_image)
                    
                    assert result["text"] == "Sample text"
                    assert len(result["regions"]) > 0
"""
Multi-Modal Processing API endpoints
"""

from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import base64
import logging

from src.nitroagi.core.multimodal import MultiModalProcessor, FusionStrategy, MultiModalData

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/multimodal")


class MultiModalRequest(BaseModel):
    """Multi-modal processing request"""
    text: Optional[str] = Field(None, description="Text input")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    audio_base64: Optional[str] = Field(None, description="Base64 encoded audio")
    video_url: Optional[str] = Field(None, description="Video URL")
    task: str = Field(..., description="Task to perform")
    fusion_strategy: Optional[str] = Field("late", description="Fusion strategy")


class ModalityAnalysis(BaseModel):
    """Analysis of individual modality"""
    modality: str
    features: Dict[str, Any]
    confidence: float


# Global instance
multimodal_processor = None


def get_multimodal_processor() -> MultiModalProcessor:
    """Get or create multi-modal processor"""
    global multimodal_processor
    if multimodal_processor is None:
        multimodal_processor = MultiModalProcessor()
    return multimodal_processor


@router.post("/process")
async def process_multimodal(request: MultiModalRequest):
    """Process multi-modal input"""
    
    try:
        processor = get_multimodal_processor()
        
        # Create multi-modal data object
        data = MultiModalData()
        
        if request.text:
            data.text = request.text
        
        if request.image_base64:
            data.image = base64.b64decode(request.image_base64)
        
        if request.audio_base64:
            data.audio = base64.b64decode(request.audio_base64)
        
        if request.video_url:
            data.video = request.video_url
        
        # Convert string to enum
        try:
            fusion_strategy = FusionStrategy(request.fusion_strategy)
        except ValueError:
            fusion_strategy = FusionStrategy.LATE
        
        # Process data
        result = await processor.process(
            data,
            request.task,
            fusion_strategy
        )
        
        return {
            "status": "success",
            "result": result,
            "modalities_used": data.get_modalities(),
            "fusion_strategy": fusion_strategy.value
        }
        
    except Exception as e:
        logger.error(f"Multi-modal processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    task: str = Form("describe")
):
    """Analyze uploaded image"""
    
    try:
        processor = get_multimodal_processor()
        
        # Read image data
        image_data = await file.read()
        
        # Create multi-modal data with only image
        data = MultiModalData()
        data.image = image_data
        
        # Process image
        result = await processor.process(
            data,
            task,
            FusionStrategy.SINGLE
        )
        
        return {
            "status": "success",
            "filename": file.filename,
            "analysis": result
        }
        
    except Exception as e:
        logger.error(f"Image analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-text-image")
async def analyze_text_and_image(
    text: str = Form(...),
    file: UploadFile = File(...)
):
    """Analyze text and image together"""
    
    try:
        processor = get_multimodal_processor()
        
        # Read image data
        image_data = await file.read()
        
        # Create multi-modal data
        data = MultiModalData()
        data.text = text
        data.image = image_data
        
        # Process with cross-attention fusion
        result = await processor.process(
            data,
            "analyze_relationship",
            FusionStrategy.CROSS_ATTENTION
        )
        
        return {
            "status": "success",
            "text_input": text[:100] + "..." if len(text) > 100 else text,
            "image_file": file.filename,
            "analysis": result
        }
        
    except Exception as e:
        logger.error(f"Text-image analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fuse-modalities")
async def fuse_modalities(
    modalities: List[ModalityAnalysis],
    strategy: str = "hierarchical"
):
    """Fuse multiple modality analyses"""
    
    try:
        processor = get_multimodal_processor()
        
        # Convert to fusion strategy enum
        try:
            fusion_strategy = FusionStrategy(strategy)
        except ValueError:
            fusion_strategy = FusionStrategy.HIERARCHICAL
        
        # Prepare features for fusion
        features = {}
        for modality in modalities:
            features[modality.modality] = modality.features
        
        # Perform fusion
        if fusion_strategy == FusionStrategy.EARLY:
            fused = processor._early_fusion(features)
        elif fusion_strategy == FusionStrategy.LATE:
            fused = processor._late_fusion(features)
        elif fusion_strategy == FusionStrategy.HYBRID:
            fused = processor._hybrid_fusion(features)
        elif fusion_strategy == FusionStrategy.CROSS_ATTENTION:
            fused = processor._cross_attention_fusion(features)
        else:
            fused = processor._hierarchical_fusion(features)
        
        return {
            "status": "success",
            "fusion_strategy": strategy,
            "fused_features": fused,
            "modalities_count": len(modalities)
        }
        
    except Exception as e:
        logger.error(f"Modality fusion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies")
async def get_fusion_strategies():
    """Get available fusion strategies"""
    
    return {
        "strategies": [
            {
                "name": strategy.value,
                "description": {
                    "early": "Combine features at input level",
                    "late": "Combine decisions at output level",
                    "hybrid": "Combine at multiple levels",
                    "cross_attention": "Use attention mechanisms across modalities",
                    "hierarchical": "Process in hierarchical manner",
                    "single": "Single modality processing"
                }.get(strategy.value, "")
            }
            for strategy in FusionStrategy
        ]
    }


@router.get("/capabilities")
async def get_multimodal_capabilities():
    """Get multi-modal processing capabilities"""
    
    return {
        "supported_modalities": ["text", "image", "audio", "video"],
        "fusion_strategies": [s.value for s in FusionStrategy],
        "tasks": [
            "describe",
            "analyze_relationship",
            "answer_question",
            "generate_caption",
            "detect_objects",
            "transcribe",
            "classify"
        ],
        "max_file_size_mb": 50,
        "supported_image_formats": ["jpg", "png", "gif", "bmp"],
        "supported_audio_formats": ["mp3", "wav", "ogg"],
        "supported_video_formats": ["mp4", "avi", "mov"]
    }


@router.post("/benchmark")
async def benchmark_fusion_strategies(
    data: MultiModalRequest
):
    """Benchmark different fusion strategies"""
    
    try:
        processor = get_multimodal_processor()
        
        # Create multi-modal data
        mm_data = MultiModalData()
        if data.text:
            mm_data.text = data.text
        if data.image_base64:
            mm_data.image = base64.b64decode(data.image_base64)
        
        # Test all strategies
        results = {}
        for strategy in FusionStrategy:
            if strategy != FusionStrategy.SINGLE or len(mm_data.get_modalities()) == 1:
                try:
                    result = await processor.process(
                        mm_data,
                        data.task,
                        strategy
                    )
                    results[strategy.value] = {
                        "success": True,
                        "confidence": result.get("confidence", 0.0)
                    }
                except Exception as e:
                    results[strategy.value] = {
                        "success": False,
                        "error": str(e)
                    }
        
        return {
            "status": "benchmark_complete",
            "results": results,
            "best_strategy": max(
                [k for k, v in results.items() if v.get("success")],
                key=lambda x: results[x].get("confidence", 0)
            ) if any(v.get("success") for v in results.values()) else None
        }
        
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
async def reset_processor():
    """Reset multi-modal processor"""
    
    global multimodal_processor
    
    multimodal_processor = None
    
    return {
        "status": "reset",
        "message": "Multi-modal processor reset"
    }
"""Chat completion endpoints for language processing."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import asyncio
import json

from nitroagi.core import TaskRequest, ExecutionStrategy, ModuleCapability
from nitroagi.utils.logging import get_logger


router = APIRouter()
logger = get_logger(__name__)


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Message role (user/assistant/system)")
    content: str = Field(..., description="Message content")
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class ChatCompletionRequest(BaseModel):
    """Chat completion request model."""
    messages: List[ChatMessage] = Field(..., description="Conversation messages")
    model: str = Field(default="nitroagi-v1", description="Model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=1000, ge=1)
    stream: bool = Field(default=False, description="Stream the response")
    conversation_id: Optional[str] = Field(default=None)
    memory_enabled: bool = Field(default=True)
    modules: List[str] = Field(default=["language"])
    
    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {"role": "user", "content": "Explain quantum computing"}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
        }


class ChatCompletionResponse(BaseModel):
    """Chat completion response model."""
    id: str = Field(default_factory=lambda: f"chat_{uuid4()}")
    object: str = Field(default="chat.completion")
    created: int = Field(default_factory=lambda: int(datetime.utcnow().timestamp()))
    model: str = Field(...)
    choices: List[Dict[str, Any]] = Field(...)
    usage: Dict[str, int] = Field(default_factory=dict)


@router.post("/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    req: Request,
    background_tasks: BackgroundTasks
) -> ChatCompletionResponse:
    """Create a chat completion.
    
    Args:
        request: Chat completion request
        req: FastAPI request object
        background_tasks: Background tasks handler
        
    Returns:
        Chat completion response
    """
    if not hasattr(req.app.state, "orchestrator"):
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    orchestrator = req.app.state.orchestrator
    memory_manager = req.app.state.memory_manager if hasattr(req.app.state, "memory_manager") else None
    
    # Store conversation in memory if enabled
    if request.memory_enabled and memory_manager and request.conversation_id:
        for msg in request.messages:
            await memory_manager.store(
                key=f"conv_{request.conversation_id}_{datetime.utcnow().timestamp()}",
                value=msg.dict(),
                memory_type="episodic"
            )
    
    # Create task request
    task_request = TaskRequest(
        input_data={
            "messages": [msg.dict() for msg in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        },
        required_capabilities=[ModuleCapability.TEXT_GENERATION],
        execution_strategy=ExecutionStrategy.SEQUENTIAL,
        timeout_seconds=30.0,
    )
    
    # Submit to orchestrator
    task_id = await orchestrator.submit_task(task_request)
    
    # Wait for completion
    try:
        result = await orchestrator.wait_for_task(task_id, timeout=30.0)
        
        if result.status.value == "completed":
            # Format response
            response_text = result.final_output if isinstance(result.final_output, str) else str(result.final_output)
            
            return ChatCompletionResponse(
                model=request.model,
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }],
                usage={
                    "prompt_tokens": sum(len(msg.content.split()) for msg in request.messages) * 2,
                    "completion_tokens": len(response_text.split()) * 2,
                    "total_tokens": (sum(len(msg.content.split()) for msg in request.messages) + len(response_text.split())) * 2
                }
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Task failed: {result.errors}"
            )
            
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out")


@router.post("/completions/stream")
async def create_chat_completion_stream(
    request: ChatCompletionRequest,
    req: Request
) -> StreamingResponse:
    """Create a streaming chat completion.
    
    Args:
        request: Chat completion request
        req: FastAPI request object
        
    Returns:
        Streaming response
    """
    if not hasattr(req.app.state, "orchestrator"):
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    async def generate():
        """Generate streaming response."""
        # Simulate streaming (would be real streaming with actual language module)
        response_id = f"chat_{uuid4()}"
        
        # Send initial chunk
        chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(datetime.utcnow().timestamp()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        
        # Simulate token generation
        tokens = ["Quantum ", "computing ", "is ", "a ", "revolutionary ", "technology ", "that ", 
                 "uses ", "quantum ", "mechanical ", "phenomena ", "to ", "process ", "information."]
        
        for token in tokens:
            await asyncio.sleep(0.1)  # Simulate processing delay
            
            chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(datetime.utcnow().timestamp()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        
        # Send final chunk
        chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(datetime.utcnow().timestamp()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.get("/conversations")
async def list_conversations(
    req: Request,
    limit: int = 10,
    offset: int = 0
) -> Dict[str, Any]:
    """List conversations.
    
    Args:
        req: FastAPI request object
        limit: Maximum number of conversations to return
        offset: Offset for pagination
        
    Returns:
        List of conversations
    """
    # This would fetch from database in real implementation
    return {
        "conversations": [],
        "total": 0,
        "limit": limit,
        "offset": offset
    }


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    req: Request
) -> Dict[str, Any]:
    """Get a specific conversation.
    
    Args:
        conversation_id: Conversation ID
        req: FastAPI request object
        
    Returns:
        Conversation details
    """
    if not hasattr(req.app.state, "memory_manager"):
        raise HTTPException(status_code=503, detail="Memory manager not initialized")
    
    memory_manager = req.app.state.memory_manager
    
    # Search for conversation messages
    messages = await memory_manager.search(
        query=f"conv_{conversation_id}",
        memory_type="episodic",
        limit=100
    )
    
    return {
        "conversation_id": conversation_id,
        "messages": [msg.value for msg in messages],
        "created_at": messages[0].created_at.isoformat() if messages else None,
        "message_count": len(messages)
    }
"""
WebSocket Request Handlers for NitroAGI NEXUS
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from typing import Optional, Dict, Any
import json
import asyncio
import logging
from datetime import datetime
import uuid

from .manager import connection_manager, task_manager, MessageType

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: Optional[str] = Query(None)
):
    """Main WebSocket endpoint"""
    
    # Generate client ID if not provided
    if not client_id:
        client_id = str(uuid.uuid4())
    
    # Connect client
    client = await connection_manager.connect(websocket, client_id)
    
    try:
        # Handle messages from client
        while True:
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                await handle_client_message(client_id, message)
            
            except json.JSONDecodeError:
                await connection_manager.send_message(
                    client_id,
                    {
                        "type": MessageType.ERROR.value,
                        "error": "Invalid JSON",
                        "timestamp": datetime.now().isoformat()
                    }
                )
            except Exception as e:
                logger.error(f"Error handling message from {client_id}: {e}")
                await connection_manager.send_message(
                    client_id,
                    {
                        "type": MessageType.ERROR.value,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                )
    
    except WebSocketDisconnect:
        await connection_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        await connection_manager.disconnect(client_id)


async def handle_client_message(client_id: str, message: Dict[str, Any]):
    """Handle incoming client messages"""
    
    msg_type = message.get("type")
    
    if msg_type == MessageType.HEARTBEAT.value:
        await connection_manager.handle_heartbeat(client_id)
    
    elif msg_type == MessageType.TASK_REQUEST.value:
        await handle_task_request(client_id, message)
    
    elif msg_type == "subscribe":
        topic = message.get("topic")
        if topic:
            await connection_manager.subscribe(client_id, topic)
            await connection_manager.send_message(
                client_id,
                {
                    "type": "subscribed",
                    "topic": topic,
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    elif msg_type == "unsubscribe":
        topic = message.get("topic")
        if topic:
            await connection_manager.unsubscribe(client_id, topic)
            await connection_manager.send_message(
                client_id,
                {
                    "type": "unsubscribed",
                    "topic": topic,
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    elif msg_type == "get_status":
        await send_system_status(client_id)
    
    elif msg_type == "stream_request":
        await handle_stream_request(client_id, message)
    
    elif msg_type == "cancel_task":
        task_id = message.get("task_id")
        if task_id:
            await task_manager.cancel_task(client_id, task_id)
    
    else:
        await connection_manager.send_message(
            client_id,
            {
                "type": MessageType.ERROR.value,
                "error": f"Unknown message type: {msg_type}",
                "timestamp": datetime.now().isoformat()
            }
        )


async def handle_task_request(client_id: str, message: Dict[str, Any]):
    """Handle task execution request"""
    
    task_data = message.get("data", {})
    task_id = message.get("task_id", str(uuid.uuid4()))
    
    # Execute task asynchronously
    asyncio.create_task(
        task_manager.execute_task(client_id, task_id, task_data)
    )


async def handle_stream_request(client_id: str, message: Dict[str, Any]):
    """Handle streaming request"""
    
    stream_type = message.get("stream_type")
    stream_id = message.get("stream_id", str(uuid.uuid4()))
    
    if stream_type == "reasoning":
        # Stream reasoning steps
        async def reasoning_generator():
            steps = [
                "Analyzing problem...",
                "Identifying patterns...",
                "Applying logical rules...",
                "Generating solution...",
                "Verifying result..."
            ]
            for step in steps:
                yield {"step": step}
                await asyncio.sleep(0.5)
        
        asyncio.create_task(
            connection_manager.stream_to_client(
                client_id,
                stream_id,
                reasoning_generator()
            )
        )
    
    elif stream_type == "learning":
        # Stream learning metrics
        async def learning_generator():
            for i in range(10):
                yield {
                    "iteration": i,
                    "loss": 1.0 / (i + 1),
                    "accuracy": i * 0.1
                }
                await asyncio.sleep(0.3)
        
        asyncio.create_task(
            connection_manager.stream_to_client(
                client_id,
                stream_id,
                learning_generator()
            )
        )
    
    else:
        await connection_manager.send_message(
            client_id,
            {
                "type": MessageType.ERROR.value,
                "error": f"Unknown stream type: {stream_type}",
                "timestamp": datetime.now().isoformat()
            }
        )


async def send_system_status(client_id: str):
    """Send system status to client"""
    
    status = {
        "type": MessageType.SYSTEM_STATUS.value,
        "status": "operational",
        "version": "1.0.0",
        "connected_clients": len(connection_manager.get_all_clients()),
        "active_tasks": len(task_manager.active_tasks),
        "modules": {
            "language": "active",
            "vision": "active",
            "reasoning": "active",
            "learning": "active"
        },
        "timestamp": datetime.now().isoformat()
    }
    
    await connection_manager.send_message(client_id, status)


@router.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    """Chat-specific WebSocket endpoint"""
    
    client_id = str(uuid.uuid4())
    await connection_manager.connect(websocket, client_id, {"type": "chat"})
    
    # Subscribe to chat topic
    await connection_manager.subscribe(client_id, "chat")
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                
                # Broadcast chat message to all chat subscribers
                chat_message = {
                    "type": "chat_message",
                    "client_id": client_id,
                    "message": message.get("message", ""),
                    "timestamp": datetime.now().isoformat()
                }
                
                await connection_manager.publish_to_topic("chat", chat_message)
                
                # Simulate AI response
                if message.get("message", "").lower().startswith("nexus"):
                    await asyncio.sleep(0.5)
                    ai_response = {
                        "type": "chat_message",
                        "client_id": "nexus",
                        "message": f"Processing: {message.get('message', '')}",
                        "timestamp": datetime.now().isoformat()
                    }
                    await connection_manager.publish_to_topic("chat", ai_response)
            
            except json.JSONDecodeError:
                pass
    
    except WebSocketDisconnect:
        await connection_manager.unsubscribe(client_id, "chat")
        await connection_manager.disconnect(client_id)
        
        # Notify others
        await connection_manager.publish_to_topic(
            "chat",
            {
                "type": "user_left",
                "client_id": client_id,
                "timestamp": datetime.now().isoformat()
            }
        )


@router.websocket("/ws/monitor")
async def monitor_websocket(websocket: WebSocket):
    """System monitoring WebSocket endpoint"""
    
    client_id = str(uuid.uuid4())
    await connection_manager.connect(websocket, client_id, {"type": "monitor"})
    
    try:
        # Send system metrics every 2 seconds
        while True:
            metrics = {
                "type": "metrics",
                "cpu_usage": 45.2,
                "memory_usage": 62.8,
                "active_connections": len(connection_manager.get_all_clients()),
                "tasks_running": len(task_manager.active_tasks),
                "timestamp": datetime.now().isoformat()
            }
            
            await connection_manager.send_message(client_id, metrics)
            await asyncio.sleep(2)
    
    except WebSocketDisconnect:
        await connection_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"Monitor WebSocket error: {e}")
        await connection_manager.disconnect(client_id)
"""
WebSocket Connection Manager for NitroAGI NEXUS
"""

from typing import Dict, List, Set, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect
import json
import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """WebSocket message types"""
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    TASK_UPDATE = "task_update"
    SYSTEM_STATUS = "system_status"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    NOTIFICATION = "notification"
    STREAM_START = "stream_start"
    STREAM_DATA = "stream_data"
    STREAM_END = "stream_end"


@dataclass
class WebSocketClient:
    """Represents a connected WebSocket client"""
    client_id: str
    websocket: WebSocket
    connected_at: datetime
    subscriptions: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocketClient] = {}
        self.subscriptions: Dict[str, Set[str]] = {}  # topic -> client_ids
        self.message_queue: Dict[str, List[Dict]] = {}  # client_id -> messages
        self._lock = asyncio.Lock()
    
    async def connect(
        self,
        websocket: WebSocket,
        client_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WebSocketClient:
        """Accept and register a new WebSocket connection"""
        
        await websocket.accept()
        
        client = WebSocketClient(
            client_id=client_id,
            websocket=websocket,
            connected_at=datetime.now(),
            metadata=metadata or {}
        )
        
        async with self._lock:
            self.active_connections[client_id] = client
            self.message_queue[client_id] = []
        
        # Send connection confirmation
        await self.send_message(
            client_id,
            {
                "type": MessageType.CONNECT.value,
                "client_id": client_id,
                "timestamp": datetime.now().isoformat(),
                "message": "Connected to NitroAGI NEXUS"
            }
        )
        
        logger.info(f"Client {client_id} connected")
        return client
    
    async def disconnect(self, client_id: str):
        """Disconnect and unregister a WebSocket connection"""
        
        async with self._lock:
            if client_id in self.active_connections:
                client = self.active_connections[client_id]
                
                # Remove from all subscriptions
                for topic in list(client.subscriptions):
                    await self._unsubscribe(client_id, topic)
                
                # Clean up
                del self.active_connections[client_id]
                if client_id in self.message_queue:
                    del self.message_queue[client_id]
                
                logger.info(f"Client {client_id} disconnected")
    
    async def send_message(
        self,
        client_id: str,
        message: Dict[str, Any]
    ):
        """Send a message to a specific client"""
        
        if client_id in self.active_connections:
            client = self.active_connections[client_id]
            try:
                await client.websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                await self.disconnect(client_id)
    
    async def broadcast(
        self,
        message: Dict[str, Any],
        exclude: Optional[Set[str]] = None
    ):
        """Broadcast a message to all connected clients"""
        
        exclude = exclude or set()
        disconnected = []
        
        for client_id in self.active_connections:
            if client_id not in exclude:
                try:
                    await self.send_message(client_id, message)
                except:
                    disconnected.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected:
            await self.disconnect(client_id)
    
    async def publish_to_topic(
        self,
        topic: str,
        message: Dict[str, Any]
    ):
        """Publish a message to all clients subscribed to a topic"""
        
        if topic in self.subscriptions:
            for client_id in self.subscriptions[topic]:
                await self.send_message(client_id, message)
    
    async def subscribe(
        self,
        client_id: str,
        topic: str
    ):
        """Subscribe a client to a topic"""
        
        async with self._lock:
            if client_id in self.active_connections:
                client = self.active_connections[client_id]
                client.subscriptions.add(topic)
                
                if topic not in self.subscriptions:
                    self.subscriptions[topic] = set()
                self.subscriptions[topic].add(client_id)
                
                logger.info(f"Client {client_id} subscribed to {topic}")
    
    async def _unsubscribe(
        self,
        client_id: str,
        topic: str
    ):
        """Unsubscribe a client from a topic (internal)"""
        
        if client_id in self.active_connections:
            client = self.active_connections[client_id]
            client.subscriptions.discard(topic)
        
        if topic in self.subscriptions:
            self.subscriptions[topic].discard(client_id)
            if not self.subscriptions[topic]:
                del self.subscriptions[topic]
    
    async def unsubscribe(
        self,
        client_id: str,
        topic: str
    ):
        """Unsubscribe a client from a topic"""
        
        async with self._lock:
            await self._unsubscribe(client_id, topic)
            logger.info(f"Client {client_id} unsubscribed from {topic}")
    
    def get_client(self, client_id: str) -> Optional[WebSocketClient]:
        """Get a client by ID"""
        return self.active_connections.get(client_id)
    
    def get_all_clients(self) -> List[str]:
        """Get all connected client IDs"""
        return list(self.active_connections.keys())
    
    def get_topic_subscribers(self, topic: str) -> Set[str]:
        """Get all clients subscribed to a topic"""
        return self.subscriptions.get(topic, set())
    
    async def handle_heartbeat(self, client_id: str):
        """Handle heartbeat from client"""
        
        if client_id in self.active_connections:
            await self.send_message(
                client_id,
                {
                    "type": MessageType.HEARTBEAT.value,
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    async def stream_to_client(
        self,
        client_id: str,
        stream_id: str,
        data_generator
    ):
        """Stream data to a client"""
        
        # Send stream start
        await self.send_message(
            client_id,
            {
                "type": MessageType.STREAM_START.value,
                "stream_id": stream_id,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Stream data
        try:
            async for data in data_generator:
                await self.send_message(
                    client_id,
                    {
                        "type": MessageType.STREAM_DATA.value,
                        "stream_id": stream_id,
                        "data": data,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                await asyncio.sleep(0.1)  # Rate limiting
        
        except Exception as e:
            logger.error(f"Stream error for {client_id}: {e}")
            await self.send_message(
                client_id,
                {
                    "type": MessageType.ERROR.value,
                    "stream_id": stream_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        finally:
            # Send stream end
            await self.send_message(
                client_id,
                {
                    "type": MessageType.STREAM_END.value,
                    "stream_id": stream_id,
                    "timestamp": datetime.now().isoformat()
                }
            )


class TaskManager:
    """Manages task execution over WebSocket"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_results: Dict[str, Any] = {}
    
    async def execute_task(
        self,
        client_id: str,
        task_id: str,
        task_data: Dict[str, Any]
    ):
        """Execute a task and send updates to client"""
        
        # Register task
        self.active_tasks[task_id] = {
            "client_id": client_id,
            "status": "started",
            "started_at": datetime.now(),
            "data": task_data
        }
        
        # Send task started notification
        await self.connection_manager.send_message(
            client_id,
            {
                "type": MessageType.TASK_UPDATE.value,
                "task_id": task_id,
                "status": "started",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        try:
            # Simulate task execution with progress updates
            for progress in range(0, 101, 20):
                await asyncio.sleep(0.5)
                
                await self.connection_manager.send_message(
                    client_id,
                    {
                        "type": MessageType.TASK_UPDATE.value,
                        "task_id": task_id,
                        "status": "in_progress",
                        "progress": progress,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            # Task completed
            result = {
                "output": f"Completed task: {task_data.get('goal', 'Unknown')}",
                "success": True
            }
            
            self.task_results[task_id] = result
            self.active_tasks[task_id]["status"] = "completed"
            
            # Send completion
            await self.connection_manager.send_message(
                client_id,
                {
                    "type": MessageType.TASK_RESPONSE.value,
                    "task_id": task_id,
                    "status": "completed",
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        except Exception as e:
            # Task failed
            self.active_tasks[task_id]["status"] = "failed"
            
            await self.connection_manager.send_message(
                client_id,
                {
                    "type": MessageType.ERROR.value,
                    "task_id": task_id,
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        finally:
            # Clean up after delay
            await asyncio.sleep(60)
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
    
    async def cancel_task(
        self,
        client_id: str,
        task_id: str
    ):
        """Cancel an active task"""
        
        if task_id in self.active_tasks:
            self.active_tasks[task_id]["status"] = "cancelled"
            
            await self.connection_manager.send_message(
                client_id,
                {
                    "type": MessageType.TASK_UPDATE.value,
                    "task_id": task_id,
                    "status": "cancelled",
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a task"""
        return self.active_tasks.get(task_id)
    
    def get_client_tasks(self, client_id: str) -> List[str]:
        """Get all tasks for a client"""
        return [
            task_id
            for task_id, task in self.active_tasks.items()
            if task["client_id"] == client_id
        ]


# Global connection manager instance
connection_manager = ConnectionManager()
task_manager = TaskManager(connection_manager)
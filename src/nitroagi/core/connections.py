"""
Advanced Connection System for NitroAGI NEXUS
Implements brain-inspired multi-layer connection architecture
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple
import logging
from multiprocessing import shared_memory
import pickle

import aioredis
from nitroagi.utils.logging import get_logger


class ConnectionType(Enum):
    """Types of connections available in the system."""
    SHARED_MEMORY = "shared_memory"      # < 1ms - Ultra-fast local
    GRPC_DIRECT = "grpc_direct"          # 1-5ms - Direct RPC
    WEBSOCKET = "websocket"              # 5-20ms - Streaming
    EVENT_BUS = "event_bus"              # 10-50ms - Pub/Sub
    MESSAGE_QUEUE = "message_queue"      # 50ms+ - Guaranteed delivery


class ConnectionPriority(Enum):
    """Priority levels for connections."""
    CRITICAL = 0  # Executive control
    HIGH = 1      # Real-time processing
    NORMAL = 2    # Standard operations
    LOW = 3       # Background tasks


@dataclass
class ConnectionRequirements:
    """Requirements for a connection."""
    max_latency_ms: float = 100.0
    min_bandwidth_mbps: float = 10.0
    reliability: str = "normal"  # normal, high, critical, guaranteed
    data_size_bytes: int = 1024
    is_streaming: bool = False
    is_broadcast: bool = False
    priority: ConnectionPriority = ConnectionPriority.NORMAL


@dataclass
class ConnectionMetrics:
    """Metrics for connection performance."""
    latency_ms: float
    throughput_mbps: float
    success_rate: float
    error_count: int
    last_used: float
    total_messages: int


class Connection(ABC):
    """Abstract base class for all connection types."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(__name__)
        self.metrics = ConnectionMetrics(
            latency_ms=0,
            throughput_mbps=0,
            success_rate=1.0,
            error_count=0,
            last_used=time.time(),
            total_messages=0
        )
    
    @abstractmethod
    async def send(self, source: str, target: str, data: Any) -> bool:
        """Send data from source to target."""
        pass
    
    @abstractmethod
    async def receive(self, target: str) -> Optional[Any]:
        """Receive data for target."""
        pass
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Close connection."""
        pass
    
    def update_metrics(self, latency: float, success: bool, data_size: int):
        """Update connection metrics."""
        self.metrics.total_messages += 1
        self.metrics.last_used = time.time()
        
        # Update latency (moving average)
        alpha = 0.1  # Smoothing factor
        self.metrics.latency_ms = (
            alpha * latency + (1 - alpha) * self.metrics.latency_ms
        )
        
        # Update success rate
        if success:
            self.metrics.success_rate = (
                (self.metrics.success_rate * (self.metrics.total_messages - 1) + 1) /
                self.metrics.total_messages
            )
        else:
            self.metrics.error_count += 1
            self.metrics.success_rate = (
                (self.metrics.success_rate * (self.metrics.total_messages - 1)) /
                self.metrics.total_messages
            )
        
        # Calculate throughput
        if latency > 0:
            self.metrics.throughput_mbps = (data_size * 8) / (latency * 1000)


class SharedMemoryConnection(Connection):
    """Ultra-fast shared memory connection for local modules."""
    
    def __init__(self, name: str = "shared_memory"):
        super().__init__(name)
        self.memory_blocks: Dict[str, shared_memory.SharedMemory] = {}
        self.locks: Dict[str, asyncio.Lock] = {}
    
    async def connect(self) -> bool:
        """Initialize shared memory blocks."""
        try:
            # Create default shared memory block
            self.memory_blocks["default"] = shared_memory.SharedMemory(
                create=True, 
                size=1024 * 1024  # 1MB default
            )
            self.locks["default"] = asyncio.Lock()
            self.logger.info("Shared memory connection established")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create shared memory: {e}")
            return False
    
    async def send(self, source: str, target: str, data: Any) -> bool:
        """Send data through shared memory."""
        start_time = time.time()
        success = False
        
        try:
            # Get or create memory block for this channel
            channel = f"{source}_{target}"
            if channel not in self.memory_blocks:
                self.memory_blocks[channel] = shared_memory.SharedMemory(
                    create=True,
                    size=1024 * 1024
                )
                self.locks[channel] = asyncio.Lock()
            
            # Serialize and write data
            serialized = pickle.dumps(data)
            
            async with self.locks[channel]:
                mem = self.memory_blocks[channel]
                # Write size header (4 bytes) + data
                size_bytes = len(serialized).to_bytes(4, 'little')
                mem.buf[0:4] = size_bytes
                mem.buf[4:4+len(serialized)] = serialized
            
            success = True
            self.logger.debug(f"Sent {len(serialized)} bytes via shared memory")
            
        except Exception as e:
            self.logger.error(f"Shared memory send failed: {e}")
        
        finally:
            latency = (time.time() - start_time) * 1000
            self.update_metrics(latency, success, len(serialized) if success else 0)
        
        return success
    
    async def receive(self, target: str) -> Optional[Any]:
        """Receive data from shared memory."""
        try:
            # Find any channel ending with target
            for channel, mem in self.memory_blocks.items():
                if channel.endswith(target):
                    async with self.locks[channel]:
                        # Read size header
                        size = int.from_bytes(mem.buf[0:4], 'little')
                        if size > 0:
                            # Read and deserialize data
                            data = pickle.loads(mem.buf[4:4+size])
                            # Clear the buffer
                            mem.buf[0:4] = b'\x00\x00\x00\x00'
                            return data
            return None
            
        except Exception as e:
            self.logger.error(f"Shared memory receive failed: {e}")
            return None
    
    async def disconnect(self) -> bool:
        """Clean up shared memory blocks."""
        try:
            for mem in self.memory_blocks.values():
                mem.close()
                mem.unlink()
            self.memory_blocks.clear()
            self.locks.clear()
            return True
        except Exception as e:
            self.logger.error(f"Failed to cleanup shared memory: {e}")
            return False


class EventBusConnection(Connection):
    """Redis-based event bus for pub/sub communication."""
    
    def __init__(self, name: str = "event_bus", redis_url: str = "redis://localhost"):
        super().__init__(name)
        self.redis_url = redis_url
        self.redis = None
        self.subscriptions: Dict[str, List[Callable]] = {}
    
    async def connect(self) -> bool:
        """Connect to Redis."""
        try:
            self.redis = await aioredis.create_redis_pool(self.redis_url)
            self.logger.info("Event bus connection established")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            return False
    
    async def send(self, source: str, target: str, data: Any) -> bool:
        """Publish event to the bus."""
        start_time = time.time()
        success = False
        
        try:
            channel = target if target != "*" else "broadcast"
            message = {
                "source": source,
                "target": target,
                "data": data,
                "timestamp": time.time()
            }
            
            serialized = json.dumps(message, default=str)
            await self.redis.publish(channel, serialized)
            
            success = True
            self.logger.debug(f"Published event to {channel}")
            
        except Exception as e:
            self.logger.error(f"Event publish failed: {e}")
        
        finally:
            latency = (time.time() - start_time) * 1000
            self.update_metrics(latency, success, len(serialized) if success else 0)
        
        return success
    
    async def receive(self, target: str) -> Optional[Any]:
        """Subscribe to events for target."""
        try:
            channel = self.redis.pubsub()
            await channel.subscribe(target)
            
            # Non-blocking receive
            message = await channel.get_message(ignore_subscribe_messages=True)
            if message:
                data = json.loads(message['data'])
                return data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Event receive failed: {e}")
            return None
    
    async def disconnect(self) -> bool:
        """Close Redis connection."""
        try:
            if self.redis:
                self.redis.close()
                await self.redis.wait_closed()
            return True
        except Exception as e:
            self.logger.error(f"Failed to disconnect from Redis: {e}")
            return False


class ConnectionManager:
    """
    Intelligent connection manager that chooses optimal connection types
    based on requirements and implements neural plasticity.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.connections: Dict[ConnectionType, Connection] = {}
        self.connection_weights: Dict[str, float] = {}  # Neural plasticity
        self.routing_table: Dict[Tuple[str, str], ConnectionType] = {}
        
        # Initialize connections
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize all connection types."""
        self.connections[ConnectionType.SHARED_MEMORY] = SharedMemoryConnection()
        self.connections[ConnectionType.EVENT_BUS] = EventBusConnection()
        # Add more connection types as implemented
    
    async def start(self):
        """Start all connections."""
        self.logger.info("Starting connection manager...")
        
        for conn_type, connection in self.connections.items():
            success = await connection.connect()
            if success:
                self.logger.info(f"{conn_type.value} connection established")
            else:
                self.logger.warning(f"{conn_type.value} connection failed")
    
    async def stop(self):
        """Stop all connections."""
        self.logger.info("Stopping connection manager...")
        
        for connection in self.connections.values():
            await connection.disconnect()
    
    def select_connection(self, requirements: ConnectionRequirements) -> ConnectionType:
        """
        Select optimal connection type based on requirements.
        Implements intelligent routing decision tree.
        """
        latency = requirements.max_latency_ms
        reliability = requirements.reliability
        data_size = requirements.data_size_bytes
        
        # Ultra-low latency required
        if latency < 1:
            if data_size < 1024 * 100:  # < 100KB
                return ConnectionType.SHARED_MEMORY
            else:
                return ConnectionType.GRPC_DIRECT
        
        # Low latency required
        elif latency < 10:
            if reliability in ["critical", "guaranteed"]:
                return ConnectionType.GRPC_DIRECT
            else:
                return ConnectionType.WEBSOCKET
        
        # Standard latency
        elif latency < 100:
            if requirements.is_broadcast:
                return ConnectionType.EVENT_BUS
            elif requirements.is_streaming:
                return ConnectionType.WEBSOCKET
            else:
                return ConnectionType.EVENT_BUS
        
        # Can tolerate higher latency
        else:
            if reliability == "guaranteed":
                return ConnectionType.MESSAGE_QUEUE
            else:
                return ConnectionType.EVENT_BUS
    
    async def send(self, 
                   source: str, 
                   target: str, 
                   data: Any,
                   requirements: Optional[ConnectionRequirements] = None) -> bool:
        """
        Send data using optimal connection type.
        Implements neural plasticity for adaptive routing.
        """
        if requirements is None:
            requirements = ConnectionRequirements()
        
        # Check if we have a learned route
        route_key = f"{source}→{target}"
        
        # Get connection weight (neural plasticity)
        weight = self.connection_weights.get(route_key, 0.5)
        
        # Select connection based on requirements and learned weights
        if weight > 0.8 and route_key in self.routing_table:
            # Use previously successful connection
            conn_type = self.routing_table[route_key]
        else:
            # Select new connection
            conn_type = self.select_connection(requirements)
        
        # Get connection
        connection = self.connections.get(conn_type)
        if not connection:
            self.logger.error(f"Connection type {conn_type} not available")
            return False
        
        # Send data
        success = await connection.send(source, target, data)
        
        # Update neural plasticity weights
        self._update_connection_weight(route_key, conn_type, success)
        
        return success
    
    def _update_connection_weight(self, 
                                  route_key: str, 
                                  conn_type: ConnectionType,
                                  success: bool):
        """
        Update connection weights based on success/failure.
        Implements Hebbian learning: strengthen successful connections.
        """
        current_weight = self.connection_weights.get(route_key, 0.5)
        
        if success:
            # Strengthen connection
            new_weight = min(1.0, current_weight + 0.1)
            # Remember successful route
            self.routing_table[route_key] = conn_type
        else:
            # Weaken connection
            new_weight = max(0.0, current_weight - 0.2)
            # Forget failed route
            if route_key in self.routing_table:
                del self.routing_table[route_key]
        
        self.connection_weights[route_key] = new_weight
        
        self.logger.debug(
            f"Updated weight for {route_key}: {current_weight:.2f} → {new_weight:.2f}"
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for all connections."""
        metrics = {}
        
        for conn_type, connection in self.connections.items():
            metrics[conn_type.value] = {
                "latency_ms": connection.metrics.latency_ms,
                "throughput_mbps": connection.metrics.throughput_mbps,
                "success_rate": connection.metrics.success_rate,
                "error_count": connection.metrics.error_count,
                "total_messages": connection.metrics.total_messages
            }
        
        # Add neural plasticity metrics
        metrics["neural_plasticity"] = {
            "learned_routes": len(self.routing_table),
            "average_weight": sum(self.connection_weights.values()) / len(self.connection_weights) if self.connection_weights else 0,
            "strongest_connections": sorted(
                self.connection_weights.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
        
        return metrics


class ModuleConnector:
    """
    High-level interface for modules to connect with each other.
    Provides simple API while handling complex routing underneath.
    """
    
    def __init__(self, module_name: str, connection_manager: ConnectionManager):
        self.module_name = module_name
        self.connection_manager = connection_manager
        self.logger = get_logger(__name__)
    
    async def call(self, 
                   target: str, 
                   data: Any,
                   timeout: float = 5.0,
                   priority: ConnectionPriority = ConnectionPriority.NORMAL) -> Any:
        """
        Call another module and wait for response.
        High-level API that handles all complexity.
        """
        requirements = ConnectionRequirements(
            max_latency_ms=timeout * 1000,
            priority=priority,
            reliability="high" if priority <= ConnectionPriority.HIGH else "normal"
        )
        
        # Send request
        success = await self.connection_manager.send(
            self.module_name,
            target,
            data,
            requirements
        )
        
        if not success:
            raise ConnectionError(f"Failed to send to {target}")
        
        # Wait for response (simplified - real implementation would use correlation IDs)
        # This is a placeholder for proper request-response correlation
        return {"status": "success", "data": data}
    
    async def broadcast(self, event: Dict[str, Any]) -> bool:
        """Broadcast event to all interested modules."""
        requirements = ConnectionRequirements(
            is_broadcast=True,
            reliability="normal"
        )
        
        return await self.connection_manager.send(
            self.module_name,
            "*",  # Broadcast target
            event,
            requirements
        )
    
    async def stream(self, 
                    target: str, 
                    data_generator: AsyncIterator[Any]) -> bool:
        """Stream data to another module."""
        requirements = ConnectionRequirements(
            is_streaming=True,
            reliability="normal"
        )
        
        async for data in data_generator:
            success = await self.connection_manager.send(
                self.module_name,
                target,
                data,
                requirements
            )
            
            if not success:
                return False
        
        return True
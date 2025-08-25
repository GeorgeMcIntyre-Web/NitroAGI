"""Memory system for NitroAGI with multi-tier storage."""

import asyncio
import json
import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4

import redis.asyncio as redis
import numpy as np
from pydantic import BaseModel, Field

from nitroagi.core.exceptions import MemoryException


class MemoryType(Enum):
    """Types of memory in the system."""
    WORKING = "working"  # Short-term, fast access
    EPISODIC = "episodic"  # Event-based memories
    SEMANTIC = "semantic"  # Facts and knowledge
    PROCEDURAL = "procedural"  # Skills and procedures
    SENSORY = "sensory"  # Raw sensory data (very short-term)


class MemoryPriority(Enum):
    """Priority levels for memory storage."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: UUID = field(default_factory=uuid4)
    key: str = ""
    value: Any = None
    memory_type: MemoryType = MemoryType.WORKING
    priority: MemoryPriority = MemoryPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.utcnow)
    accessed_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    associations: Set[UUID] = field(default_factory=set)
    
    def is_expired(self) -> bool:
        """Check if the memory has expired."""
        if self.ttl_seconds:
            expiry_time = self.created_at + timedelta(seconds=self.ttl_seconds)
            return datetime.utcnow() > expiry_time
        return False
    
    def update_access(self) -> None:
        """Update access timestamp and count."""
        self.accessed_at = datetime.utcnow()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": str(self.id),
            "key": self.key,
            "value": self.value,
            "memory_type": self.memory_type.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "ttl_seconds": self.ttl_seconds,
            "metadata": self.metadata,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "associations": [str(aid) for aid in self.associations]
        }


class MemoryStore(ABC):
    """Abstract base class for memory storage backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from memory."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in memory."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a value from memory."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all memory."""
        pass


class RedisMemoryStore(MemoryStore):
    """Redis-based memory storage."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", prefix: str = "nitroagi"):
        """Initialize Redis memory store.
        
        Args:
            redis_url: Redis connection URL
            prefix: Key prefix for namespacing
        """
        self.redis_url = redis_url
        self.prefix = prefix
        self.client: Optional[redis.Redis] = None
        self.logger = logging.getLogger("nitroagi.memory.redis")
    
    async def connect(self) -> None:
        """Connect to Redis."""
        self.client = await redis.from_url(self.redis_url, decode_responses=False)
        await self.client.ping()
        self.logger.info("Connected to Redis")
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.client:
            await self.client.close()
            self.logger.info("Disconnected from Redis")
    
    def _make_key(self, key: str) -> str:
        """Create a namespaced key."""
        return f"{self.prefix}:{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from Redis."""
        if not self.client:
            raise MemoryException("Redis client not connected")
        
        full_key = self._make_key(key)
        value = await self.client.get(full_key)
        
        if value:
            try:
                return pickle.loads(value)
            except Exception as e:
                self.logger.error(f"Error deserializing value for key {key}: {e}")
                return None
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in Redis."""
        if not self.client:
            raise MemoryException("Redis client not connected")
        
        full_key = self._make_key(key)
        
        try:
            serialized = pickle.dumps(value)
            if ttl:
                await self.client.setex(full_key, ttl, serialized)
            else:
                await self.client.set(full_key, serialized)
            return True
        except Exception as e:
            self.logger.error(f"Error setting value for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete a value from Redis."""
        if not self.client:
            raise MemoryException("Redis client not connected")
        
        full_key = self._make_key(key)
        result = await self.client.delete(full_key)
        return result > 0
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in Redis."""
        if not self.client:
            raise MemoryException("Redis client not connected")
        
        full_key = self._make_key(key)
        return await self.client.exists(full_key) > 0
    
    async def clear(self) -> None:
        """Clear all keys with our prefix."""
        if not self.client:
            raise MemoryException("Redis client not connected")
        
        pattern = f"{self.prefix}:*"
        cursor = 0
        
        while True:
            cursor, keys = await self.client.scan(cursor, match=pattern, count=100)
            if keys:
                await self.client.delete(*keys)
            if cursor == 0:
                break


class MemoryManager:
    """Main memory management system for NitroAGI."""
    
    def __init__(
        self,
        working_memory_size: int = 1000,
        episodic_memory_size: int = 10000,
        semantic_memory_size: int = 100000
    ):
        """Initialize the memory manager.
        
        Args:
            working_memory_size: Maximum size of working memory
            episodic_memory_size: Maximum size of episodic memory
            semantic_memory_size: Maximum size of semantic memory
        """
        self.logger = logging.getLogger("nitroagi.memory.manager")
        
        # Memory stores
        self.working_memory: Dict[str, MemoryEntry] = {}
        self.episodic_memory: Dict[str, MemoryEntry] = {}
        self.semantic_memory: Dict[str, MemoryEntry] = {}
        
        # Memory limits
        self.limits = {
            MemoryType.WORKING: working_memory_size,
            MemoryType.EPISODIC: episodic_memory_size,
            MemoryType.SEMANTIC: semantic_memory_size,
        }
        
        # Redis stores for persistence
        self.redis_stores: Dict[MemoryType, RedisMemoryStore] = {}
        
        # Memory indices for fast lookup
        self.type_index: Dict[MemoryType, Set[str]] = {
            memory_type: set() for memory_type in MemoryType
        }
        
        # Metrics
        self.metrics = {
            "total_memories": 0,
            "memory_hits": 0,
            "memory_misses": 0,
            "evictions": 0,
        }
        
        self._lock = asyncio.Lock()
    
    async def initialize(self, redis_url: Optional[str] = None) -> None:
        """Initialize the memory system.
        
        Args:
            redis_url: Optional Redis connection URL
        """
        if redis_url:
            # Initialize Redis stores for each memory type
            for memory_type in [MemoryType.WORKING, MemoryType.EPISODIC, MemoryType.SEMANTIC]:
                store = RedisMemoryStore(
                    redis_url,
                    prefix=f"nitroagi:{memory_type.value}"
                )
                await store.connect()
                self.redis_stores[memory_type] = store
        
        self.logger.info("Memory manager initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the memory system."""
        # Disconnect from Redis stores
        for store in self.redis_stores.values():
            await store.disconnect()
        
        self.logger.info("Memory manager shutdown")
    
    async def store(
        self,
        key: str,
        value: Any,
        memory_type: MemoryType = MemoryType.WORKING,
        priority: MemoryPriority = MemoryPriority.NORMAL,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[np.ndarray] = None
    ) -> UUID:
        """Store a memory entry.
        
        Args:
            key: Key for the memory
            value: Value to store
            memory_type: Type of memory
            priority: Priority of the memory
            ttl_seconds: Time to live in seconds
            metadata: Additional metadata
            embedding: Vector embedding for semantic search
            
        Returns:
            UUID of the stored memory
        """
        async with self._lock:
            # Create memory entry
            entry = MemoryEntry(
                key=key,
                value=value,
                memory_type=memory_type,
                priority=priority,
                ttl_seconds=ttl_seconds,
                metadata=metadata or {},
                embedding=embedding
            )
            
            # Check memory limits and evict if necessary
            await self._enforce_memory_limits(memory_type, priority)
            
            # Store in appropriate memory
            memory_store = self._get_memory_store(memory_type)
            memory_store[key] = entry
            
            # Update indices
            self.type_index[memory_type].add(key)
            
            # Store in Redis if available
            if memory_type in self.redis_stores:
                await self.redis_stores[memory_type].set(
                    key,
                    entry.to_dict(),
                    ttl=ttl_seconds
                )
            
            # Update metrics
            self.metrics["total_memories"] += 1
            
            self.logger.debug(f"Stored memory: {key} ({memory_type.value})")
            return entry.id
    
    async def retrieve(
        self,
        key: str,
        memory_type: Optional[MemoryType] = None
    ) -> Optional[Any]:
        """Retrieve a memory entry.
        
        Args:
            key: Key to retrieve
            memory_type: Optional memory type to search in
            
        Returns:
            Value if found, None otherwise
        """
        async with self._lock:
            # Search in specified memory type or all types
            memory_types = [memory_type] if memory_type else list(MemoryType)
            
            for mem_type in memory_types:
                memory_store = self._get_memory_store(mem_type)
                
                if key in memory_store:
                    entry = memory_store[key]
                    
                    # Check if expired
                    if entry.is_expired():
                        await self._remove_memory(key, mem_type)
                        continue
                    
                    # Update access info
                    entry.update_access()
                    
                    # Update metrics
                    self.metrics["memory_hits"] += 1
                    
                    self.logger.debug(f"Retrieved memory: {key} ({mem_type.value})")
                    return entry.value
                
                # Check Redis if not in memory
                if mem_type in self.redis_stores:
                    redis_value = await self.redis_stores[mem_type].get(key)
                    if redis_value:
                        # Restore to memory
                        entry_dict = redis_value
                        entry = self._dict_to_entry(entry_dict)
                        memory_store[key] = entry
                        self.type_index[mem_type].add(key)
                        
                        self.metrics["memory_hits"] += 1
                        return entry.value
            
            self.metrics["memory_misses"] += 1
            return None
    
    async def delete(self, key: str, memory_type: Optional[MemoryType] = None) -> bool:
        """Delete a memory entry.
        
        Args:
            key: Key to delete
            memory_type: Optional memory type to delete from
            
        Returns:
            True if deleted, False otherwise
        """
        async with self._lock:
            deleted = False
            memory_types = [memory_type] if memory_type else list(MemoryType)
            
            for mem_type in memory_types:
                if await self._remove_memory(key, mem_type):
                    deleted = True
            
            return deleted
    
    async def search(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """Search for memories.
        
        Args:
            query: Search query
            memory_type: Optional memory type to search in
            limit: Maximum number of results
            
        Returns:
            List of matching memory entries
        """
        results = []
        memory_types = [memory_type] if memory_type else list(MemoryType)
        
        for mem_type in memory_types:
            memory_store = self._get_memory_store(mem_type)
            
            for key, entry in memory_store.items():
                # Simple string matching - can be enhanced with embeddings
                if query.lower() in key.lower() or query.lower() in str(entry.value).lower():
                    results.append(entry)
                    
                    if len(results) >= limit:
                        break
            
            if len(results) >= limit:
                break
        
        return results[:limit]
    
    async def consolidate(self) -> None:
        """Consolidate memories by moving important ones to long-term storage."""
        async with self._lock:
            # Move frequently accessed working memories to episodic
            for key, entry in list(self.working_memory.items()):
                if entry.access_count > 5 and entry.priority >= MemoryPriority.NORMAL:
                    # Move to episodic memory
                    self.episodic_memory[key] = entry
                    self.type_index[MemoryType.EPISODIC].add(key)
                    del self.working_memory[key]
                    self.type_index[MemoryType.WORKING].discard(key)
                    
                    entry.memory_type = MemoryType.EPISODIC
                    
                    self.logger.debug(f"Consolidated memory {key} to episodic")
            
            # Move very frequently accessed episodic memories to semantic
            for key, entry in list(self.episodic_memory.items()):
                if entry.access_count > 20 and entry.priority >= MemoryPriority.HIGH:
                    # Move to semantic memory
                    self.semantic_memory[key] = entry
                    self.type_index[MemoryType.SEMANTIC].add(key)
                    del self.episodic_memory[key]
                    self.type_index[MemoryType.EPISODIC].discard(key)
                    
                    entry.memory_type = MemoryType.SEMANTIC
                    
                    self.logger.debug(f"Consolidated memory {key} to semantic")
    
    def _get_memory_store(self, memory_type: MemoryType) -> Dict[str, MemoryEntry]:
        """Get the appropriate memory store for a type."""
        if memory_type == MemoryType.WORKING:
            return self.working_memory
        elif memory_type == MemoryType.EPISODIC:
            return self.episodic_memory
        elif memory_type == MemoryType.SEMANTIC:
            return self.semantic_memory
        else:
            return self.working_memory
    
    async def _enforce_memory_limits(
        self,
        memory_type: MemoryType,
        new_priority: MemoryPriority
    ) -> None:
        """Enforce memory limits by evicting low-priority memories."""
        memory_store = self._get_memory_store(memory_type)
        limit = self.limits.get(memory_type, 1000)
        
        if len(memory_store) >= limit:
            # Find memories to evict (lowest priority, least recently accessed)
            candidates = sorted(
                memory_store.items(),
                key=lambda x: (x[1].priority.value, x[1].accessed_at)
            )
            
            # Evict until we have space
            for key, entry in candidates:
                if entry.priority.value < new_priority.value:
                    await self._remove_memory(key, memory_type)
                    self.metrics["evictions"] += 1
                    
                    if len(memory_store) < limit:
                        break
    
    async def _remove_memory(self, key: str, memory_type: MemoryType) -> bool:
        """Remove a memory from storage."""
        memory_store = self._get_memory_store(memory_type)
        
        if key in memory_store:
            del memory_store[key]
            self.type_index[memory_type].discard(key)
            
            # Remove from Redis if available
            if memory_type in self.redis_stores:
                await self.redis_stores[memory_type].delete(key)
            
            self.metrics["total_memories"] -= 1
            return True
        
        return False
    
    def _dict_to_entry(self, data: Dict[str, Any]) -> MemoryEntry:
        """Convert dictionary to MemoryEntry."""
        entry = MemoryEntry(
            id=UUID(data["id"]),
            key=data["key"],
            value=data["value"],
            memory_type=MemoryType(data["memory_type"]),
            priority=MemoryPriority(data["priority"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            accessed_at=datetime.fromisoformat(data["accessed_at"]),
            access_count=data["access_count"],
            ttl_seconds=data.get("ttl_seconds"),
            metadata=data.get("metadata", {}),
            embedding=np.array(data["embedding"]) if data.get("embedding") else None,
            associations=set(UUID(aid) for aid in data.get("associations", []))
        )
        return entry
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get memory system metrics."""
        return {
            **self.metrics,
            "working_memory_size": len(self.working_memory),
            "episodic_memory_size": len(self.episodic_memory),
            "semantic_memory_size": len(self.semantic_memory),
            "total_capacity": sum(self.limits.values()),
        }
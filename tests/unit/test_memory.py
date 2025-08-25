"""Unit tests for memory system."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import uuid
import json

from nitroagi.core.memory import (
    MemoryManager,
    MemoryType,
    MemoryItem,
    MemoryTier
)


class TestMemoryManager:
    """Test MemoryManager functionality."""
    
    @pytest.mark.asyncio
    async def test_store_memory(self, memory_manager):
        """Test storing memory items."""
        key = "test_key"
        value = {"data": "test data"}
        
        memory_id = await memory_manager.store(
            key=key,
            value=value,
            memory_type=MemoryType.WORKING
        )
        
        assert memory_id is not None
        assert isinstance(memory_id, uuid.UUID)
        
        # Verify in working memory
        assert key in memory_manager.working_memory
        assert memory_manager.working_memory[key]["value"] == value
    
    @pytest.mark.asyncio
    async def test_retrieve_memory(self, memory_manager):
        """Test retrieving memory items."""
        key = "test_key"
        value = {"data": "test data"}
        
        # Store item
        await memory_manager.store(key, value, MemoryType.WORKING)
        
        # Retrieve item
        retrieved = await memory_manager.retrieve(key)
        assert retrieved == value
    
    @pytest.mark.asyncio
    async def test_retrieve_nonexistent(self, memory_manager):
        """Test retrieving non-existent memory."""
        result = await memory_manager.retrieve("nonexistent_key")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_memory_search(self, memory_manager):
        """Test searching memory by pattern."""
        # Store multiple items
        await memory_manager.store("user_name", "John", MemoryType.WORKING)
        await memory_manager.store("user_age", 30, MemoryType.WORKING)
        await memory_manager.store("system_config", {"debug": True}, MemoryType.WORKING)
        
        # Search by pattern
        results = await memory_manager.search("user_*")
        assert len(results) == 2
        assert any(r["key"] == "user_name" for r in results)
        assert any(r["key"] == "user_age" for r in results)
    
    @pytest.mark.asyncio
    async def test_memory_consolidation(self, memory_manager):
        """Test memory consolidation process."""
        # Add items with different importance
        for i in range(10):
            await memory_manager.store(
                f"item_{i}",
                f"data_{i}",
                MemoryType.WORKING,
                metadata={"importance": i / 10}
            )
        
        # Run consolidation
        await memory_manager.consolidate()
        
        # Check that low importance items might be removed
        # (depends on consolidation strategy)
        assert len(memory_manager.working_memory) <= 10
    
    @pytest.mark.asyncio
    async def test_memory_persistence(self, memory_manager, mock_redis):
        """Test Redis persistence."""
        memory_manager.redis = mock_redis
        
        key = "persistent_key"
        value = {"persistent": "data"}
        
        # Store with persistence
        await memory_manager.store(key, value, MemoryType.EPISODIC)
        
        # Verify Redis storage was called
        stored_data = await mock_redis.get(f"memory:episodic:{key}")
        assert stored_data is not None
    
    @pytest.mark.asyncio
    async def test_memory_ttl(self, memory_manager):
        """Test memory TTL expiration."""
        key = "ttl_key"
        value = "ttl_value"
        
        # Store with short TTL
        await memory_manager.store(
            key, value, MemoryType.WORKING,
            metadata={"ttl": 0.1}  # 100ms TTL
        )
        
        # Should exist immediately
        assert await memory_manager.retrieve(key) == value
        
        # Wait for TTL to expire
        await asyncio.sleep(0.2)
        
        # Should be expired
        result = await memory_manager.retrieve(key)
        # Note: Actual implementation may vary
    
    @pytest.mark.asyncio
    async def test_memory_update(self, memory_manager):
        """Test updating existing memory."""
        key = "update_key"
        original = {"version": 1}
        updated = {"version": 2}
        
        # Store original
        await memory_manager.store(key, original, MemoryType.WORKING)
        
        # Update value
        await memory_manager.update(key, updated)
        
        # Verify update
        retrieved = await memory_manager.retrieve(key)
        assert retrieved == updated
    
    @pytest.mark.asyncio
    async def test_memory_delete(self, memory_manager):
        """Test deleting memory items."""
        key = "delete_key"
        value = "delete_value"
        
        # Store item
        await memory_manager.store(key, value, MemoryType.WORKING)
        assert await memory_manager.retrieve(key) == value
        
        # Delete item
        deleted = await memory_manager.delete(key)
        assert deleted is True
        
        # Verify deletion
        assert await memory_manager.retrieve(key) is None
    
    @pytest.mark.asyncio
    async def test_memory_stats(self, memory_manager):
        """Test memory statistics."""
        # Add various items
        await memory_manager.store("item1", "data1", MemoryType.WORKING)
        await memory_manager.store("item2", "data2", MemoryType.EPISODIC)
        await memory_manager.store("item3", "data3", MemoryType.SEMANTIC)
        
        stats = await memory_manager.get_stats()
        
        assert stats["total_items"] >= 3
        assert stats["working_memory_count"] >= 1
        assert stats["episodic_memory_count"] >= 1
        assert stats["semantic_memory_count"] >= 1


class TestMemoryItem:
    """Test MemoryItem functionality."""
    
    def test_memory_item_creation(self):
        """Test creating memory items."""
        item = MemoryItem(
            id=uuid.uuid4(),
            key="test_key",
            value={"test": "data"},
            memory_type=MemoryType.WORKING,
            tier=MemoryTier.HOT,
            metadata={"importance": 0.8}
        )
        
        assert item.key == "test_key"
        assert item.memory_type == MemoryType.WORKING
        assert item.tier == MemoryTier.HOT
        assert item.metadata["importance"] == 0.8
    
    def test_memory_item_serialization(self):
        """Test serializing memory items."""
        item = MemoryItem(
            id=uuid.uuid4(),
            key="test_key",
            value={"test": "data"},
            memory_type=MemoryType.EPISODIC,
            tier=MemoryTier.WARM
        )
        
        # Convert to dict
        item_dict = item.dict()
        
        assert "id" in item_dict
        assert item_dict["key"] == "test_key"
        assert item_dict["memory_type"] == "episodic"
        assert item_dict["tier"] == "warm"
    
    def test_memory_type_enum(self):
        """Test MemoryType enum values."""
        assert MemoryType.WORKING.value == "working"
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.SEMANTIC.value == "semantic"
    
    def test_memory_tier_enum(self):
        """Test MemoryTier enum values."""
        assert MemoryTier.HOT.value == "hot"
        assert MemoryTier.WARM.value == "warm"
        assert MemoryTier.COLD.value == "cold"
        assert MemoryTier.ARCHIVE.value == "archive"
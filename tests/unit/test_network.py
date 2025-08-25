"""Unit tests for network module."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import time

from nitroagi.core.network import (
    NetworkProfile,
    NetworkOptimizer,
    NetworkManager,
    NETWORK_PROFILES,
    ConnectionType,
    NetworkMetrics
)


class TestNetworkProfile:
    """Test NetworkProfile functionality."""
    
    def test_profile_creation(self):
        """Test creating network profiles."""
        profile = NetworkProfile(
            name="test_profile",
            max_latency_ms=5.0,
            min_bandwidth_mbps=1000,
            packet_loss_threshold=0.01,
            jitter_ms=1.0,
            priority=10
        )
        
        assert profile.name == "test_profile"
        assert profile.max_latency_ms == 5.0
        assert profile.min_bandwidth_mbps == 1000
        assert profile.packet_loss_threshold == 0.01
        assert profile.priority == 10
    
    def test_predefined_profiles(self):
        """Test predefined network profiles."""
        # Test 6G profiles
        assert "real_time_ai" in NETWORK_PROFILES
        assert "holographic_communication" in NETWORK_PROFILES
        assert "brain_computer_interface" in NETWORK_PROFILES
        
        # Check ultra-low latency profile
        real_time = NETWORK_PROFILES["real_time_ai"]
        assert real_time.max_latency_ms <= 1.0
        assert real_time.min_bandwidth_mbps >= 10000
    
    def test_profile_validation(self):
        """Test profile validation."""
        # Valid profile
        profile = NetworkProfile(
            name="valid",
            max_latency_ms=10.0,
            min_bandwidth_mbps=100
        )
        assert profile.is_valid()
        
        # Invalid profile (negative latency)
        with pytest.raises(ValueError):
            NetworkProfile(
                name="invalid",
                max_latency_ms=-1.0,
                min_bandwidth_mbps=100
            )


class TestNetworkOptimizer:
    """Test NetworkOptimizer functionality."""
    
    @pytest.mark.asyncio
    async def test_optimizer_initialization(self):
        """Test network optimizer initialization."""
        optimizer = NetworkOptimizer()
        await optimizer.initialize()
        
        assert optimizer.initialized
        assert optimizer.current_profile is not None
    
    @pytest.mark.asyncio
    async def test_profile_selection(self):
        """Test automatic profile selection."""
        optimizer = NetworkOptimizer()
        await optimizer.initialize()
        
        # Test with different network conditions
        metrics = NetworkMetrics(
            latency_ms=0.5,
            bandwidth_mbps=100000,
            packet_loss=0.0,
            jitter_ms=0.1
        )
        
        profile = await optimizer.select_profile(metrics)
        assert profile.name in ["real_time_ai", "holographic_communication"]
    
    @pytest.mark.asyncio
    async def test_connection_optimization(self):
        """Test connection optimization."""
        optimizer = NetworkOptimizer()
        await optimizer.initialize()
        
        connection = await optimizer.optimize_connection(
            connection_type=ConnectionType.ULTRA_LOW_LATENCY,
            target_latency_ms=0.5
        )
        
        assert connection is not None
        assert connection["optimized"] is True
        assert connection["latency_ms"] <= 0.5
    
    @pytest.mark.asyncio
    async def test_adaptive_optimization(self):
        """Test adaptive network optimization."""
        optimizer = NetworkOptimizer()
        optimizer.enable_adaptive = True
        await optimizer.initialize()
        
        # Simulate changing network conditions
        for latency in [10.0, 5.0, 1.0, 0.5]:
            metrics = NetworkMetrics(
                latency_ms=latency,
                bandwidth_mbps=10000 * (11 - latency),
                packet_loss=0.001 * latency,
                jitter_ms=latency / 10
            )
            
            await optimizer.adapt_to_conditions(metrics)
            
            assert optimizer.current_profile is not None
            assert optimizer.current_profile.max_latency_ms >= latency
    
    @pytest.mark.asyncio
    async def test_qos_management(self):
        """Test Quality of Service management."""
        optimizer = NetworkOptimizer()
        await optimizer.initialize()
        
        qos_params = await optimizer.configure_qos(
            priority=10,
            guaranteed_bandwidth_mbps=1000,
            max_latency_ms=1.0
        )
        
        assert qos_params["priority"] == 10
        assert qos_params["guaranteed_bandwidth"] >= 1000
        assert qos_params["max_latency"] <= 1.0


class TestNetworkManager:
    """Test NetworkManager functionality."""
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self):
        """Test network manager initialization."""
        manager = NetworkManager()
        await manager.start()
        
        assert manager.is_running
        assert manager.optimizer is not None
        
        await manager.stop()
        assert not manager.is_running
    
    @pytest.mark.asyncio
    async def test_connection_pool(self):
        """Test connection pool management."""
        manager = NetworkManager()
        await manager.start()
        
        # Create multiple connections
        connections = []
        for i in range(5):
            conn = await manager.get_connection(
                endpoint=f"service_{i}",
                connection_type=ConnectionType.STANDARD
            )
            connections.append(conn)
        
        assert len(manager.connection_pool) == 5
        
        # Release connections
        for conn in connections:
            await manager.release_connection(conn)
        
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_bandwidth_allocation(self):
        """Test bandwidth allocation."""
        manager = NetworkManager()
        await manager.start()
        
        # Allocate bandwidth for different services
        allocations = [
            ("ai_service", 10000),
            ("video_stream", 50000),
            ("data_sync", 5000)
        ]
        
        for service, bandwidth in allocations:
            allocated = await manager.allocate_bandwidth(
                service_name=service,
                required_mbps=bandwidth
            )
            assert allocated >= bandwidth * 0.9  # Allow 10% variance
        
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_latency_monitoring(self):
        """Test latency monitoring."""
        manager = NetworkManager()
        await manager.start()
        
        # Start monitoring
        monitor_task = asyncio.create_task(
            manager.monitor_latency("test_endpoint", interval=0.1)
        )
        
        # Let it run for a bit
        await asyncio.sleep(0.5)
        
        # Check metrics
        metrics = manager.get_latency_metrics("test_endpoint")
        assert metrics is not None
        assert "average_ms" in metrics
        assert "min_ms" in metrics
        assert "max_ms" in metrics
        
        monitor_task.cancel()
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_6g_readiness(self):
        """Test 6G network readiness."""
        manager = NetworkManager()
        manager.enable_6g = True
        await manager.start()
        
        # Check 6G capabilities
        capabilities = await manager.check_6g_capabilities()
        
        assert capabilities["6g_ready"] is True
        assert capabilities["max_bandwidth_gbps"] >= 1  # Terabit speeds
        assert capabilities["min_latency_us"] <= 100  # Sub-millisecond
        assert "holographic" in capabilities["supported_features"]
        assert "brain_interface" in capabilities["supported_features"]
        
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_network_slicing(self):
        """Test network slicing for different services."""
        manager = NetworkManager()
        await manager.start()
        
        # Create network slices
        slices = [
            {"name": "ai_inference", "priority": 10, "latency_ms": 1.0},
            {"name": "bulk_transfer", "priority": 1, "bandwidth_mbps": 10000},
            {"name": "real_time_comm", "priority": 8, "latency_ms": 5.0}
        ]
        
        for slice_config in slices:
            slice_id = await manager.create_network_slice(**slice_config)
            assert slice_id is not None
            
            # Verify slice properties
            slice_info = await manager.get_slice_info(slice_id)
            assert slice_info["name"] == slice_config["name"]
            assert slice_info["active"] is True
        
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_edge_computing_integration(self):
        """Test edge computing integration."""
        manager = NetworkManager()
        await manager.start()
        
        # Find nearest edge node
        edge_node = await manager.find_nearest_edge_node(
            service_type="ai_inference",
            max_latency_ms=2.0
        )
        
        assert edge_node is not None
        assert edge_node["latency_ms"] <= 2.0
        assert edge_node["available_compute"] > 0
        
        # Deploy to edge
        deployment = await manager.deploy_to_edge(
            edge_node["id"],
            service_name="test_model",
            required_compute=100
        )
        
        assert deployment["status"] == "deployed"
        assert deployment["edge_node_id"] == edge_node["id"]
        
        await manager.stop()
"""Network configuration and 6G-ready communication layer for NitroAGI."""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import aiohttp
import websockets
from pydantic import BaseModel, Field

from nitroagi.core.exceptions import NitroAGIException


class NetworkGeneration(Enum):
    """Network generation types."""
    WIRED = "wired"
    WIFI = "wifi"
    G3 = "3g"
    G4 = "4g"
    G5 = "5g"
    G6 = "6g"  # Future-ready for 6G networks
    SATELLITE = "satellite"
    QUANTUM = "quantum"  # Future quantum networks


class NetworkCapability(Enum):
    """Network capabilities for different use cases."""
    LOW_LATENCY = "low_latency"  # < 1ms for 6G
    HIGH_BANDWIDTH = "high_bandwidth"  # Tbps for 6G
    MASSIVE_IOT = "massive_iot"  # Million devices per km²
    EDGE_COMPUTING = "edge_computing"
    NETWORK_SLICING = "network_slicing"
    AI_NATIVE = "ai_native"  # 6G AI-integrated networks
    HOLOGRAPHIC = "holographic"  # Holographic communications
    DIGITAL_TWIN = "digital_twin"  # Digital twin synchronization
    BRAIN_INTERFACE = "brain_interface"  # Neural interface support
    ENERGY_HARVESTING = "energy_harvesting"  # Self-powered nodes


@dataclass
class NetworkMetrics:
    """Network performance metrics."""
    latency_ms: float = 0.0
    bandwidth_mbps: float = 0.0
    packet_loss_rate: float = 0.0
    jitter_ms: float = 0.0
    throughput_mbps: float = 0.0
    connection_quality: float = 1.0  # 0-1 scale
    signal_strength_dbm: float = -50.0
    network_generation: NetworkGeneration = NetworkGeneration.G5
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def is_6g_ready(self) -> bool:
        """Check if metrics meet 6G requirements."""
        return (
            self.latency_ms < 1.0 and  # Sub-millisecond latency
            self.bandwidth_mbps > 100000 and  # 100+ Gbps
            self.packet_loss_rate < 0.00001 and  # Ultra-reliable
            self.jitter_ms < 0.1  # Ultra-stable
        )


class NetworkProfile(BaseModel):
    """Network profile for different scenarios."""
    name: str = Field(..., description="Profile name")
    min_bandwidth_mbps: float = Field(default=100.0, description="Minimum bandwidth required")
    max_latency_ms: float = Field(default=100.0, description="Maximum acceptable latency")
    max_packet_loss: float = Field(default=0.01, description="Maximum packet loss rate")
    required_capabilities: List[NetworkCapability] = Field(default_factory=list)
    preferred_generation: NetworkGeneration = Field(default=NetworkGeneration.G5)
    priority: int = Field(default=0, description="Priority level")
    
    class Config:
        use_enum_values = True


# Predefined network profiles
NETWORK_PROFILES = {
    "real_time_ai": NetworkProfile(
        name="Real-Time AI Processing",
        min_bandwidth_mbps=10000,  # 10 Gbps
        max_latency_ms=1.0,  # 1ms for real-time
        max_packet_loss=0.0001,
        required_capabilities=[
            NetworkCapability.LOW_LATENCY,
            NetworkCapability.HIGH_BANDWIDTH,
            NetworkCapability.AI_NATIVE
        ],
        preferred_generation=NetworkGeneration.G6,
        priority=10
    ),
    "holographic_communication": NetworkProfile(
        name="Holographic Communication",
        min_bandwidth_mbps=100000,  # 100 Gbps for holograms
        max_latency_ms=0.5,  # Sub-millisecond
        max_packet_loss=0.00001,
        required_capabilities=[
            NetworkCapability.HOLOGRAPHIC,
            NetworkCapability.LOW_LATENCY,
            NetworkCapability.HIGH_BANDWIDTH
        ],
        preferred_generation=NetworkGeneration.G6,
        priority=9
    ),
    "edge_ai": NetworkProfile(
        name="Edge AI Computing",
        min_bandwidth_mbps=1000,  # 1 Gbps
        max_latency_ms=5.0,  # 5ms for edge
        max_packet_loss=0.001,
        required_capabilities=[
            NetworkCapability.EDGE_COMPUTING,
            NetworkCapability.AI_NATIVE,
            NetworkCapability.NETWORK_SLICING
        ],
        preferred_generation=NetworkGeneration.G5,
        priority=7
    ),
    "brain_computer_interface": NetworkProfile(
        name="Brain-Computer Interface",
        min_bandwidth_mbps=50000,  # 50 Gbps for neural data
        max_latency_ms=0.1,  # 100 microseconds
        max_packet_loss=0.000001,  # Ultra-reliable
        required_capabilities=[
            NetworkCapability.BRAIN_INTERFACE,
            NetworkCapability.LOW_LATENCY,
            NetworkCapability.HIGH_BANDWIDTH
        ],
        preferred_generation=NetworkGeneration.G6,
        priority=10
    ),
    "standard": NetworkProfile(
        name="Standard Operations",
        min_bandwidth_mbps=100,
        max_latency_ms=50.0,
        max_packet_loss=0.01,
        required_capabilities=[],
        preferred_generation=NetworkGeneration.G5,
        priority=1
    )
}


class NetworkOptimizer:
    """Optimizer for network performance and 6G readiness."""
    
    def __init__(self):
        """Initialize the network optimizer."""
        self.logger = logging.getLogger("nitroagi.network.optimizer")
        self.current_metrics = NetworkMetrics()
        self.profile_history: List[Tuple[datetime, NetworkProfile]] = []
        self.optimization_enabled = True
        self._metrics_lock = asyncio.Lock()
    
    async def select_optimal_profile(
        self,
        task_requirements: Dict[str, Any],
        available_networks: List[NetworkMetrics]
    ) -> NetworkProfile:
        """Select the optimal network profile for a task.
        
        Args:
            task_requirements: Requirements for the task
            available_networks: Available network metrics
            
        Returns:
            Optimal network profile
        """
        # Determine required capabilities
        required_capabilities = []
        
        if task_requirements.get("real_time", False):
            required_capabilities.append(NetworkCapability.LOW_LATENCY)
        
        if task_requirements.get("bandwidth_intensive", False):
            required_capabilities.append(NetworkCapability.HIGH_BANDWIDTH)
        
        if task_requirements.get("ai_processing", False):
            required_capabilities.append(NetworkCapability.AI_NATIVE)
        
        if task_requirements.get("holographic", False):
            required_capabilities.append(NetworkCapability.HOLOGRAPHIC)
        
        # Find matching profiles
        matching_profiles = []
        for profile_name, profile in NETWORK_PROFILES.items():
            if all(cap in profile.required_capabilities for cap in required_capabilities):
                matching_profiles.append(profile)
        
        if not matching_profiles:
            return NETWORK_PROFILES["standard"]
        
        # Sort by priority and select best
        matching_profiles.sort(key=lambda p: p.priority, reverse=True)
        
        # Check if any network can support the profile
        selected_profile = matching_profiles[0]
        
        for network in available_networks:
            if self._network_supports_profile(network, selected_profile):
                self.logger.info(f"Selected network profile: {selected_profile.name}")
                return selected_profile
        
        # Fallback to standard if no network supports requirements
        self.logger.warning("No network supports requirements, falling back to standard")
        return NETWORK_PROFILES["standard"]
    
    def _network_supports_profile(
        self,
        network: NetworkMetrics,
        profile: NetworkProfile
    ) -> bool:
        """Check if a network supports a profile.
        
        Args:
            network: Network metrics
            profile: Network profile
            
        Returns:
            True if network supports profile
        """
        return (
            network.bandwidth_mbps >= profile.min_bandwidth_mbps and
            network.latency_ms <= profile.max_latency_ms and
            network.packet_loss_rate <= profile.max_packet_loss
        )
    
    async def optimize_for_6g(self, current_metrics: NetworkMetrics) -> Dict[str, Any]:
        """Optimize network settings for 6G readiness.
        
        Args:
            current_metrics: Current network metrics
            
        Returns:
            Optimization recommendations
        """
        async with self._metrics_lock:
            self.current_metrics = current_metrics
        
        recommendations = {
            "is_6g_ready": current_metrics.is_6g_ready(),
            "optimizations": [],
            "estimated_improvement": 0.0
        }
        
        # Latency optimization
        if current_metrics.latency_ms > 1.0:
            recommendations["optimizations"].append({
                "type": "latency",
                "action": "enable_edge_computing",
                "description": "Move processing to edge nodes",
                "expected_improvement_ms": current_metrics.latency_ms * 0.7
            })
        
        # Bandwidth optimization
        if current_metrics.bandwidth_mbps < 100000:
            recommendations["optimizations"].append({
                "type": "bandwidth",
                "action": "enable_carrier_aggregation",
                "description": "Aggregate multiple carriers for higher bandwidth",
                "expected_improvement_mbps": current_metrics.bandwidth_mbps * 2
            })
        
        # AI-native optimization
        recommendations["optimizations"].append({
            "type": "ai_native",
            "action": "enable_ai_routing",
            "description": "Use AI for intelligent packet routing",
            "expected_improvement_percent": 25
        })
        
        # Calculate estimated improvement
        if recommendations["optimizations"]:
            recommendations["estimated_improvement"] = min(
                len(recommendations["optimizations"]) * 25, 100
            )
        
        return recommendations


class NetworkManager:
    """Manager for network connections and 6G integration."""
    
    def __init__(self):
        """Initialize the network manager."""
        self.logger = logging.getLogger("nitroagi.network.manager")
        self.optimizer = NetworkOptimizer()
        self.active_connections: Dict[str, Any] = {}
        self.metrics_history: List[NetworkMetrics] = []
        self.max_history_size = 1000
        self._monitoring_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the network manager."""
        self._monitoring_task = asyncio.create_task(self._monitor_network())
        self.logger.info("Network manager started")
    
    async def stop(self) -> None:
        """Stop the network manager."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        for conn_id in list(self.active_connections.keys()):
            await self.close_connection(conn_id)
        
        self.logger.info("Network manager stopped")
    
    async def establish_connection(
        self,
        endpoint: str,
        profile: NetworkProfile,
        connection_type: str = "websocket"
    ) -> str:
        """Establish a network connection with specified profile.
        
        Args:
            endpoint: Connection endpoint
            profile: Network profile to use
            connection_type: Type of connection (websocket, http, etc.)
            
        Returns:
            Connection ID
        """
        conn_id = f"{connection_type}_{datetime.utcnow().timestamp()}"
        
        try:
            if connection_type == "websocket":
                conn = await websockets.connect(endpoint)
            elif connection_type == "http":
                conn = aiohttp.ClientSession()
            else:
                raise ValueError(f"Unknown connection type: {connection_type}")
            
            self.active_connections[conn_id] = {
                "connection": conn,
                "endpoint": endpoint,
                "profile": profile,
                "type": connection_type,
                "created_at": datetime.utcnow(),
                "metrics": NetworkMetrics()
            }
            
            self.logger.info(f"Established {connection_type} connection: {conn_id}")
            return conn_id
            
        except Exception as e:
            self.logger.error(f"Failed to establish connection: {e}")
            raise NitroAGIException(f"Connection failed: {e}")
    
    async def close_connection(self, conn_id: str) -> None:
        """Close a network connection.
        
        Args:
            conn_id: Connection ID to close
        """
        if conn_id in self.active_connections:
            conn_info = self.active_connections[conn_id]
            conn = conn_info["connection"]
            
            try:
                if conn_info["type"] == "websocket":
                    await conn.close()
                elif conn_info["type"] == "http":
                    await conn.close()
            except Exception as e:
                self.logger.error(f"Error closing connection {conn_id}: {e}")
            
            del self.active_connections[conn_id]
            self.logger.info(f"Closed connection: {conn_id}")
    
    async def send_data(
        self,
        conn_id: str,
        data: Any,
        optimize_for_6g: bool = True
    ) -> None:
        """Send data through a connection.
        
        Args:
            conn_id: Connection ID
            data: Data to send
            optimize_for_6g: Whether to apply 6G optimizations
        """
        if conn_id not in self.active_connections:
            raise NitroAGIException(f"Connection not found: {conn_id}")
        
        conn_info = self.active_connections[conn_id]
        conn = conn_info["connection"]
        
        # Apply 6G optimizations if enabled
        if optimize_for_6g:
            metrics = conn_info["metrics"]
            optimizations = await self.optimizer.optimize_for_6g(metrics)
            
            if optimizations["is_6g_ready"]:
                self.logger.debug("Using 6G-optimized transmission")
                # Apply optimizations (placeholder for actual implementation)
        
        try:
            if conn_info["type"] == "websocket":
                await conn.send(data)
            elif conn_info["type"] == "http":
                # For HTTP, we'd need to make a request
                pass
        except Exception as e:
            self.logger.error(f"Failed to send data on {conn_id}: {e}")
            raise
    
    async def _monitor_network(self) -> None:
        """Monitor network performance continuously."""
        while True:
            try:
                # Collect metrics for all connections
                for conn_id, conn_info in self.active_connections.items():
                    # Simulate metrics collection (would be real measurements)
                    metrics = NetworkMetrics(
                        latency_ms=0.5,  # Simulated
                        bandwidth_mbps=10000,  # Simulated
                        packet_loss_rate=0.0001,  # Simulated
                        network_generation=NetworkGeneration.G5
                    )
                    
                    conn_info["metrics"] = metrics
                    self._add_to_history(metrics)
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in network monitoring: {e}")
                await asyncio.sleep(10)
    
    def _add_to_history(self, metrics: NetworkMetrics) -> None:
        """Add metrics to history.
        
        Args:
            metrics: Network metrics to add
        """
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history.pop(0)
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics.
        
        Returns:
            Network statistics dictionary
        """
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-100:]  # Last 100 measurements
        
        avg_latency = sum(m.latency_ms for m in recent_metrics) / len(recent_metrics)
        avg_bandwidth = sum(m.bandwidth_mbps for m in recent_metrics) / len(recent_metrics)
        
        return {
            "active_connections": len(self.active_connections),
            "average_latency_ms": avg_latency,
            "average_bandwidth_mbps": avg_bandwidth,
            "6g_ready_percentage": sum(
                1 for m in recent_metrics if m.is_6g_ready()
            ) / len(recent_metrics) * 100,
            "network_generations": {
                gen.value: sum(1 for m in recent_metrics if m.network_generation == gen)
                for gen in NetworkGeneration
            }
        }
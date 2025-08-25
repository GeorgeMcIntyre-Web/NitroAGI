"""
Enhanced Executive Controller for NitroAGI NEXUS
Advanced task routing, resource allocation, and decision making
"""

import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import heapq
from collections import defaultdict

from nitroagi.core.base import (
    ModuleCapability,
    ModuleStatus,
    AIModule
)
from nitroagi.core.orchestrator import (
    TaskRequest,
    TaskResult,
    TaskStatus,
    ExecutionStrategy
)
from nitroagi.utils.logging import get_logger


class ResourceType(Enum):
    """Types of system resources."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    NETWORK = "network"
    STORAGE = "storage"


class AllocationStrategy(Enum):
    """Resource allocation strategies."""
    FIRST_FIT = "first_fit"          # First available resource
    BEST_FIT = "best_fit"            # Best matching resource
    WORST_FIT = "worst_fit"          # Largest available resource
    ROUND_ROBIN = "round_robin"      # Distribute evenly
    PRIORITY_BASED = "priority_based" # Based on task priority
    DYNAMIC = "dynamic"              # Adaptive allocation


@dataclass
class ResourceRequirement:
    """Resource requirements for a task."""
    cpu_cores: int = 1
    memory_mb: int = 512
    gpu_memory_mb: int = 0
    network_bandwidth_mbps: float = 10.0
    storage_gb: float = 1.0
    estimated_duration_ms: float = 1000.0


@dataclass
class ResourcePool:
    """Available system resources."""
    total_cpu_cores: int = 8
    total_memory_mb: int = 16384
    total_gpu_memory_mb: int = 8192
    total_network_bandwidth_mbps: float = 1000.0
    total_storage_gb: float = 100.0
    
    available_cpu_cores: int = 8
    available_memory_mb: int = 16384
    available_gpu_memory_mb: int = 8192
    available_network_bandwidth_mbps: float = 1000.0
    available_storage_gb: float = 100.0
    
    def allocate(self, requirement: ResourceRequirement) -> bool:
        """
        Attempt to allocate resources.
        
        Args:
            requirement: Resource requirement
            
        Returns:
            True if allocation successful
        """
        if (self.available_cpu_cores >= requirement.cpu_cores and
            self.available_memory_mb >= requirement.memory_mb and
            self.available_gpu_memory_mb >= requirement.gpu_memory_mb and
            self.available_network_bandwidth_mbps >= requirement.network_bandwidth_mbps and
            self.available_storage_gb >= requirement.storage_gb):
            
            self.available_cpu_cores -= requirement.cpu_cores
            self.available_memory_mb -= requirement.memory_mb
            self.available_gpu_memory_mb -= requirement.gpu_memory_mb
            self.available_network_bandwidth_mbps -= requirement.network_bandwidth_mbps
            self.available_storage_gb -= requirement.storage_gb
            
            return True
        return False
    
    def release(self, requirement: ResourceRequirement):
        """
        Release allocated resources.
        
        Args:
            requirement: Resource requirement to release
        """
        self.available_cpu_cores = min(
            self.total_cpu_cores,
            self.available_cpu_cores + requirement.cpu_cores
        )
        self.available_memory_mb = min(
            self.total_memory_mb,
            self.available_memory_mb + requirement.memory_mb
        )
        self.available_gpu_memory_mb = min(
            self.total_gpu_memory_mb,
            self.available_gpu_memory_mb + requirement.gpu_memory_mb
        )
        self.available_network_bandwidth_mbps = min(
            self.total_network_bandwidth_mbps,
            self.available_network_bandwidth_mbps + requirement.network_bandwidth_mbps
        )
        self.available_storage_gb = min(
            self.total_storage_gb,
            self.available_storage_gb + requirement.storage_gb
        )
    
    def get_utilization(self) -> Dict[ResourceType, float]:
        """
        Get resource utilization percentages.
        
        Returns:
            Utilization for each resource type
        """
        return {
            ResourceType.CPU: 1 - (self.available_cpu_cores / self.total_cpu_cores),
            ResourceType.MEMORY: 1 - (self.available_memory_mb / self.total_memory_mb),
            ResourceType.GPU: 1 - (self.available_gpu_memory_mb / self.total_gpu_memory_mb) if self.total_gpu_memory_mb > 0 else 0,
            ResourceType.NETWORK: 1 - (self.available_network_bandwidth_mbps / self.total_network_bandwidth_mbps),
            ResourceType.STORAGE: 1 - (self.available_storage_gb / self.total_storage_gb)
        }


@dataclass
class RoutingDecision:
    """Decision for task routing."""
    task_id: str
    selected_modules: List[str]
    execution_strategy: ExecutionStrategy
    resource_allocation: ResourceRequirement
    confidence: float
    reasoning: str
    alternatives: List[Dict[str, Any]] = field(default_factory=list)


class TaskRouter:
    """
    Advanced task routing with intelligent decision making.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.routing_history = []
        self.module_performance = defaultdict(lambda: {"success": 0, "failure": 0, "avg_time": 0})
    
    async def route_task(
        self,
        task: TaskRequest,
        available_modules: List[AIModule],
        resource_pool: ResourcePool
    ) -> RoutingDecision:
        """
        Route task to appropriate modules.
        
        Args:
            task: Task to route
            available_modules: Available AI modules
            resource_pool: Available resources
            
        Returns:
            Routing decision
        """
        # Analyze task requirements
        required_capabilities = task.required_capabilities
        task_complexity = self._assess_complexity(task)
        
        # Find capable modules
        capable_modules = self._find_capable_modules(
            required_capabilities,
            available_modules
        )
        
        if not capable_modules:
            return RoutingDecision(
                task_id=str(task.id),
                selected_modules=[],
                execution_strategy=ExecutionStrategy.SEQUENTIAL,
                resource_allocation=ResourceRequirement(),
                confidence=0.0,
                reasoning="No capable modules found"
            )
        
        # Estimate resource requirements
        resource_req = self._estimate_resources(task, task_complexity)
        
        # Select execution strategy
        strategy = self._select_strategy(task, capable_modules, task_complexity)
        
        # Rank and select modules
        selected_modules = await self._select_best_modules(
            capable_modules,
            task,
            strategy,
            resource_pool
        )
        
        # Generate alternatives
        alternatives = self._generate_alternatives(
            capable_modules,
            task,
            selected_modules
        )
        
        # Build routing decision
        decision = RoutingDecision(
            task_id=str(task.id),
            selected_modules=[m.name for m in selected_modules],
            execution_strategy=strategy,
            resource_allocation=resource_req,
            confidence=self._calculate_confidence(selected_modules, task),
            reasoning=self._explain_routing(selected_modules, strategy, task_complexity),
            alternatives=alternatives
        )
        
        # Record decision
        self.routing_history.append({
            "timestamp": datetime.utcnow(),
            "decision": decision
        })
        
        return decision
    
    def _assess_complexity(self, task: TaskRequest) -> float:
        """
        Assess task complexity.
        
        Args:
            task: Task to assess
            
        Returns:
            Complexity score (0-1)
        """
        complexity = 0.0
        
        # Check number of required capabilities
        complexity += len(task.required_capabilities) * 0.1
        
        # Check data size (if available)
        if isinstance(task.input_data, dict):
            complexity += len(str(task.input_data)) / 10000  # Normalize by 10KB
        
        # Check for multi-modal requirements
        multi_modal_caps = [
            ModuleCapability.TEXT_UNDERSTANDING,
            ModuleCapability.IMAGE_UNDERSTANDING,
            ModuleCapability.AUDIO_PROCESSING
        ]
        modal_count = sum(1 for cap in multi_modal_caps if cap in task.required_capabilities)
        if modal_count > 1:
            complexity += 0.3
        
        # Check execution strategy complexity
        if task.execution_strategy in [ExecutionStrategy.PARALLEL, ExecutionStrategy.CONSENSUS]:
            complexity += 0.2
        
        return min(1.0, complexity)
    
    def _find_capable_modules(
        self,
        capabilities: List[ModuleCapability],
        modules: List[AIModule]
    ) -> List[AIModule]:
        """
        Find modules capable of handling required capabilities.
        
        Args:
            capabilities: Required capabilities
            modules: Available modules
            
        Returns:
            List of capable modules
        """
        capable = []
        
        for module in modules:
            module_caps = module.get_capabilities()
            if any(cap in module_caps for cap in capabilities):
                capable.append(module)
        
        return capable
    
    def _estimate_resources(
        self,
        task: TaskRequest,
        complexity: float
    ) -> ResourceRequirement:
        """
        Estimate resource requirements for task.
        
        Args:
            task: Task request
            complexity: Task complexity
            
        Returns:
            Resource requirement
        """
        # Base requirements
        req = ResourceRequirement()
        
        # Scale by complexity
        req.cpu_cores = max(1, int(complexity * 4))
        req.memory_mb = max(512, int(complexity * 4096))
        
        # Check for GPU requirements
        gpu_caps = [
            ModuleCapability.IMAGE_PROCESSING,
            ModuleCapability.VIDEO_PROCESSING
        ]
        if any(cap in task.required_capabilities for cap in gpu_caps):
            req.gpu_memory_mb = max(1024, int(complexity * 4096))
        
        # Estimate duration
        req.estimated_duration_ms = max(100, complexity * 5000)
        
        return req
    
    def _select_strategy(
        self,
        task: TaskRequest,
        capable_modules: List[AIModule],
        complexity: float
    ) -> ExecutionStrategy:
        """
        Select execution strategy.
        
        Args:
            task: Task request
            capable_modules: Capable modules
            complexity: Task complexity
            
        Returns:
            Selected execution strategy
        """
        # Use requested strategy if specified
        if task.execution_strategy != ExecutionStrategy.SEQUENTIAL:
            return task.execution_strategy
        
        # Select based on task characteristics
        if len(capable_modules) > 2 and complexity > 0.5:
            # Complex task with multiple modules - use parallel
            return ExecutionStrategy.PARALLEL
        elif len(task.required_capabilities) > 2:
            # Multiple capabilities - use pipeline
            return ExecutionStrategy.PIPELINE
        elif complexity > 0.7:
            # Very complex - use consensus for reliability
            return ExecutionStrategy.CONSENSUS
        else:
            # Simple task - sequential
            return ExecutionStrategy.SEQUENTIAL
    
    async def _select_best_modules(
        self,
        capable_modules: List[AIModule],
        task: TaskRequest,
        strategy: ExecutionStrategy,
        resource_pool: ResourcePool
    ) -> List[AIModule]:
        """
        Select best modules for task.
        
        Args:
            capable_modules: Capable modules
            task: Task request
            strategy: Execution strategy
            resource_pool: Available resources
            
        Returns:
            Selected modules
        """
        # Score each module
        scored_modules = []
        
        for module in capable_modules:
            score = await self._score_module(module, task, resource_pool)
            scored_modules.append((score, module))
        
        # Sort by score
        scored_modules.sort(reverse=True)
        
        # Select based on strategy
        if strategy == ExecutionStrategy.CONSENSUS:
            # Select top 3 for consensus
            return [m for _, m in scored_modules[:3]]
        elif strategy == ExecutionStrategy.PARALLEL:
            # Select all high-scoring modules
            threshold = scored_modules[0][0] * 0.8 if scored_modules else 0
            return [m for s, m in scored_modules if s >= threshold]
        else:
            # Select best module
            return [scored_modules[0][1]] if scored_modules else []
    
    async def _score_module(
        self,
        module: AIModule,
        task: TaskRequest,
        resource_pool: ResourcePool
    ) -> float:
        """
        Score module for task.
        
        Args:
            module: Module to score
            task: Task request
            resource_pool: Available resources
            
        Returns:
            Module score
        """
        score = 0.0
        
        # Capability match
        module_caps = module.get_capabilities()
        cap_match = sum(1 for cap in task.required_capabilities if cap in module_caps)
        score += cap_match * 0.3
        
        # Historical performance
        perf = self.module_performance[module.name]
        if perf["success"] + perf["failure"] > 0:
            success_rate = perf["success"] / (perf["success"] + perf["failure"])
            score += success_rate * 0.3
        
        # Resource availability
        utilization = resource_pool.get_utilization()
        avg_util = sum(utilization.values()) / len(utilization)
        score += (1 - avg_util) * 0.2
        
        # Module status
        if hasattr(module, 'status'):
            if module.status == ModuleStatus.READY:
                score += 0.2
            elif module.status == ModuleStatus.PROCESSING:
                score += 0.1
        
        return score
    
    def _calculate_confidence(
        self,
        selected_modules: List[AIModule],
        task: TaskRequest
    ) -> float:
        """
        Calculate routing confidence.
        
        Args:
            selected_modules: Selected modules
            task: Task request
            
        Returns:
            Confidence score
        """
        if not selected_modules:
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # Check capability coverage
        covered_caps = set()
        for module in selected_modules:
            covered_caps.update(module.get_capabilities())
        
        coverage = len(set(task.required_capabilities) & covered_caps) / len(task.required_capabilities) if task.required_capabilities else 1.0
        confidence += coverage * 0.3
        
        # Check module performance
        avg_success = 0
        for module in selected_modules:
            perf = self.module_performance[module.name]
            if perf["success"] + perf["failure"] > 0:
                avg_success += perf["success"] / (perf["success"] + perf["failure"])
        
        if selected_modules:
            avg_success /= len(selected_modules)
            confidence += avg_success * 0.2
        
        return min(1.0, confidence)
    
    def _explain_routing(
        self,
        selected_modules: List[AIModule],
        strategy: ExecutionStrategy,
        complexity: float
    ) -> str:
        """
        Explain routing decision.
        
        Args:
            selected_modules: Selected modules
            strategy: Execution strategy
            complexity: Task complexity
            
        Returns:
            Explanation string
        """
        explanation = f"Task complexity: {complexity:.2f}. "
        
        if selected_modules:
            module_names = [m.name for m in selected_modules]
            explanation += f"Selected modules: {', '.join(module_names)}. "
            explanation += f"Execution strategy: {strategy.value}. "
            
            if len(selected_modules) > 1:
                explanation += f"Multiple modules selected for {'consensus' if strategy == ExecutionStrategy.CONSENSUS else 'parallel processing'}."
        else:
            explanation += "No suitable modules found."
        
        return explanation
    
    def _generate_alternatives(
        self,
        capable_modules: List[AIModule],
        task: TaskRequest,
        selected_modules: List[AIModule]
    ) -> List[Dict[str, Any]]:
        """
        Generate alternative routing options.
        
        Args:
            capable_modules: All capable modules
            task: Task request
            selected_modules: Currently selected modules
            
        Returns:
            List of alternatives
        """
        alternatives = []
        
        # Generate different strategy alternatives
        strategies = [
            ExecutionStrategy.SEQUENTIAL,
            ExecutionStrategy.PARALLEL,
            ExecutionStrategy.CONSENSUS
        ]
        
        for strategy in strategies:
            if strategy != task.execution_strategy:
                alternatives.append({
                    "strategy": strategy.value,
                    "modules": [m.name for m in selected_modules],
                    "reasoning": f"Alternative using {strategy.value} execution"
                })
        
        # Generate different module combinations
        other_modules = [m for m in capable_modules if m not in selected_modules]
        if other_modules:
            alternatives.append({
                "strategy": task.execution_strategy.value,
                "modules": [m.name for m in other_modules[:2]],
                "reasoning": "Alternative module selection"
            })
        
        return alternatives[:3]  # Limit to 3 alternatives
    
    def update_performance(
        self,
        module_name: str,
        success: bool,
        execution_time_ms: float
    ):
        """
        Update module performance metrics.
        
        Args:
            module_name: Module name
            success: Whether execution was successful
            execution_time_ms: Execution time
        """
        perf = self.module_performance[module_name]
        
        if success:
            perf["success"] += 1
        else:
            perf["failure"] += 1
        
        # Update average time
        total_executions = perf["success"] + perf["failure"]
        perf["avg_time"] = (
            (perf["avg_time"] * (total_executions - 1) + execution_time_ms) /
            total_executions
        )


class ResourceAllocator:
    """
    Manages resource allocation for tasks.
    """
    
    def __init__(self, resource_pool: ResourcePool):
        self.logger = get_logger(__name__)
        self.resource_pool = resource_pool
        self.allocations = {}  # task_id -> ResourceRequirement
        self.allocation_history = []
    
    async def allocate(
        self,
        task_id: str,
        requirement: ResourceRequirement,
        strategy: AllocationStrategy = AllocationStrategy.BEST_FIT
    ) -> bool:
        """
        Allocate resources for task.
        
        Args:
            task_id: Task ID
            requirement: Resource requirement
            strategy: Allocation strategy
            
        Returns:
            True if allocation successful
        """
        # Check if already allocated
        if task_id in self.allocations:
            self.logger.warning(f"Task {task_id} already has resources allocated")
            return True
        
        # Try to allocate based on strategy
        success = False
        
        if strategy == AllocationStrategy.FIRST_FIT:
            success = self.resource_pool.allocate(requirement)
        elif strategy == AllocationStrategy.BEST_FIT:
            success = await self._best_fit_allocate(requirement)
        elif strategy == AllocationStrategy.PRIORITY_BASED:
            success = await self._priority_allocate(requirement, task_id)
        else:
            success = self.resource_pool.allocate(requirement)
        
        if success:
            self.allocations[task_id] = requirement
            self.allocation_history.append({
                "timestamp": datetime.utcnow(),
                "task_id": task_id,
                "requirement": requirement,
                "action": "allocate"
            })
            self.logger.info(f"Allocated resources for task {task_id}")
        else:
            self.logger.warning(f"Failed to allocate resources for task {task_id}")
        
        return success
    
    async def release(self, task_id: str):
        """
        Release resources for task.
        
        Args:
            task_id: Task ID
        """
        if task_id in self.allocations:
            requirement = self.allocations[task_id]
            self.resource_pool.release(requirement)
            del self.allocations[task_id]
            
            self.allocation_history.append({
                "timestamp": datetime.utcnow(),
                "task_id": task_id,
                "requirement": requirement,
                "action": "release"
            })
            
            self.logger.info(f"Released resources for task {task_id}")
    
    async def _best_fit_allocate(self, requirement: ResourceRequirement) -> bool:
        """
        Best-fit allocation strategy.
        
        Args:
            requirement: Resource requirement
            
        Returns:
            True if allocation successful
        """
        # Check if resources fit well (not too much waste)
        utilization = self.resource_pool.get_utilization()
        
        # Calculate fit score
        cpu_fit = requirement.cpu_cores / self.resource_pool.available_cpu_cores if self.resource_pool.available_cpu_cores > 0 else 0
        mem_fit = requirement.memory_mb / self.resource_pool.available_memory_mb if self.resource_pool.available_memory_mb > 0 else 0
        
        # Good fit if utilization will be between 60-90%
        if 0.6 <= cpu_fit <= 0.9 and 0.6 <= mem_fit <= 0.9:
            return self.resource_pool.allocate(requirement)
        
        # Otherwise try standard allocation
        return self.resource_pool.allocate(requirement)
    
    async def _priority_allocate(
        self,
        requirement: ResourceRequirement,
        task_id: str
    ) -> bool:
        """
        Priority-based allocation.
        
        Args:
            requirement: Resource requirement
            task_id: Task ID
            
        Returns:
            True if allocation successful
        """
        # Try direct allocation first
        if self.resource_pool.allocate(requirement):
            return True
        
        # TODO: Implement preemption for high-priority tasks
        # For now, just return False
        return False
    
    def get_allocation_stats(self) -> Dict[str, Any]:
        """
        Get allocation statistics.
        
        Returns:
            Allocation statistics
        """
        utilization = self.resource_pool.get_utilization()
        
        return {
            "active_allocations": len(self.allocations),
            "utilization": {k.value: v for k, v in utilization.items()},
            "available_resources": {
                "cpu_cores": self.resource_pool.available_cpu_cores,
                "memory_mb": self.resource_pool.available_memory_mb,
                "gpu_memory_mb": self.resource_pool.available_gpu_memory_mb
            },
            "total_resources": {
                "cpu_cores": self.resource_pool.total_cpu_cores,
                "memory_mb": self.resource_pool.total_memory_mb,
                "gpu_memory_mb": self.resource_pool.total_gpu_memory_mb
            }
        }


class DecisionMaker:
    """
    Multi-criteria decision making for task execution.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.decision_history = []
        self.criteria_weights = {
            "performance": 0.3,
            "reliability": 0.25,
            "cost": 0.2,
            "speed": 0.25
        }
    
    async def make_decision(
        self,
        options: List[Dict[str, Any]],
        criteria: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Make decision based on multiple criteria.
        
        Args:
            options: List of options to evaluate
            criteria: Optional custom criteria weights
            
        Returns:
            Selected option with reasoning
        """
        if not options:
            return {
                "selected": None,
                "reasoning": "No options available",
                "confidence": 0.0
            }
        
        # Use custom criteria or defaults
        weights = criteria or self.criteria_weights
        
        # Score each option
        scored_options = []
        for option in options:
            score = await self._score_option(option, weights)
            scored_options.append((score, option))
        
        # Sort by score
        scored_options.sort(reverse=True)
        
        # Select best option
        best_score, best_option = scored_options[0]
        
        # Calculate confidence
        confidence = self._calculate_decision_confidence(scored_options)
        
        # Generate reasoning
        reasoning = self._explain_decision(best_option, weights, best_score)
        
        decision = {
            "selected": best_option,
            "score": best_score,
            "reasoning": reasoning,
            "confidence": confidence,
            "alternatives": [opt for _, opt in scored_options[1:3]]
        }
        
        # Record decision
        self.decision_history.append({
            "timestamp": datetime.utcnow(),
            "decision": decision
        })
        
        return decision
    
    async def _score_option(
        self,
        option: Dict[str, Any],
        weights: Dict[str, float]
    ) -> float:
        """
        Score an option based on criteria.
        
        Args:
            option: Option to score
            weights: Criteria weights
            
        Returns:
            Option score
        """
        score = 0.0
        
        for criterion, weight in weights.items():
            if criterion in option:
                # Normalize criterion value to 0-1
                value = option[criterion]
                if isinstance(value, (int, float)):
                    normalized = min(1.0, max(0.0, value))
                elif isinstance(value, bool):
                    normalized = 1.0 if value else 0.0
                else:
                    normalized = 0.5  # Default for unknown types
                
                score += normalized * weight
        
        return score
    
    def _calculate_decision_confidence(
        self,
        scored_options: List[Tuple[float, Dict[str, Any]]]
    ) -> float:
        """
        Calculate confidence in decision.
        
        Args:
            scored_options: Scored options
            
        Returns:
            Confidence score
        """
        if len(scored_options) < 2:
            return 0.9  # High confidence if only one option
        
        # Calculate based on score separation
        best_score = scored_options[0][0]
        second_score = scored_options[1][0]
        
        if best_score > 0:
            separation = (best_score - second_score) / best_score
            confidence = min(0.95, 0.5 + separation)
        else:
            confidence = 0.5
        
        return confidence
    
    def _explain_decision(
        self,
        option: Dict[str, Any],
        weights: Dict[str, float],
        score: float
    ) -> str:
        """
        Explain decision reasoning.
        
        Args:
            option: Selected option
            weights: Criteria weights
            score: Option score
            
        Returns:
            Explanation string
        """
        explanation = f"Selected option with score {score:.2f}. "
        
        # Find top contributing criteria
        contributions = []
        for criterion, weight in weights.items():
            if criterion in option:
                value = option[criterion]
                if isinstance(value, (int, float)):
                    contribution = value * weight
                    contributions.append((contribution, criterion))
        
        contributions.sort(reverse=True)
        
        if contributions:
            top_criteria = [c for _, c in contributions[:2]]
            explanation += f"Key factors: {', '.join(top_criteria)}."
        
        return explanation
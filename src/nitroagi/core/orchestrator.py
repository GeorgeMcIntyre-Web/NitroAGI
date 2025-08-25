"""NEXUS - Neural Executive Unit System for NitroAGI.

The NEXUS core engine manages and coordinates all AI modules, handling request routing,
load balancing, and system-wide coordination. NEXUS is the brain of NitroAGI.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

from nitroagi.core.base import (
    AIModule,
    ModuleCapability,
    ModuleRegistry,
    ModuleRequest,
    ModuleResponse,
    ProcessingContext,
)
from nitroagi.core.exceptions import OrchestratorException, TimeoutException
from nitroagi.core.message_bus import Message, MessageBus, MessagePriority, MessageType
from nitroagi.core.prefrontal_cortex import PrefrontalCortex


class TaskStatus(Enum):
    """Status of a task in the orchestrator."""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ExecutionStrategy(Enum):
    """Strategy for executing tasks."""
    SEQUENTIAL = "sequential"  # Execute tasks one after another
    PARALLEL = "parallel"  # Execute tasks in parallel
    PIPELINE = "pipeline"  # Execute as a pipeline (output of one is input to next)
    CONDITIONAL = "conditional"  # Execute based on conditions
    CONSENSUS = "consensus"  # Execute multiple and take consensus


@dataclass
class TaskRequest:
    """Request for the orchestrator to process."""
    id: UUID = field(default_factory=uuid4)
    input_data: Any = None
    required_capabilities: List[ModuleCapability] = field(default_factory=list)
    execution_strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
    context: ProcessingContext = field(default_factory=ProcessingContext)
    priority: int = 0
    timeout_seconds: float = 60.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_task_id: Optional[UUID] = None
    dependencies: List[UUID] = field(default_factory=list)


@dataclass
class TaskResult:
    """Result from processing a task."""
    task_id: UUID
    status: TaskStatus
    results: List[ModuleResponse] = field(default_factory=list)
    final_output: Any = None
    errors: List[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, error: str) -> None:
        """Add an error to the result."""
        self.errors.append(error)
    
    def complete(self, final_output: Any = None) -> None:
        """Mark the task as completed."""
        self.status = TaskStatus.COMPLETED
        self.final_output = final_output
        self.end_time = datetime.utcnow()
        self.execution_time_ms = (
            (self.end_time - self.start_time).total_seconds() * 1000
        )
    
    def fail(self, error: str) -> None:
        """Mark the task as failed."""
        self.status = TaskStatus.FAILED
        self.add_error(error)
        self.end_time = datetime.utcnow()
        self.execution_time_ms = (
            (self.end_time - self.start_time).total_seconds() * 1000
        )


class TaskQueue:
    """Priority queue for tasks."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize the task queue.
        
        Args:
            max_size: Maximum queue size
        """
        self._queue: List[Tuple[int, TaskRequest]] = []
        self._max_size = max_size
        self._lock = asyncio.Lock()
    
    async def put(self, task: TaskRequest) -> None:
        """Add a task to the queue.
        
        Args:
            task: Task to add
        """
        async with self._lock:
            if len(self._queue) >= self._max_size:
                raise OrchestratorException("Task queue is full")
            
            # Insert task based on priority (higher priority first)
            priority = -task.priority  # Negative for max heap behavior
            insert_pos = 0
            
            for i, (p, _) in enumerate(self._queue):
                if priority < p:
                    insert_pos = i + 1
                else:
                    break
            
            self._queue.insert(insert_pos, (priority, task))
    
    async def get(self) -> Optional[TaskRequest]:
        """Get the highest priority task from the queue.
        
        Returns:
            Task if available, None otherwise
        """
        async with self._lock:
            if self._queue:
                _, task = self._queue.pop(0)
                return task
            return None
    
    async def size(self) -> int:
        """Get the current queue size.
        
        Returns:
            Number of tasks in queue
        """
        async with self._lock:
            return len(self._queue)
    
    async def clear(self) -> None:
        """Clear all tasks from the queue."""
        async with self._lock:
            self._queue.clear()


class Orchestrator:
    """NEXUS - Neural Executive Unit System.
    
    The core intelligence engine that orchestrates and coordinates all AI modules
    in the NitroAGI system. NEXUS acts as the central brain, managing module
    communication, load balancing, and complex multi-modal AI workflows.
    """
    
    def __init__(
        self,
        registry: ModuleRegistry,
        message_bus: MessageBus,
        max_concurrent_tasks: int = 10,
        max_queue_size: int = 1000
    ):
        """Initialize the NEXUS core engine.
        
        Args:
            registry: Module registry
            message_bus: Message bus for communication
            max_concurrent_tasks: Maximum concurrent tasks
            max_queue_size: Maximum task queue size
        """
        self.logger = logging.getLogger("nitroagi.core.nexus")
        self.registry = registry
        self.message_bus = message_bus
        self.max_concurrent_tasks = max_concurrent_tasks
        
        self._task_queue = TaskQueue(max_queue_size)
        self._active_tasks: Dict[UUID, TaskResult] = {}
        self._completed_tasks: Dict[UUID, TaskResult] = {}
        self._task_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._running = False
        self._workers: List[asyncio.Task] = []
        
        # Initialize prefrontal cortex for executive control
        self.prefrontal_cortex = PrefrontalCortex()
        
        self._metrics = {
            "tasks_received": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_timeout": 0,
            "average_execution_time_ms": 0.0,
        }
    
    async def start(self) -> None:
        """Start the orchestrator."""
        self._running = True
        
        # Start worker tasks
        for i in range(self.max_concurrent_tasks):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self._workers.append(worker)
        
        self.logger.info(f"NEXUS core engine started with {self.max_concurrent_tasks} workers")
        self.logger.info("Prefrontal cortex executive control system activated")
    
    async def stop(self) -> None:
        """Stop the orchestrator."""
        self._running = False
        
        # Cancel all workers
        for worker in self._workers:
            worker.cancel()
        
        # Wait for workers to finish
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        
        self.logger.info("NEXUS core engine stopped")
    
    async def submit_task(self, request: TaskRequest) -> UUID:
        """Submit a task for processing.
        
        Args:
            request: Task request to process
            
        Returns:
            Task ID for tracking
        """
        self._metrics["tasks_received"] += 1
        
        # Create task result
        result = TaskResult(
            task_id=request.id,
            status=TaskStatus.QUEUED
        )
        self._active_tasks[request.id] = result
        
        # Add to queue
        await self._task_queue.put(request)
        
        self.logger.info(f"Task {request.id} submitted for processing")
        return request.id
    
    async def get_task_status(self, task_id: UUID) -> Optional[TaskResult]:
        """Get the status of a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task result if found
        """
        if task_id in self._active_tasks:
            return self._active_tasks[task_id]
        return self._completed_tasks.get(task_id)
    
    async def wait_for_task(
        self,
        task_id: UUID,
        timeout: Optional[float] = None
    ) -> TaskResult:
        """Wait for a task to complete.
        
        Args:
            task_id: ID of the task
            timeout: Optional timeout in seconds
            
        Returns:
            Task result
        """
        start_time = asyncio.get_event_loop().time()
        
        while True:
            result = await self.get_task_status(task_id)
            
            if result and result.status in [
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.TIMEOUT
            ]:
                return result
            
            if timeout and (asyncio.get_event_loop().time() - start_time) > timeout:
                raise TimeoutException(
                    f"Timeout waiting for task {task_id}",
                    operation="wait_for_task",
                    timeout_seconds=timeout
                )
            
            await asyncio.sleep(0.1)
    
    async def cancel_task(self, task_id: UUID) -> bool:
        """Cancel a task.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            True if task was cancelled
        """
        if task_id in self._active_tasks:
            result = self._active_tasks[task_id]
            if result.status in [TaskStatus.PENDING, TaskStatus.QUEUED]:
                result.status = TaskStatus.CANCELLED
                result.end_time = datetime.utcnow()
                self._move_to_completed(task_id)
                self.logger.info(f"Task {task_id} cancelled")
                return True
        return False
    
    async def _worker(self, worker_id: str) -> None:
        """Worker task for processing requests.
        
        Args:
            worker_id: ID of the worker
        """
        self.logger.debug(f"Worker {worker_id} started")
        
        while self._running:
            try:
                # Get task from queue
                task = await self._task_queue.get()
                
                if task:
                    async with self._task_semaphore:
                        await self._process_task(task)
                else:
                    # No task available, wait a bit
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}", exc_info=True)
        
        self.logger.debug(f"Worker {worker_id} stopped")
    
    async def _process_task(self, request: TaskRequest) -> None:
        """Process a task request using prefrontal cortex executive control.
        
        Args:
            request: Task request to process
        """
        result = self._active_tasks.get(request.id)
        if not result:
            self.logger.error(f"Task {request.id} not found in active tasks")
            return
        
        result.status = TaskStatus.PROCESSING
        self.logger.info(f"Processing task {request.id} with executive control")
        
        try:
            # Use prefrontal cortex for executive processing
            available_modules = [module.name for module in self._get_available_modules()]
            context = {
                "task_id": str(request.id),
                "execution_strategy": request.execution_strategy.value,
                "required_capabilities": [cap.value for cap in request.required_capabilities],
                "priority": request.priority,
                "timeout": request.timeout_seconds,
                "metadata": request.metadata
            }
            
            # Convert input data to goal string for prefrontal cortex
            goal = self._extract_goal_from_request(request)
            
            # Execute with prefrontal cortex coordination
            executive_result = await self.prefrontal_cortex.executive_process(
                goal=goal,
                context=context,
                available_modules=available_modules
            )
            
            if "error" in executive_result:
                raise OrchestratorException(executive_result["error"])
            
            # Execute the planned steps with traditional orchestrator
            await self._execute_with_executive_plan(request, result, executive_result)
            
            # Mark as completed
            result.complete(result.final_output)
            self._metrics["tasks_completed"] += 1
            
        except asyncio.TimeoutError:
            result.status = TaskStatus.TIMEOUT
            result.add_error("Task timed out")
            self._metrics["tasks_timeout"] += 1
            
        except Exception as e:
            result.fail(str(e))
            self._metrics["tasks_failed"] += 1
            self.logger.error(f"Task {request.id} failed: {e}", exc_info=True)
        
        finally:
            # Update metrics
            self._update_metrics(result)
            
            # Move to completed
            self._move_to_completed(request.id)
    
    async def _execute_sequential(
        self,
        request: TaskRequest,
        result: TaskResult
    ) -> None:
        """Execute modules sequentially.
        
        Args:
            request: Task request
            result: Task result to update
        """
        modules = self._select_modules(request.required_capabilities)
        current_input = request.input_data
        
        for module in modules:
            module_request = ModuleRequest(
                data=current_input,
                context=request.context,
                priority=request.priority,
                required_capabilities=request.required_capabilities
            )
            
            response = await asyncio.wait_for(
                module.process(module_request),
                timeout=request.timeout_seconds
            )
            
            result.results.append(response)
            current_input = response.data  # Use output as next input
        
        result.final_output = current_input
    
    async def _execute_parallel(
        self,
        request: TaskRequest,
        result: TaskResult
    ) -> None:
        """Execute modules in parallel.
        
        Args:
            request: Task request
            result: Task result to update
        """
        modules = self._select_modules(request.required_capabilities)
        
        tasks = []
        for module in modules:
            module_request = ModuleRequest(
                data=request.input_data,
                context=request.context,
                priority=request.priority,
                required_capabilities=request.required_capabilities
            )
            
            task = asyncio.create_task(
                asyncio.wait_for(
                    module.process(module_request),
                    timeout=request.timeout_seconds
                )
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for response in responses:
            if isinstance(response, Exception):
                result.add_error(str(response))
            else:
                result.results.append(response)
        
        # Combine results
        result.final_output = [r.data for r in result.results if not isinstance(r, Exception)]
    
    async def _execute_pipeline(
        self,
        request: TaskRequest,
        result: TaskResult
    ) -> None:
        """Execute modules as a pipeline.
        
        Args:
            request: Task request
            result: Task result to update
        """
        # Similar to sequential but with more sophisticated data transformation
        await self._execute_sequential(request, result)
    
    async def _execute_consensus(
        self,
        request: TaskRequest,
        result: TaskResult
    ) -> None:
        """Execute modules and take consensus.
        
        Args:
            request: Task request
            result: Task result to update
        """
        # Execute in parallel and then determine consensus
        await self._execute_parallel(request, result)
        
        # Simple consensus: most common result
        if result.results:
            outputs = [r.data for r in result.results]
            # This is a simplified consensus - real implementation would be more sophisticated
            result.final_output = outputs[0] if outputs else None
    
    def _select_modules(
        self,
        capabilities: List[ModuleCapability]
    ) -> List[AIModule]:
        """Select modules based on required capabilities.
        
        Args:
            capabilities: Required capabilities
            
        Returns:
            List of modules that can handle the capabilities
        """
        selected_modules = []
        
        for capability in capabilities:
            modules = self.registry.get_modules_by_capability(capability)
            if modules:
                # Select the first available module for each capability
                # More sophisticated selection logic can be implemented
                selected_modules.append(modules[0])
        
        return selected_modules
    
    def _move_to_completed(self, task_id: UUID) -> None:
        """Move a task from active to completed.
        
        Args:
            task_id: ID of the task to move
        """
        if task_id in self._active_tasks:
            result = self._active_tasks.pop(task_id)
            self._completed_tasks[task_id] = result
            
            # Limit completed tasks history
            if len(self._completed_tasks) > 100:
                oldest_id = next(iter(self._completed_tasks))
                del self._completed_tasks[oldest_id]
    
    def _update_metrics(self, result: TaskResult) -> None:
        """Update orchestrator metrics.
        
        Args:
            result: Task result to use for metrics
        """
        if result.execution_time_ms > 0:
            total_time = (
                self._metrics["average_execution_time_ms"] * 
                (self._metrics["tasks_completed"] + self._metrics["tasks_failed"])
            )
            total_time += result.execution_time_ms
            total_tasks = (
                self._metrics["tasks_completed"] + 
                self._metrics["tasks_failed"] + 1
            )
            self._metrics["average_execution_time_ms"] = total_time / total_tasks
    
    def _extract_goal_from_request(self, request: TaskRequest) -> str:
        """Extract a goal string from task request for prefrontal cortex.
        
        Args:
            request: Task request
            
        Returns:
            Goal string for executive processing
        """
        if isinstance(request.input_data, str):
            return request.input_data
        elif isinstance(request.input_data, dict) and "goal" in request.input_data:
            return request.input_data["goal"]
        else:
            # Generate goal from capabilities
            capabilities = [cap.value for cap in request.required_capabilities]
            return f"Process using capabilities: {', '.join(capabilities)}"
    
    def _get_available_modules(self) -> List[AIModule]:
        """Get list of available modules.
        
        Returns:
            List of available AI modules
        """
        # Get all modules from registry
        modules = []
        for capability in ModuleCapability:
            modules.extend(self.registry.get_modules_by_capability(capability))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_modules = []
        for module in modules:
            if module.name not in seen:
                seen.add(module.name)
                unique_modules.append(module)
        
        return unique_modules
    
    async def _execute_with_executive_plan(
        self,
        request: TaskRequest,
        result: TaskResult,
        executive_result: Dict[str, Any]
    ) -> None:
        """Execute task using executive plan from prefrontal cortex.
        
        Args:
            request: Original task request
            result: Task result to update
            executive_result: Result from prefrontal cortex planning
        """
        execution_plan = executive_result.get("execution_plan", [])
        
        if not execution_plan:
            # Fallback to traditional execution
            await self._execute_traditional(request, result)
            return
        
        # Execute each step in the plan
        current_input = request.input_data
        
        for step in execution_plan:
            task_info = step["task"]
            action = step["action"]
            prediction = step["prediction"]
            
            self.logger.info(f"Executing step: {task_info['task']} using {action}")
            
            # Find the appropriate module
            module = self._find_module_by_name(action)
            if not module:
                self.logger.warning(f"Module {action} not found, using fallback")
                continue
            
            # Create module request
            module_request = ModuleRequest(
                data=current_input,
                context=request.context,
                priority=request.priority,
                required_capabilities=request.required_capabilities
            )
            
            # Execute with timeout from prediction
            timeout = prediction.get("predicted_time", 1000) / 1000.0  # Convert to seconds
            
            try:
                response = await asyncio.wait_for(
                    module.process(module_request),
                    timeout=min(timeout * 2, request.timeout_seconds)  # Use prediction but cap at request timeout
                )
                
                result.results.append(response)
                current_input = response.data  # Chain outputs
                
            except asyncio.TimeoutError:
                self.logger.warning(f"Step timed out: {task_info['task']}")
                result.add_error(f"Step timeout: {task_info['task']}")
                continue
            except Exception as e:
                self.logger.error(f"Step failed: {task_info['task']}: {e}")
                result.add_error(f"Step error: {task_info['task']}: {str(e)}")
                continue
        
        result.final_output = current_input
    
    def _find_module_by_name(self, name: str) -> Optional[AIModule]:
        """Find module by name.
        
        Args:
            name: Module name to find
            
        Returns:
            Module if found, None otherwise
        """
        for module in self._get_available_modules():
            if module.name == name or module.name.lower() == name.lower():
                return module
        return None
    
    async def _execute_traditional(
        self,
        request: TaskRequest,
        result: TaskResult
    ) -> None:
        """Execute using traditional orchestrator logic as fallback.
        
        Args:
            request: Task request
            result: Task result to update
        """
        # Execute based on strategy
        if request.execution_strategy == ExecutionStrategy.SEQUENTIAL:
            await self._execute_sequential(request, result)
        elif request.execution_strategy == ExecutionStrategy.PARALLEL:
            await self._execute_parallel(request, result)
        elif request.execution_strategy == ExecutionStrategy.PIPELINE:
            await self._execute_pipeline(request, result)
        elif request.execution_strategy == ExecutionStrategy.CONSENSUS:
            await self._execute_consensus(request, result)
        else:
            raise OrchestratorException(
                f"Unknown execution strategy: {request.execution_strategy}"
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics.
        
        Returns:
            Dictionary containing metrics
        """
        return {
            **self._metrics,
            "active_tasks": len(self._active_tasks),
            "completed_tasks": len(self._completed_tasks),
            "queue_size": asyncio.run(self._task_queue.size()),
            "prefrontal_cortex_state": self.prefrontal_cortex.get_executive_state(),
        }
"""Unit tests for prefrontal cortex module."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from nitroagi.core.prefrontal_cortex import (
    PrefrontalCortex,
    TaskDecomposer,
    ActionSelector,
    ExecutiveMonitor,
    StatePredictor,
    StateEvaluator,
    ExecutiveState,
    CognitiveFunction
)
from nitroagi.core.base import ModuleResponse


class TestTaskDecomposer:
    """Test TaskDecomposer functionality."""
    
    @pytest.mark.asyncio
    async def test_simple_task_decomposition(self):
        """Test decomposing a simple task."""
        decomposer = TaskDecomposer()
        
        goal = "Generate text"
        context = {"user": "test"}
        
        subtasks = await decomposer.decompose_task(goal, context)
        
        assert len(subtasks) == 1
        assert subtasks[0]["task"] == goal
        assert subtasks[0]["priority"] == 1
        assert subtasks[0]["dependencies"] == []
    
    @pytest.mark.asyncio
    async def test_complex_task_decomposition(self):
        """Test decomposing a complex task."""
        decomposer = TaskDecomposer()
        
        goal = "Analyze multiple documents and create a comprehensive report with various comparisons"
        context = {"user": "test"}
        
        subtasks = await decomposer.decompose_task(goal, context)
        
        assert len(subtasks) == 3  # Complex task should be decomposed
        assert "Analyze requirements" in subtasks[0]["task"]
        assert "Execute main components" in subtasks[1]["task"]
        assert "Validate and refine" in subtasks[2]["task"]
        assert subtasks[1]["dependencies"] == [1]
        assert subtasks[2]["dependencies"] == [2]
    
    def test_complexity_assessment(self):
        """Test complexity assessment logic."""
        decomposer = TaskDecomposer()
        
        simple_goal = "Hello world"
        complex_goal = "Plan and analyze multiple steps to create various outputs"
        
        simple_complexity = decomposer._assess_complexity(simple_goal)
        complex_complexity = decomposer._assess_complexity(complex_goal)
        
        assert simple_complexity < complex_complexity
        assert complex_complexity >= 3


class TestActionSelector:
    """Test ActionSelector functionality."""
    
    @pytest.mark.asyncio
    async def test_action_selection_text(self):
        """Test selecting module for text tasks."""
        selector = ActionSelector()
        
        task = {"task": "Generate text about AI"}
        available_modules = ["language", "vision", "audio"]
        
        action = await selector.select_action(task, available_modules)
        
        assert action == "language"
    
    @pytest.mark.asyncio
    async def test_action_selection_image(self):
        """Test selecting module for image tasks."""
        selector = ActionSelector()
        
        task = {"task": "Process this image"}
        available_modules = ["language", "vision", "audio"]
        
        action = await selector.select_action(task, available_modules)
        
        assert action == "vision"
    
    @pytest.mark.asyncio
    async def test_fallback_selection(self):
        """Test fallback when no matching module found."""
        selector = ActionSelector()
        
        task = {"task": "Unknown task type"}
        available_modules = ["vision", "audio"]
        
        action = await selector.select_action(task, available_modules)
        
        assert action == "vision"  # First available module


class TestExecutiveMonitor:
    """Test ExecutiveMonitor functionality."""
    
    @pytest.mark.asyncio
    async def test_monitor_successful_execution(self):
        """Test monitoring successful execution."""
        monitor = ExecutiveMonitor()
        
        response = ModuleResponse(
            request_id="test",
            module_name="language",
            status="success",
            data="Good response",
            processing_time_ms=1000,
            confidence_score=0.9
        )
        
        result = await monitor.monitor_execution(response)
        
        assert result["status"] == "normal"
        assert result["intervention_needed"] is False
        assert len(result["issues"]) == 0
        assert len(monitor.execution_history) == 1
    
    @pytest.mark.asyncio
    async def test_monitor_error_execution(self):
        """Test monitoring failed execution."""
        monitor = ExecutiveMonitor()
        
        response = ModuleResponse(
            request_id="test",
            module_name="language",
            status="error",
            data="Error occurred",
            processing_time_ms=500,
            confidence_score=0.0
        )
        
        result = await monitor.monitor_execution(response)
        
        assert result["intervention_needed"] is True
        assert "Module execution error" in result["issues"]
        assert "Retry with fallback module" in result["recommendations"]
    
    @pytest.mark.asyncio
    async def test_monitor_low_confidence(self):
        """Test monitoring low confidence responses."""
        monitor = ExecutiveMonitor()
        
        response = ModuleResponse(
            request_id="test",
            module_name="language",
            status="success",
            data="Uncertain response",
            processing_time_ms=1000,
            confidence_score=0.3
        )
        
        result = await monitor.monitor_execution(response)
        
        assert "Low confidence response" in result["issues"]
        assert "Request clarification or use consensus" in result["recommendations"]
    
    @pytest.mark.asyncio
    async def test_monitor_slow_execution(self):
        """Test monitoring slow execution."""
        monitor = ExecutiveMonitor()
        
        response = ModuleResponse(
            request_id="test",
            module_name="language",
            status="success",
            data="Slow response",
            processing_time_ms=6000,  # 6 seconds
            confidence_score=0.8
        )
        
        result = await monitor.monitor_execution(response)
        
        assert "Slow response time" in result["issues"]
        assert "Consider caching or optimization" in result["recommendations"]


class TestStatePredictor:
    """Test StatePredictor functionality."""
    
    @pytest.mark.asyncio
    async def test_predict_simple_action(self):
        """Test predicting outcome of simple action."""
        predictor = StatePredictor()
        
        action = "generate simple text"
        context = {"user": "test"}
        
        prediction = await predictor.predict_outcome(action, context)
        
        assert prediction["likely_success"] == 0.9  # Generate is optimistic
        assert prediction["predicted_time"] == 2000
        assert prediction["confidence"] == 0.7
    
    @pytest.mark.asyncio
    async def test_predict_complex_action(self):
        """Test predicting outcome of complex action."""
        predictor = StatePredictor()
        
        action = "complex analysis of multiple data sources"
        context = {"user": "test"}
        
        prediction = await predictor.predict_outcome(action, context)
        
        assert prediction["likely_success"] == 0.6  # Complex is less optimistic
        assert prediction["predicted_time"] == 3000
    
    @pytest.mark.asyncio
    async def test_predict_default_action(self):
        """Test predicting outcome of unrecognized action."""
        predictor = StatePredictor()
        
        action = "unknown action"
        context = {"user": "test"}
        
        prediction = await predictor.predict_outcome(action, context)
        
        assert prediction["likely_success"] == 0.8  # Default optimism
        assert prediction["predicted_time"] == 1000  # Default time


class TestStateEvaluator:
    """Test StateEvaluator functionality."""
    
    @pytest.mark.asyncio
    async def test_evaluate_no_progress(self):
        """Test evaluating when no tasks completed."""
        evaluator = StateEvaluator()
        
        goal = "Complete project"
        completed_tasks = []
        current_state = {}
        
        evaluation = await evaluator.evaluate_progress(goal, completed_tasks, current_state)
        
        assert evaluation["progress_percentage"] == 0.0
        assert evaluation["goal_achievable"] is True
        assert len(evaluation["next_steps"]) == 0
    
    @pytest.mark.asyncio
    async def test_evaluate_with_progress(self):
        """Test evaluating with some progress."""
        evaluator = StateEvaluator()
        
        goal = "Complete project"
        completed_tasks = [{"task": "step1"}, {"task": "step2"}]
        current_state = {"working": True}
        
        evaluation = await evaluator.evaluate_progress(goal, completed_tasks, current_state)
        
        assert evaluation["progress_percentage"] == 0.6  # 2 * 0.3
        assert evaluation["goal_achievable"] is True
        assert "Continue with planned tasks" in evaluation["next_steps"]


class TestPrefrontalCortex:
    """Test main PrefrontalCortex functionality."""
    
    @pytest.mark.asyncio
    async def test_executive_process_initialization(self):
        """Test prefrontal cortex initialization."""
        cortex = PrefrontalCortex()
        
        assert cortex.task_decomposer is not None
        assert cortex.action_selector is not None
        assert cortex.monitor is not None
        assert cortex.predictor is not None
        assert cortex.evaluator is not None
        assert cortex.working_memory_capacity == 7
    
    @pytest.mark.asyncio
    async def test_executive_process_simple_goal(self):
        """Test executive processing of simple goal."""
        cortex = PrefrontalCortex()
        
        goal = "Generate hello world"
        context = {"user": "test"}
        available_modules = ["language", "vision"]
        
        result = await cortex.executive_process(goal, context, available_modules)
        
        assert "goal" in result
        assert "execution_plan" in result
        assert "results" in result
        assert "evaluation" in result
        assert "executive_state" in result
        assert result["goal"] == goal
        assert len(result["execution_plan"]) > 0
    
    @pytest.mark.asyncio
    async def test_executive_process_complex_goal(self):
        """Test executive processing of complex goal."""
        cortex = PrefrontalCortex()
        
        goal = "Analyze and create a comprehensive report with multiple comparisons"
        context = {"user": "test", "documents": ["doc1", "doc2"]}
        available_modules = ["language", "vision", "audio"]
        
        result = await cortex.executive_process(goal, context, available_modules)
        
        assert "goal" in result
        assert result["goal"] == goal
        assert len(result["execution_plan"]) == 3  # Complex goal should have 3 steps
        
        # Check that each step has the required components
        for step in result["execution_plan"]:
            assert "task" in step
            assert "action" in step
            assert "prediction" in step
    
    @pytest.mark.asyncio
    async def test_working_memory_update(self):
        """Test working memory updates."""
        cortex = PrefrontalCortex()
        
        # Fill working memory beyond capacity
        for i in range(10):
            cortex._update_working_memory({"item": i})
        
        # Should maintain capacity limit
        assert len(cortex.current_state.working_memory) <= cortex.working_memory_capacity
        
        # Should have most recent items
        recent_items = [item["item"]["item"] for item in cortex.current_state.working_memory]
        assert 9 in recent_items  # Most recent item should be present
    
    @pytest.mark.asyncio
    async def test_executive_state_tracking(self):
        """Test executive state tracking."""
        cortex = PrefrontalCortex()
        
        goal = "Test goal"
        context = {"test": True}
        available_modules = ["language"]
        
        await cortex.executive_process(goal, context, available_modules)
        
        state = cortex.get_executive_state()
        
        assert "current_goal" in state
        assert "working_memory_items" in state
        assert "attention_focus" in state
        assert "planning_stack_depth" in state
        assert "recent_executions" in state
        assert state["current_goal"] == goal
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in executive processing."""
        cortex = PrefrontalCortex()
        
        # Mock an error in task decomposer
        with patch.object(cortex.task_decomposer, 'decompose_task', side_effect=Exception("Test error")):
            goal = "Test goal"
            context = {}
            available_modules = ["language"]
            
            result = await cortex.executive_process(goal, context, available_modules)
            
            assert "error" in result
            assert result["goal"] == goal
            assert "Test error" in result["error"]
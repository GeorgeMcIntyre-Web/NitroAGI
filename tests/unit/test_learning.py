"""
Unit tests for Learning and Adaptation System
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, MagicMock
from datetime import datetime

from src.nitroagi.core.learning import (
    State,
    Experience,
    ExperienceReplay,
    QNetwork,
    ReinforcementLearner,
    MetaLearner,
    ContinuousLearner,
    LearningOrchestrator,
    PerformanceMonitor,
    FeedbackLoop,
    ActionSpace
)


class TestState:
    """Test State class"""
    
    def test_state_creation(self):
        """Test state initialization"""
        state = State(
            task_complexity=0.7,
            available_resources={'cpu': 0.8, 'memory': 0.6},
            module_performance={'vision': 0.9, 'language': 0.85},
            context_embedding=np.array([1, 2, 3]),
            temporal_features={'time_of_day': 14},
            goal_alignment=0.8,
            uncertainty=0.2
        )
        
        assert state.task_complexity == 0.7
        assert state.goal_alignment == 0.8
        assert state.uncertainty == 0.2
    
    def test_state_to_vector(self):
        """Test state vectorization"""
        state = State(
            task_complexity=0.5,
            available_resources={'cpu': 0.7, 'memory': 0.8},
            module_performance={'module1': 0.9},
            context_embedding=np.ones(15),
            temporal_features={},
            goal_alignment=0.9,
            uncertainty=0.1
        )
        
        vector = state.to_vector()
        assert isinstance(vector, np.ndarray)
        assert len(vector) == 15  # 5 base features + 10 embedding dims


class TestExperienceReplay:
    """Test Experience Replay buffer"""
    
    def test_buffer_initialization(self):
        """Test buffer creation"""
        buffer = ExperienceReplay(capacity=100)
        assert buffer.capacity == 100
        assert len(buffer.buffer) == 0
    
    def test_add_experience(self):
        """Test adding experiences"""
        buffer = ExperienceReplay(capacity=10)
        
        state = Mock(spec=State)
        experience = Experience(
            state=state,
            action="test_action",
            reward=1.0,
            next_state=state,
            done=False
        )
        
        buffer.add(experience)
        assert len(buffer.buffer) == 1
    
    def test_buffer_overflow(self):
        """Test buffer capacity limit"""
        buffer = ExperienceReplay(capacity=2)
        
        state = Mock(spec=State)
        for i in range(3):
            exp = Experience(
                state=state,
                action=f"action_{i}",
                reward=float(i),
                next_state=state,
                done=False
            )
            buffer.add(exp)
        
        assert len(buffer.buffer) == 2
        assert buffer.buffer[0].action == "action_1"  # First one removed
    
    def test_sampling(self):
        """Test experience sampling"""
        buffer = ExperienceReplay(capacity=10, prioritized=False)
        
        state = Mock(spec=State)
        for i in range(5):
            exp = Experience(
                state=state,
                action=f"action_{i}",
                reward=float(i),
                next_state=state,
                done=False
            )
            buffer.add(exp)
        
        batch = buffer.sample(3)
        assert len(batch) == 3
        assert all(isinstance(exp, Experience) for exp in batch)


class TestQNetwork:
    """Test Q-Network"""
    
    def test_network_initialization(self):
        """Test Q-network creation"""
        net = QNetwork(state_dim=10, action_dim=5, hidden_dim=64)
        
        assert net.state_dim == 10
        assert net.action_dim == 5
        assert net.hidden_dim == 64
        assert 'w1' in net.weights
        assert 'w2' in net.weights
        assert 'w3' in net.weights
    
    def test_forward_pass(self):
        """Test forward propagation"""
        net = QNetwork(state_dim=5, action_dim=3)
        state = np.random.randn(5)
        
        q_values = net.forward(state)
        
        assert isinstance(q_values, np.ndarray)
        assert len(q_values) == 3
    
    def test_target_update(self):
        """Test soft target network update"""
        net = QNetwork(state_dim=5, action_dim=3)
        
        # Modify main network
        net.weights['w1'] += 1.0
        
        # Update target
        net.update_target(tau=0.1)
        
        # Check target was updated
        assert not np.array_equal(net.weights['w1'], net.target_weights['w1'])


class TestReinforcementLearner:
    """Test RL Agent"""
    
    def test_agent_initialization(self):
        """Test agent creation"""
        agent = ReinforcementLearner()
        
        assert agent.epsilon == 1.0
        assert agent.gamma == 0.99
        assert len(agent.actions) == len(ActionSpace)
    
    def test_action_selection_exploration(self):
        """Test exploration action selection"""
        agent = ReinforcementLearner()
        agent.epsilon = 1.0  # Full exploration
        
        state = Mock(spec=State)
        state.to_vector.return_value = np.zeros(25)
        
        action = agent.select_action(state, explore=True)
        assert action in [a.value for a in ActionSpace]
    
    def test_action_selection_exploitation(self):
        """Test exploitation action selection"""
        agent = ReinforcementLearner()
        agent.epsilon = 0.0  # No exploration
        
        state = Mock(spec=State)
        state.to_vector.return_value = np.zeros(25)
        
        action = agent.select_action(state, explore=False)
        assert action in [a.value for a in ActionSpace]
    
    def test_learning(self):
        """Test learning from batch"""
        agent = ReinforcementLearner()
        
        # Add experiences
        state = Mock(spec=State)
        state.to_vector.return_value = np.random.randn(25)
        
        for i in range(10):
            exp = Experience(
                state=state,
                action=ActionSpace.SELECT_MODULE.value,
                reward=1.0,
                next_state=state,
                done=False
            )
            agent.replay_buffer.add(exp)
        
        initial_epsilon = agent.epsilon
        agent.learn(batch_size=5)
        
        # Check epsilon decayed
        assert agent.epsilon < initial_epsilon


class TestMetaLearner:
    """Test Meta-Learning framework"""
    
    @pytest.mark.asyncio
    async def test_meta_learner_initialization(self):
        """Test meta-learner creation"""
        meta = MetaLearner()
        
        assert 'learning_rate' in meta.meta_parameters
        assert 'exploration_rate' in meta.meta_parameters
    
    @pytest.mark.asyncio
    async def test_task_adaptation(self):
        """Test adaptation to new task"""
        meta = MetaLearner()
        
        context = {
            'complexity': 0.7,
            'urgency': 0.5,
            'resource_requirements': 0.6
        }
        
        adapted = await meta.adapt_to_task('classification', context)
        
        assert isinstance(adapted, dict)
        assert 'learning_rate' in adapted
    
    def test_task_embedding_generation(self):
        """Test task embedding creation"""
        meta = MetaLearner()
        
        embedding = meta._generate_task_embedding(
            'test_task',
            {'complexity': 0.5}
        )
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 5
    
    def test_knowledge_transfer(self):
        """Test knowledge transfer from similar tasks"""
        meta = MetaLearner()
        
        # Add some task history
        meta.strategy_performance['task1'] = {
            'learning_rate': 0.01,
            'batch_size': 64
        }
        
        similar_tasks = [('task1', 0.9)]
        transferred = meta._transfer_knowledge(similar_tasks)
        
        assert 'learning_rate' in transferred
        assert transferred['learning_rate'] > 0


class TestContinuousLearner:
    """Test Continuous Learning system"""
    
    @pytest.mark.asyncio
    async def test_continuous_learner_initialization(self):
        """Test continuous learner creation"""
        learner = ContinuousLearner()
        
        assert hasattr(learner, 'knowledge_base')
        assert hasattr(learner, 'skill_registry')
        assert hasattr(learner, 'elastic_weights')
    
    @pytest.mark.asyncio
    async def test_incremental_learning(self):
        """Test incremental knowledge acquisition"""
        learner = ContinuousLearner()
        
        new_knowledge = {
            'fact1': 'value1',
            'fact2': 0.5
        }
        
        await learner.learn_incrementally(new_knowledge, importance=0.8)
        
        assert 'fact1' in learner.knowledge_base
        assert 'fact2' in learner.knowledge_base
    
    @pytest.mark.asyncio
    async def test_catastrophic_forgetting_prevention(self):
        """Test EWC for preventing forgetting"""
        learner = ContinuousLearner()
        
        # Learn initial knowledge
        initial = {'param': 1.0}
        await learner.learn_incrementally(initial, importance=1.0)
        
        # Learn new knowledge
        new = {'param': 2.0}
        await learner.learn_incrementally(new, importance=0.5)
        
        # Check knowledge was consolidated, not replaced
        assert learner.knowledge_base['param'] != 2.0
        assert learner.elastic_weights['param'] > 1.0
    
    def test_skill_tracking(self):
        """Test skill level tracking"""
        learner = ContinuousLearner()
        
        # Update skills
        learner._update_skills({'skill_coding': True})
        
        assert 'coding' in learner.skill_registry
        assert learner.get_skill_level('coding') == 1
    
    def test_learning_progress(self):
        """Test progress reporting"""
        learner = ContinuousLearner()
        
        progress = learner.get_learning_progress()
        
        assert 'total_knowledge' in progress
        assert 'total_skills' in progress
        assert 'average_skill_level' in progress


class TestLearningOrchestrator:
    """Test main learning orchestrator"""
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self):
        """Test orchestrator creation"""
        orchestrator = LearningOrchestrator()
        
        assert hasattr(orchestrator, 'rl_agent')
        assert hasattr(orchestrator, 'meta_learner')
        assert hasattr(orchestrator, 'continuous_learner')
        assert orchestrator.learning_enabled
    
    @pytest.mark.asyncio
    async def test_learn_from_interaction(self):
        """Test learning from single interaction"""
        orchestrator = LearningOrchestrator()
        
        state = Mock(spec=State)
        state.to_vector.return_value = np.zeros(25)
        
        next_state = Mock(spec=State)
        next_state.to_vector.return_value = np.ones(25)
        
        await orchestrator.learn_from_interaction(
            state=state,
            action=ActionSpace.SELECT_MODULE.value,
            reward=1.0,
            next_state=next_state,
            done=False,
            context={'task_type': 'test'}
        )
        
        # Check experience was stored
        assert len(orchestrator.rl_agent.replay_buffer.buffer) == 1
    
    @pytest.mark.asyncio
    async def test_batch_learning(self):
        """Test batch learning mode"""
        orchestrator = LearningOrchestrator()
        orchestrator.adaptation_mode = "offline"
        
        # Add some experiences first
        state = Mock(spec=State)
        state.to_vector.return_value = np.zeros(25)
        
        for i in range(10):
            exp = Experience(
                state=state,
                action=ActionSpace.SELECT_MODULE.value,
                reward=1.0,
                next_state=state,
                done=False
            )
            orchestrator.rl_agent.replay_buffer.add(exp)
        
        await orchestrator.batch_learning(batch_size=5)
        
        # Should have triggered learning
        assert orchestrator.rl_agent.epsilon < 1.0
    
    def test_learning_status(self):
        """Test status reporting"""
        orchestrator = LearningOrchestrator()
        
        status = orchestrator.get_learning_status()
        
        assert 'rl_agent' in status
        assert 'meta_learner' in status
        assert 'continuous_learner' in status
        assert 'performance' in status


class TestPerformanceMonitor:
    """Test performance monitoring"""
    
    def test_monitor_initialization(self):
        """Test monitor creation"""
        monitor = PerformanceMonitor()
        
        assert hasattr(monitor, 'metrics')
        assert hasattr(monitor, 'action_counts')
        assert hasattr(monitor, 'reward_history')
    
    def test_recording_metrics(self):
        """Test metric recording"""
        monitor = PerformanceMonitor()
        
        monitor.record(
            action='test_action',
            reward=1.0,
            context={'response_time': 0.5}
        )
        
        assert monitor.action_counts['test_action'] == 1
        assert len(monitor.reward_history) == 1
        assert monitor.reward_history[0] == 1.0
    
    def test_performance_calculation(self):
        """Test performance metric calculation"""
        monitor = PerformanceMonitor()
        
        # Record some metrics
        for i in range(10):
            monitor.record(
                action='action',
                reward=1.0 if i % 2 == 0 else -1.0,
                context={}
            )
        
        perf = monitor.get_recent_performance(window=5)
        
        assert 'mean_reward' in perf
        assert 'success_rate' in perf
        assert 'total_actions' in perf
    
    def test_trend_calculation(self):
        """Test performance trend detection"""
        monitor = PerformanceMonitor()
        
        # Create improving trend
        for i in range(30):
            reward = i / 30.0  # Increasing rewards
            monitor.record('action', reward, {})
        
        trend = monitor._calculate_trend()
        assert trend == "improving"


class TestFeedbackLoop:
    """Test feedback loop system"""
    
    @pytest.mark.asyncio
    async def test_feedback_loop_initialization(self):
        """Test feedback loop creation"""
        orchestrator = LearningOrchestrator()
        feedback = FeedbackLoop(orchestrator)
        
        assert feedback.orchestrator == orchestrator
        assert hasattr(feedback, 'feedback_history')
    
    @pytest.mark.asyncio
    async def test_process_feedback(self):
        """Test feedback processing"""
        orchestrator = LearningOrchestrator()
        feedback = FeedbackLoop(orchestrator)
        
        adjustments = await feedback.process_feedback(
            feedback_type='performance',
            value=0.3,  # Poor performance
            context={}
        )
        
        assert 'increase_exploration' in adjustments
        assert len(feedback.feedback_history) == 1
    
    @pytest.mark.asyncio
    async def test_adjustment_application(self):
        """Test applying adjustments"""
        orchestrator = LearningOrchestrator()
        feedback = FeedbackLoop(orchestrator)
        
        initial_epsilon = orchestrator.rl_agent.epsilon
        
        adjustments = {'increase_exploration': True}
        await feedback._apply_adjustments(adjustments)
        
        # Check exploration increased
        assert orchestrator.rl_agent.epsilon > initial_epsilon
    
    @pytest.mark.asyncio
    async def test_error_rate_feedback(self):
        """Test error rate based adjustments"""
        orchestrator = LearningOrchestrator()
        feedback = FeedbackLoop(orchestrator)
        
        adjustments = await feedback.process_feedback(
            feedback_type='error_rate',
            value=0.2,  # High error rate
            context={}
        )
        
        assert 'enable_safety_checks' in adjustments
        assert 'increase_validation' in adjustments


class TestIntegration:
    """Integration tests for learning system"""
    
    @pytest.mark.asyncio
    async def test_full_learning_cycle(self):
        """Test complete learning cycle"""
        orchestrator = LearningOrchestrator()
        feedback_loop = FeedbackLoop(orchestrator)
        
        # Create states
        state1 = State(
            task_complexity=0.5,
            available_resources={'cpu': 0.8},
            module_performance={'vision': 0.9},
            context_embedding=np.zeros(10),
            temporal_features={},
            goal_alignment=0.7,
            uncertainty=0.3
        )
        
        state2 = State(
            task_complexity=0.6,
            available_resources={'cpu': 0.7},
            module_performance={'vision': 0.85},
            context_embedding=np.ones(10),
            temporal_features={},
            goal_alignment=0.8,
            uncertainty=0.2
        )
        
        # Simulate interaction
        action = orchestrator.rl_agent.select_action(state1)
        reward = 0.8
        
        # Learn from interaction
        await orchestrator.learn_from_interaction(
            state=state1,
            action=action,
            reward=reward,
            next_state=state2,
            done=False,
            context={'task_type': 'vision_task'}
        )
        
        # Process feedback
        await feedback_loop.process_feedback(
            feedback_type='performance',
            value=reward,
            context={'task': 'vision_task'}
        )
        
        # Check learning occurred
        status = orchestrator.get_learning_status()
        assert status['rl_agent']['buffer_size'] == 1
        assert len(feedback_loop.feedback_history) == 1
    
    @pytest.mark.asyncio
    async def test_adaptation_over_time(self):
        """Test system adaptation over multiple interactions"""
        orchestrator = LearningOrchestrator()
        
        # Generate mock states
        states = []
        for i in range(20):
            state = Mock(spec=State)
            state.to_vector.return_value = np.random.randn(25)
            states.append(state)
        
        # Simulate multiple interactions
        for i in range(len(states) - 1):
            action = orchestrator.rl_agent.select_action(states[i])
            reward = np.random.random()
            
            await orchestrator.learn_from_interaction(
                state=states[i],
                action=action,
                reward=reward,
                next_state=states[i + 1],
                done=(i == len(states) - 2),
                context={'episode': i}
            )
        
        # Check adaptation
        initial_epsilon = 1.0
        final_epsilon = orchestrator.rl_agent.epsilon
        
        assert final_epsilon < initial_epsilon  # Exploration decreased
        assert orchestrator.continuous_learner.get_learning_progress()['total_knowledge'] > 0
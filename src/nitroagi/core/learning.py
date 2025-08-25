"""
Learning and Adaptation System for NitroAGI NEXUS
Implements reinforcement learning, meta-learning, and continuous adaptation
"""

import asyncio
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import random
from collections import deque, defaultdict
import logging

logger = logging.getLogger(__name__)


class ActionSpace(Enum):
    """Available actions for the RL agent"""
    SELECT_MODULE = "select_module"
    ADJUST_PARAMETERS = "adjust_parameters"
    COMBINE_OUTPUTS = "combine_outputs"
    REQUEST_FEEDBACK = "request_feedback"
    CACHE_RESULT = "cache_result"
    PARALLELIZE = "parallelize"
    SERIALIZE = "serialize"
    RETRY = "retry"
    ESCALATE = "escalate"
    OPTIMIZE = "optimize"


@dataclass
class State:
    """Environment state representation"""
    task_complexity: float
    available_resources: Dict[str, float]
    module_performance: Dict[str, float]
    context_embedding: np.ndarray
    temporal_features: Dict[str, Any]
    goal_alignment: float
    uncertainty: float
    
    def to_vector(self) -> np.ndarray:
        """Convert state to feature vector"""
        features = [
            self.task_complexity,
            self.goal_alignment,
            self.uncertainty,
            sum(self.available_resources.values()),
            np.mean(list(self.module_performance.values()))
        ]
        
        # Add context embedding
        if self.context_embedding is not None:
            features.extend(self.context_embedding.tolist()[:10])  # Use first 10 dims
        
        return np.array(features)


@dataclass
class Experience:
    """Single experience tuple for replay"""
    state: State
    action: str
    reward: float
    next_state: State
    done: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class ExperienceReplay:
    """Experience replay buffer for stable learning"""
    
    def __init__(self, capacity: int = 10000, prioritized: bool = True):
        self.capacity = capacity
        self.prioritized = prioritized
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.episode_buffer = []
        
    def add(self, experience: Experience):
        """Add experience to buffer"""
        self.buffer.append(experience)
        
        if self.prioritized:
            # TD error as priority
            priority = abs(experience.reward) + 0.01
            self.priorities.append(priority)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        if self.prioritized:
            # Prioritized sampling
            probs = np.array(self.priorities) / sum(self.priorities)
            indices = np.random.choice(
                len(self.buffer), 
                batch_size, 
                p=probs
            )
            return [self.buffer[i] for i in indices]
        else:
            # Uniform sampling
            return random.sample(self.buffer, batch_size)
    
    def update_priorities(self, indices: List[int], td_errors: List[float]):
        """Update priorities based on TD errors"""
        if self.prioritized:
            for idx, error in zip(indices, td_errors):
                self.priorities[idx] = abs(error) + 0.01


class QNetwork:
    """Q-Network for value approximation"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Simple neural network with numpy
        self.weights = {
            'w1': np.random.randn(state_dim, hidden_dim) * 0.1,
            'b1': np.zeros(hidden_dim),
            'w2': np.random.randn(hidden_dim, hidden_dim) * 0.1,
            'b2': np.zeros(hidden_dim),
            'w3': np.random.randn(hidden_dim, action_dim) * 0.1,
            'b3': np.zeros(action_dim)
        }
        
        # Target network for stability
        self.target_weights = self.weights.copy()
        
    def forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass through network"""
        # Layer 1
        h1 = np.tanh(np.dot(state, self.weights['w1']) + self.weights['b1'])
        
        # Layer 2
        h2 = np.tanh(np.dot(h1, self.weights['w2']) + self.weights['b2'])
        
        # Output layer
        q_values = np.dot(h2, self.weights['w3']) + self.weights['b3']
        
        return q_values
    
    def update_target(self, tau: float = 0.01):
        """Soft update of target network"""
        for key in self.weights:
            self.target_weights[key] = (
                tau * self.weights[key] + 
                (1 - tau) * self.target_weights[key]
            )


class ReinforcementLearner:
    """Deep Q-Learning agent for NEXUS"""
    
    def __init__(
        self,
        state_dim: int = 25,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        self.actions = list(ActionSpace)
        self.action_dim = len(self.actions)
        
        # Q-Networks
        self.q_network = QNetwork(state_dim, self.action_dim)
        self.target_network = QNetwork(state_dim, self.action_dim)
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Experience replay
        self.replay_buffer = ExperienceReplay(capacity=10000)
        
        # Performance tracking
        self.episode_rewards = []
        self.learning_history = []
        
    def select_action(self, state: State, explore: bool = True) -> str:
        """Epsilon-greedy action selection"""
        if explore and random.random() < self.epsilon:
            # Exploration
            return random.choice(self.actions).value
        
        # Exploitation
        state_vector = state.to_vector()
        q_values = self.q_network.forward(state_vector)
        action_idx = np.argmax(q_values)
        
        return self.actions[action_idx].value
    
    def learn(self, batch_size: int = 32):
        """Update Q-network from experience batch"""
        if len(self.replay_buffer.buffer) < batch_size:
            return
        
        batch = self.replay_buffer.sample(batch_size)
        
        for experience in batch:
            state_vector = experience.state.to_vector()
            next_state_vector = experience.next_state.to_vector()
            
            # Current Q-values
            q_values = self.q_network.forward(state_vector)
            
            # Target Q-values
            next_q_values = self.target_network.forward(next_state_vector)
            
            # Calculate target
            action_idx = self.actions.index(
                ActionSpace(experience.action)
            )
            
            if experience.done:
                target = experience.reward
            else:
                target = experience.reward + self.gamma * np.max(next_q_values)
            
            # TD error
            td_error = target - q_values[action_idx]
            
            # Simple gradient update (simplified backprop)
            self._update_weights(state_vector, action_idx, td_error)
        
        # Update target network
        self.q_network.update_target()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _update_weights(self, state: np.ndarray, action: int, error: float):
        """Simplified weight update"""
        # This is a simplified version - in production, use proper autodiff
        lr = self.learning_rate * error
        
        # Forward pass to get activations
        h1 = np.tanh(np.dot(state, self.q_network.weights['w1']) + 
                     self.q_network.weights['b1'])
        h2 = np.tanh(np.dot(h1, self.q_network.weights['w2']) + 
                     self.q_network.weights['b2'])
        
        # Update output layer
        self.q_network.weights['w3'][:, action] += lr * h2
        self.q_network.weights['b3'][action] += lr
        
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'q_weights': self.q_network.weights,
            'target_weights': self.q_network.target_weights,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'learning_history': self.learning_history
        }
        
        with open(path, 'w') as f:
            json.dump(
                {k: v.tolist() if isinstance(v, np.ndarray) else v 
                 for k, v in checkpoint.items()},
                f
            )
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        with open(path, 'r') as f:
            checkpoint = json.load(f)
        
        self.q_network.weights = {
            k: np.array(v) for k, v in checkpoint['q_weights'].items()
        }
        self.q_network.target_weights = {
            k: np.array(v) for k, v in checkpoint['target_weights'].items()
        }
        self.epsilon = checkpoint['epsilon']
        self.episode_rewards = checkpoint['episode_rewards']
        self.learning_history = checkpoint['learning_history']


class MetaLearner:
    """Meta-learning framework for rapid adaptation"""
    
    def __init__(self):
        self.task_embeddings = {}
        self.strategy_performance = defaultdict(lambda: defaultdict(float))
        self.adaptation_history = []
        self.meta_parameters = {
            'learning_rate': 0.001,
            'exploration_rate': 0.1,
            'batch_size': 32,
            'update_frequency': 10
        }
        
    async def adapt_to_task(
        self, 
        task_type: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt learning strategy to new task"""
        
        # Generate task embedding
        embedding = self._generate_task_embedding(task_type, context)
        
        # Find similar tasks
        similar_tasks = self._find_similar_tasks(embedding)
        
        # Transfer knowledge
        adapted_params = self._transfer_knowledge(similar_tasks)
        
        # Update meta-parameters
        self._update_meta_parameters(task_type, adapted_params)
        
        return adapted_params
    
    def _generate_task_embedding(
        self, 
        task_type: str, 
        context: Dict[str, Any]
    ) -> np.ndarray:
        """Generate embedding for task"""
        # Simplified embedding generation
        features = [
            hash(task_type) % 100,
            len(context),
            context.get('complexity', 0.5),
            context.get('urgency', 0.5),
            context.get('resource_requirements', 0.5)
        ]
        
        return np.array(features)
    
    def _find_similar_tasks(
        self, 
        embedding: np.ndarray, 
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """Find k most similar previous tasks"""
        similarities = []
        
        for task_id, task_emb in self.task_embeddings.items():
            # Cosine similarity
            sim = np.dot(embedding, task_emb) / (
                np.linalg.norm(embedding) * np.linalg.norm(task_emb) + 1e-8
            )
            similarities.append((task_id, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:k]
    
    def _transfer_knowledge(
        self, 
        similar_tasks: List[Tuple[str, float]]
    ) -> Dict[str, Any]:
        """Transfer knowledge from similar tasks"""
        if not similar_tasks:
            return self.meta_parameters.copy()
        
        # Weighted average of parameters
        transferred = {}
        total_weight = sum(sim for _, sim in similar_tasks)
        
        for param_name in self.meta_parameters:
            weighted_sum = 0
            for task_id, similarity in similar_tasks:
                if task_id in self.strategy_performance:
                    param_value = self.strategy_performance[task_id].get(
                        param_name, 
                        self.meta_parameters[param_name]
                    )
                    weighted_sum += similarity * param_value
            
            transferred[param_name] = (
                weighted_sum / total_weight if total_weight > 0 
                else self.meta_parameters[param_name]
            )
        
        return transferred
    
    def _update_meta_parameters(
        self, 
        task_type: str, 
        adapted_params: Dict[str, Any]
    ):
        """Update meta-learning parameters"""
        # MAML-style meta-update (simplified)
        alpha = 0.01  # Meta-learning rate
        
        for param_name, param_value in adapted_params.items():
            old_value = self.meta_parameters[param_name]
            self.meta_parameters[param_name] = (
                old_value + alpha * (param_value - old_value)
            )
        
        # Store adaptation
        self.adaptation_history.append({
            'task_type': task_type,
            'adapted_params': adapted_params,
            'timestamp': datetime.now()
        })
    
    def evaluate_adaptation(
        self, 
        task_id: str, 
        performance_metrics: Dict[str, float]
    ):
        """Evaluate and store adaptation performance"""
        # Store performance for future transfer
        for metric, value in performance_metrics.items():
            self.strategy_performance[task_id][metric] = value
        
        # Prune old adaptations
        if len(self.adaptation_history) > 1000:
            self.adaptation_history = self.adaptation_history[-500:]


class ContinuousLearner:
    """Continuous learning with catastrophic forgetting prevention"""
    
    def __init__(self):
        self.knowledge_base = {}
        self.skill_registry = {}
        self.consolidation_queue = deque(maxlen=100)
        self.elastic_weights = {}
        self.learning_rate_schedule = self._create_schedule()
        
    def _create_schedule(self) -> Callable:
        """Create learning rate schedule"""
        def schedule(episode: int) -> float:
            # Cosine annealing
            initial_lr = 0.01
            min_lr = 0.0001
            period = 1000
            
            cycle = episode % period
            lr = min_lr + 0.5 * (initial_lr - min_lr) * (
                1 + np.cos(np.pi * cycle / period)
            )
            return lr
        
        return schedule
    
    async def learn_incrementally(
        self, 
        new_knowledge: Dict[str, Any],
        importance: float = 1.0
    ):
        """Learn new knowledge while preserving old"""
        
        # Elastic Weight Consolidation (EWC)
        await self._consolidate_knowledge(new_knowledge, importance)
        
        # Update skill registry
        self._update_skills(new_knowledge)
        
        # Add to consolidation queue
        self.consolidation_queue.append({
            'knowledge': new_knowledge,
            'importance': importance,
            'timestamp': datetime.now()
        })
        
        # Periodic consolidation
        if len(self.consolidation_queue) >= 10:
            await self._rehearse_old_knowledge()
    
    async def _consolidate_knowledge(
        self, 
        new_knowledge: Dict[str, Any],
        importance: float
    ):
        """Consolidate knowledge using EWC"""
        
        for key, value in new_knowledge.items():
            if key in self.knowledge_base:
                # Compute Fisher information (simplified)
                fisher_info = self._compute_fisher_information(key, value)
                
                # Update with regularization
                old_value = self.knowledge_base[key]
                elasticity = self.elastic_weights.get(key, 1.0)
                
                # Weighted update based on importance and elasticity
                update_weight = importance / (1 + elasticity * fisher_info)
                
                self.knowledge_base[key] = (
                    old_value * (1 - update_weight) + 
                    value * update_weight
                )
                
                # Update elasticity
                self.elastic_weights[key] = elasticity + fisher_info
            else:
                # New knowledge
                self.knowledge_base[key] = value
                self.elastic_weights[key] = importance
    
    def _compute_fisher_information(
        self, 
        key: str, 
        value: Any
    ) -> float:
        """Compute Fisher information for parameter"""
        # Simplified Fisher information approximation
        if key not in self.knowledge_base:
            return 1.0
        
        old_value = self.knowledge_base[key]
        
        # Compute gradient magnitude (simplified)
        if isinstance(value, (int, float)):
            return abs(value - old_value)
        elif isinstance(value, dict):
            return sum(
                abs(v - old_value.get(k, 0))
                for k, v in value.items()
                if isinstance(v, (int, float))
            )
        else:
            return 1.0
    
    def _update_skills(self, new_knowledge: Dict[str, Any]):
        """Update skill registry with new capabilities"""
        
        for key, value in new_knowledge.items():
            if 'skill' in key.lower() or 'capability' in key.lower():
                skill_name = key.replace('skill_', '').replace('capability_', '')
                
                if skill_name not in self.skill_registry:
                    self.skill_registry[skill_name] = {
                        'level': 1,
                        'experience': 0,
                        'last_used': datetime.now()
                    }
                else:
                    # Upgrade skill
                    self.skill_registry[skill_name]['experience'] += 1
                    self.skill_registry[skill_name]['last_used'] = datetime.now()
                    
                    # Level up
                    if self.skill_registry[skill_name]['experience'] % 10 == 0:
                        self.skill_registry[skill_name]['level'] += 1
    
    async def _rehearse_old_knowledge(self):
        """Rehearse old knowledge to prevent forgetting"""
        
        # Sample old knowledge
        if len(self.knowledge_base) > 10:
            rehearsal_keys = random.sample(
                list(self.knowledge_base.keys()), 
                min(5, len(self.knowledge_base))
            )
            
            for key in rehearsal_keys:
                # Pseudo-rehearsal: strengthen memory
                if key in self.elastic_weights:
                    self.elastic_weights[key] *= 0.95  # Reduce elasticity
        
        logger.info(f"Rehearsed {len(rehearsal_keys)} knowledge items")
    
    def get_skill_level(self, skill_name: str) -> int:
        """Get current skill level"""
        if skill_name in self.skill_registry:
            return self.skill_registry[skill_name]['level']
        return 0
    
    def get_learning_progress(self) -> Dict[str, Any]:
        """Get overall learning progress"""
        return {
            'total_knowledge': len(self.knowledge_base),
            'total_skills': len(self.skill_registry),
            'average_skill_level': np.mean([
                s['level'] for s in self.skill_registry.values()
            ]) if self.skill_registry else 0,
            'consolidation_queue_size': len(self.consolidation_queue),
            'elastic_weight_mean': np.mean(list(self.elastic_weights.values()))
            if self.elastic_weights else 0
        }


class LearningOrchestrator:
    """Main learning system orchestrator"""
    
    def __init__(self):
        self.rl_agent = ReinforcementLearner()
        self.meta_learner = MetaLearner()
        self.continuous_learner = ContinuousLearner()
        self.performance_monitor = PerformanceMonitor()
        
        self.learning_enabled = True
        self.adaptation_mode = "online"  # online, offline, hybrid
        
    async def learn_from_interaction(
        self,
        state: State,
        action: str,
        reward: float,
        next_state: State,
        done: bool,
        context: Dict[str, Any]
    ):
        """Main learning entry point"""
        
        if not self.learning_enabled:
            return
        
        # Store experience
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            metadata=context
        )
        
        self.rl_agent.replay_buffer.add(experience)
        
        # Online learning
        if self.adaptation_mode in ["online", "hybrid"]:
            # RL update
            self.rl_agent.learn()
            
            # Meta-learning adaptation
            adapted_params = await self.meta_learner.adapt_to_task(
                context.get('task_type', 'general'),
                context
            )
            
            # Continuous learning
            await self.continuous_learner.learn_incrementally(
                {'action': action, 'reward': reward},
                importance=abs(reward)
            )
        
        # Performance monitoring
        self.performance_monitor.record(action, reward, context)
        
    async def batch_learning(self, batch_size: int = 128):
        """Offline batch learning"""
        
        if self.adaptation_mode in ["offline", "hybrid"]:
            # Larger batch learning
            self.rl_agent.learn(batch_size=batch_size)
            
            # Consolidate continuous learning
            await self.continuous_learner._rehearse_old_knowledge()
            
            # Evaluate and adapt meta-parameters
            performance = self.performance_monitor.get_recent_performance()
            self.meta_learner.evaluate_adaptation(
                f"batch_{datetime.now().timestamp()}",
                performance
            )
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get comprehensive learning status"""
        return {
            'rl_agent': {
                'epsilon': self.rl_agent.epsilon,
                'buffer_size': len(self.rl_agent.replay_buffer.buffer),
                'episode_rewards': self.rl_agent.episode_rewards[-10:]
            },
            'meta_learner': {
                'meta_parameters': self.meta_learner.meta_parameters,
                'adaptations': len(self.meta_learner.adaptation_history)
            },
            'continuous_learner': self.continuous_learner.get_learning_progress(),
            'performance': self.performance_monitor.get_summary(),
            'learning_enabled': self.learning_enabled,
            'adaptation_mode': self.adaptation_mode
        }
    
    def save_state(self, path: str):
        """Save learning system state"""
        state = {
            'rl_checkpoint': f"{path}/rl_agent.json",
            'meta_state': self.meta_learner.__dict__,
            'continuous_state': {
                'knowledge_base': self.continuous_learner.knowledge_base,
                'skill_registry': self.continuous_learner.skill_registry
            },
            'performance': self.performance_monitor.get_summary()
        }
        
        # Save RL checkpoint
        self.rl_agent.save_checkpoint(state['rl_checkpoint'])
        
        # Save orchestrator state
        with open(f"{path}/learning_state.json", 'w') as f:
            json.dump(state, f, default=str)
    
    def load_state(self, path: str):
        """Load learning system state"""
        with open(f"{path}/learning_state.json", 'r') as f:
            state = json.load(f)
        
        # Load RL checkpoint
        self.rl_agent.load_checkpoint(state['rl_checkpoint'])
        
        # Restore meta-learner
        for key, value in state['meta_state'].items():
            setattr(self.meta_learner, key, value)
        
        # Restore continuous learner
        self.continuous_learner.knowledge_base = state['continuous_state']['knowledge_base']
        self.continuous_learner.skill_registry = state['continuous_state']['skill_registry']


class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.action_counts = defaultdict(int)
        self.reward_history = deque(maxlen=1000)
        self.success_rate = deque(maxlen=100)
        
    def record(self, action: str, reward: float, context: Dict[str, Any]):
        """Record performance metric"""
        self.action_counts[action] += 1
        self.reward_history.append(reward)
        self.success_rate.append(1 if reward > 0 else 0)
        
        # Record detailed metrics
        self.metrics['rewards'].append(reward)
        self.metrics['actions'].append(action)
        self.metrics['timestamps'].append(datetime.now())
        
        # Context-specific metrics
        if 'response_time' in context:
            self.metrics['response_times'].append(context['response_time'])
        if 'error' in context:
            self.metrics['errors'].append(context['error'])
    
    def get_recent_performance(self, window: int = 100) -> Dict[str, float]:
        """Get recent performance metrics"""
        recent_rewards = list(self.reward_history)[-window:]
        
        return {
            'mean_reward': np.mean(recent_rewards) if recent_rewards else 0,
            'std_reward': np.std(recent_rewards) if recent_rewards else 0,
            'success_rate': np.mean(list(self.success_rate)) if self.success_rate else 0,
            'total_actions': sum(self.action_counts.values()),
            'unique_actions': len(self.action_counts)
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            'total_interactions': len(self.metrics['rewards']),
            'action_distribution': dict(self.action_counts),
            'performance_metrics': self.get_recent_performance(),
            'trend': self._calculate_trend()
        }
    
    def _calculate_trend(self) -> str:
        """Calculate performance trend"""
        if len(self.reward_history) < 20:
            return "insufficient_data"
        
        recent = np.mean(list(self.reward_history)[-10:])
        older = np.mean(list(self.reward_history)[-20:-10])
        
        if recent > older * 1.1:
            return "improving"
        elif recent < older * 0.9:
            return "declining"
        else:
            return "stable"


# Feedback Loop System
class FeedbackLoop:
    """Performance feedback and adjustment system"""
    
    def __init__(self, orchestrator: LearningOrchestrator):
        self.orchestrator = orchestrator
        self.feedback_history = []
        self.adjustment_policies = {}
        
    async def process_feedback(
        self,
        feedback_type: str,
        value: float,
        context: Dict[str, Any]
    ):
        """Process performance feedback"""
        
        # Store feedback
        self.feedback_history.append({
            'type': feedback_type,
            'value': value,
            'context': context,
            'timestamp': datetime.now()
        })
        
        # Determine adjustments
        adjustments = await self._determine_adjustments(
            feedback_type, 
            value, 
            context
        )
        
        # Apply adjustments
        await self._apply_adjustments(adjustments)
        
        return adjustments
    
    async def _determine_adjustments(
        self,
        feedback_type: str,
        value: float,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine necessary adjustments"""
        
        adjustments = {}
        
        if feedback_type == "performance":
            if value < 0.5:  # Poor performance
                adjustments['increase_exploration'] = True
                adjustments['reduce_learning_rate'] = True
            elif value > 0.8:  # Good performance
                adjustments['reduce_exploration'] = True
                adjustments['increase_batch_size'] = True
        
        elif feedback_type == "error_rate":
            if value > 0.1:  # High error rate
                adjustments['enable_safety_checks'] = True
                adjustments['increase_validation'] = True
        
        elif feedback_type == "user_satisfaction":
            if value < 0.7:
                adjustments['adjust_strategy'] = True
                adjustments['request_clarification'] = True
        
        return adjustments
    
    async def _apply_adjustments(self, adjustments: Dict[str, Any]):
        """Apply adjustments to learning system"""
        
        if adjustments.get('increase_exploration'):
            self.orchestrator.rl_agent.epsilon = min(
                1.0, 
                self.orchestrator.rl_agent.epsilon * 1.5
            )
        
        if adjustments.get('reduce_exploration'):
            self.orchestrator.rl_agent.epsilon = max(
                0.01,
                self.orchestrator.rl_agent.epsilon * 0.8
            )
        
        if adjustments.get('reduce_learning_rate'):
            self.orchestrator.rl_agent.learning_rate *= 0.9
        
        if adjustments.get('increase_batch_size'):
            # Adjust batch size for next learning cycle
            self.adjustment_policies['batch_size'] = 256
        
        logger.info(f"Applied adjustments: {adjustments}")
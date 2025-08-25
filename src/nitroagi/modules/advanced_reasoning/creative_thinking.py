"""
Creative Thinking Module for NitroAGI NEXUS
Handles creative problem solving, idea generation, and innovative thinking
"""

import asyncio
import random
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from datetime import datetime
from collections import defaultdict
import itertools

logger = logging.getLogger(__name__)


class CreativeStrategy(Enum):
    """Creative thinking strategies"""
    BRAINSTORMING = "brainstorming"
    LATERAL_THINKING = "lateral_thinking"
    SCAMPER = "scamper"  # Substitute, Combine, Adapt, Modify, Put to another use, Eliminate, Reverse
    MIND_MAPPING = "mind_mapping"
    ANALOGICAL_THINKING = "analogical_thinking"
    RANDOM_STIMULATION = "random_stimulation"
    MORPHOLOGICAL_ANALYSIS = "morphological_analysis"
    SYNECTICS = "synectics"
    SIX_THINKING_HATS = "six_thinking_hats"
    DESIGN_THINKING = "design_thinking"


class ThinkingHat(Enum):
    """De Bono's Six Thinking Hats"""
    WHITE = "facts_and_information"
    RED = "emotions_and_intuition"
    BLACK = "critical_and_cautious"
    YELLOW = "optimistic_and_positive"
    GREEN = "creative_and_alternative"
    BLUE = "process_and_control"


@dataclass
class Idea:
    """Represents a creative idea"""
    content: str
    strategy_used: CreativeStrategy
    originality_score: float
    feasibility_score: float
    impact_score: float
    connections: List[str] = field(default_factory=list)
    variations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CreativeProblem:
    """Problem requiring creative solution"""
    description: str
    constraints: List[str]
    goals: List[str]
    context: Dict[str, Any]
    domain: str


@dataclass
class CreativeSolution:
    """Creative solution to a problem"""
    problem: CreativeProblem
    ideas: List[Idea]
    selected_idea: Optional[Idea]
    implementation_plan: List[str]
    innovation_level: float
    confidence: float


class IdeaGenerator:
    """Generates creative ideas using various techniques"""
    
    def __init__(self):
        self.idea_database = []
        self.word_associations = defaultdict(list)
        self.creative_patterns = []
        self._initialize_creative_resources()
    
    def _initialize_creative_resources(self):
        """Initialize creative thinking resources"""
        
        # Common word associations for random stimulation
        self.word_associations = {
            "connect": ["bridge", "link", "network", "join", "merge"],
            "transform": ["change", "evolve", "morph", "adapt", "convert"],
            "simplify": ["reduce", "streamline", "clarify", "minimize", "focus"],
            "expand": ["grow", "extend", "broaden", "multiply", "scale"],
            "combine": ["merge", "blend", "fuse", "integrate", "synthesize"]
        }
        
        # Creative patterns
        self.creative_patterns = [
            "What if we reversed {X}?",
            "How might we combine {X} with {Y}?",
            "What would happen if {X} was 10x bigger/smaller?",
            "How would {domain} experts solve this?",
            "What if the constraint was actually the solution?"
        ]
    
    async def brainstorm(
        self,
        problem: CreativeProblem,
        quantity: int = 10
    ) -> List[Idea]:
        """Generate ideas through brainstorming"""
        
        ideas = []
        
        # Generate ideas without judgment
        for i in range(quantity):
            # Use different prompts
            prompts = [
                f"Direct solution to: {problem.description}",
                f"Opposite approach to: {problem.description}",
                f"Combining elements: {' + '.join(problem.goals[:2])}",
                f"Removing constraints: without {problem.constraints[0] if problem.constraints else 'limits'}",
                f"Extreme version: {problem.description} taken to extreme"
            ]
            
            prompt = prompts[i % len(prompts)]
            
            idea_content = await self._generate_idea_from_prompt(prompt, problem)
            
            idea = Idea(
                content=idea_content,
                strategy_used=CreativeStrategy.BRAINSTORMING,
                originality_score=random.uniform(0.5, 1.0),
                feasibility_score=random.uniform(0.3, 0.9),
                impact_score=random.uniform(0.4, 1.0)
            )
            
            ideas.append(idea)
            self.idea_database.append(idea)
        
        return ideas
    
    async def _generate_idea_from_prompt(
        self,
        prompt: str,
        problem: CreativeProblem
    ) -> str:
        """Generate idea from a specific prompt"""
        
        # Template-based idea generation
        templates = [
            f"Use {random.choice(['AI', 'automation', 'crowdsourcing', 'gamification'])} to {prompt}",
            f"Create a {random.choice(['platform', 'tool', 'system', 'framework'])} that {prompt}",
            f"Implement {random.choice(['modular', 'distributed', 'adaptive', 'self-organizing'])} approach for {prompt}",
            f"Apply {random.choice(['biomimicry', 'quantum', 'fractal', 'emergent'])} principles to {prompt}"
        ]
        
        return random.choice(templates)
    
    async def lateral_thinking(
        self,
        problem: CreativeProblem
    ) -> List[Idea]:
        """Apply lateral thinking techniques"""
        
        ideas = []
        
        # Random entry technique
        random_word = random.choice(list(self.word_associations.keys()))
        associations = self.word_associations[random_word]
        
        for association in associations[:3]:
            idea_content = f"Apply concept of '{association}' to {problem.description}"
            
            idea = Idea(
                content=idea_content,
                strategy_used=CreativeStrategy.LATERAL_THINKING,
                originality_score=random.uniform(0.7, 1.0),
                feasibility_score=random.uniform(0.3, 0.7),
                impact_score=random.uniform(0.5, 0.9),
                connections=[random_word, association]
            )
            
            ideas.append(idea)
        
        # Challenge assumptions
        if problem.constraints:
            for constraint in problem.constraints[:2]:
                idea_content = f"What if '{constraint}' was not a constraint but a feature?"
                
                idea = Idea(
                    content=idea_content,
                    strategy_used=CreativeStrategy.LATERAL_THINKING,
                    originality_score=0.9,
                    feasibility_score=0.5,
                    impact_score=0.8
                )
                
                ideas.append(idea)
        
        self.idea_database.extend(ideas)
        return ideas
    
    async def scamper_technique(
        self,
        problem: CreativeProblem,
        existing_solution: Optional[str] = None
    ) -> List[Idea]:
        """Apply SCAMPER technique"""
        
        ideas = []
        base = existing_solution or problem.description
        
        scamper_operations = {
            "Substitute": f"Replace key component in {base} with alternative",
            "Combine": f"Merge {base} with complementary solution",
            "Adapt": f"Adapt {base} for different context or user",
            "Modify": f"Enhance or exaggerate aspect of {base}",
            "Put to another use": f"Apply {base} to completely different problem",
            "Eliminate": f"Remove unnecessary parts from {base}",
            "Reverse": f"Invert or reverse the process in {base}"
        }
        
        for operation, idea_content in scamper_operations.items():
            idea = Idea(
                content=idea_content,
                strategy_used=CreativeStrategy.SCAMPER,
                originality_score=random.uniform(0.6, 0.9),
                feasibility_score=random.uniform(0.5, 0.9),
                impact_score=random.uniform(0.4, 0.8)
            )
            
            ideas.append(idea)
        
        self.idea_database.extend(ideas)
        return ideas
    
    async def analogical_creation(
        self,
        problem: CreativeProblem,
        source_domains: List[str]
    ) -> List[Idea]:
        """Generate ideas through analogical thinking"""
        
        ideas = []
        
        for domain in source_domains:
            # Create analogy
            analogy_prompt = f"How would {domain} solve {problem.description}?"
            
            # Generate domain-specific solutions
            domain_solutions = {
                "nature": "Use evolutionary/adaptive approach like natural selection",
                "music": "Create harmony and rhythm between components",
                "architecture": "Build modular, scalable foundation",
                "cooking": "Combine ingredients with right timing and temperature",
                "sports": "Train, practice, and optimize performance iteratively",
                "art": "Express solution through visual/aesthetic principles"
            }
            
            idea_content = domain_solutions.get(
                domain.lower(),
                f"Apply {domain} principles to problem"
            )
            
            idea = Idea(
                content=idea_content,
                strategy_used=CreativeStrategy.ANALOGICAL_THINKING,
                originality_score=random.uniform(0.7, 1.0),
                feasibility_score=random.uniform(0.4, 0.8),
                impact_score=random.uniform(0.5, 0.9),
                connections=[domain, problem.domain]
            )
            
            ideas.append(idea)
        
        self.idea_database.extend(ideas)
        return ideas


class CreativeEvaluator:
    """Evaluates and refines creative ideas"""
    
    def __init__(self):
        self.evaluation_criteria = {
            "originality": 0.3,
            "feasibility": 0.3,
            "impact": 0.4
        }
    
    async def evaluate_ideas(
        self,
        ideas: List[Idea],
        problem: CreativeProblem
    ) -> List[Tuple[Idea, float]]:
        """Evaluate ideas against problem criteria"""
        
        evaluated = []
        
        for idea in ideas:
            score = await self._calculate_idea_score(idea, problem)
            evaluated.append((idea, score))
        
        # Sort by score
        evaluated.sort(key=lambda x: x[1], reverse=True)
        
        return evaluated
    
    async def _calculate_idea_score(
        self,
        idea: Idea,
        problem: CreativeProblem
    ) -> float:
        """Calculate overall score for an idea"""
        
        # Weighted score
        score = (
            idea.originality_score * self.evaluation_criteria["originality"] +
            idea.feasibility_score * self.evaluation_criteria["feasibility"] +
            idea.impact_score * self.evaluation_criteria["impact"]
        )
        
        # Bonus for addressing multiple goals
        goals_addressed = sum(
            1 for goal in problem.goals
            if goal.lower() in idea.content.lower()
        )
        
        score += goals_addressed * 0.1
        
        # Penalty for violating constraints
        constraints_violated = sum(
            1 for constraint in problem.constraints
            if f"not {constraint}" in idea.content.lower()
        )
        
        score -= constraints_violated * 0.2
        
        return min(1.0, max(0.0, score))
    
    async def refine_idea(
        self,
        idea: Idea,
        feedback: Optional[str] = None
    ) -> Idea:
        """Refine and improve an idea"""
        
        refined = Idea(
            content=idea.content,
            strategy_used=idea.strategy_used,
            originality_score=idea.originality_score,
            feasibility_score=idea.feasibility_score,
            impact_score=idea.impact_score,
            connections=idea.connections.copy(),
            variations=idea.variations.copy()
        )
        
        # Add variations
        variations = [
            f"Simplified version: {self._simplify(idea.content)}",
            f"Enhanced version: {self._enhance(idea.content)}",
            f"Combined approach: {self._combine_with_random(idea.content)}"
        ]
        
        refined.variations.extend(variations)
        
        # Adjust scores based on refinement
        if feedback:
            if "more original" in feedback.lower():
                refined.originality_score = min(1.0, refined.originality_score + 0.2)
            if "more practical" in feedback.lower():
                refined.feasibility_score = min(1.0, refined.feasibility_score + 0.2)
        
        return refined
    
    def _simplify(self, content: str) -> str:
        """Simplify an idea"""
        simplifiers = ["core", "essential", "basic", "fundamental"]
        return f"Focus on {random.choice(simplifiers)} aspect of {content}"
    
    def _enhance(self, content: str) -> str:
        """Enhance an idea"""
        enhancers = ["AI-powered", "automated", "intelligent", "adaptive"]
        return f"Add {random.choice(enhancers)} capabilities to {content}"
    
    def _combine_with_random(self, content: str) -> str:
        """Combine with random element"""
        elements = ["blockchain", "IoT", "VR/AR", "quantum", "social"]
        return f"Integrate {random.choice(elements)} with {content}"


class SixHatsAnalyzer:
    """Analyzes ideas using Six Thinking Hats method"""
    
    def __init__(self):
        self.perspectives = {}
    
    async def analyze_with_all_hats(
        self,
        idea: Idea,
        problem: CreativeProblem
    ) -> Dict[ThinkingHat, str]:
        """Analyze idea from all six perspectives"""
        
        analysis = {}
        
        # White Hat - Facts and Information
        analysis[ThinkingHat.WHITE] = await self._white_hat_analysis(idea, problem)
        
        # Red Hat - Emotions and Intuition
        analysis[ThinkingHat.RED] = await self._red_hat_analysis(idea, problem)
        
        # Black Hat - Critical and Cautious
        analysis[ThinkingHat.BLACK] = await self._black_hat_analysis(idea, problem)
        
        # Yellow Hat - Optimistic and Positive
        analysis[ThinkingHat.YELLOW] = await self._yellow_hat_analysis(idea, problem)
        
        # Green Hat - Creative and Alternative
        analysis[ThinkingHat.GREEN] = await self._green_hat_analysis(idea, problem)
        
        # Blue Hat - Process and Control
        analysis[ThinkingHat.BLUE] = await self._blue_hat_analysis(idea, problem)
        
        self.perspectives[idea.content] = analysis
        
        return analysis
    
    async def _white_hat_analysis(self, idea: Idea, problem: CreativeProblem) -> str:
        """Factual analysis"""
        facts = [
            f"Originality: {idea.originality_score:.2f}",
            f"Feasibility: {idea.feasibility_score:.2f}",
            f"Impact: {idea.impact_score:.2f}",
            f"Strategy: {idea.strategy_used.value}"
        ]
        return f"Facts: {'; '.join(facts)}"
    
    async def _red_hat_analysis(self, idea: Idea, problem: CreativeProblem) -> str:
        """Emotional/intuitive analysis"""
        emotions = ["exciting", "concerning", "promising", "risky", "innovative"]
        return f"This idea feels {random.choice(emotions)} and {random.choice(['bold', 'cautious', 'transformative'])}"
    
    async def _black_hat_analysis(self, idea: Idea, problem: CreativeProblem) -> str:
        """Critical analysis"""
        risks = [
            "May face implementation challenges",
            "Could exceed resource constraints",
            "Might not scale effectively",
            "Depends on uncertain factors"
        ]
        return f"Caution: {random.choice(risks)}"
    
    async def _yellow_hat_analysis(self, idea: Idea, problem: CreativeProblem) -> str:
        """Optimistic analysis"""
        benefits = [
            "Could revolutionize the approach",
            "Offers significant improvements",
            "Creates new opportunities",
            "Provides competitive advantage"
        ]
        return f"Benefit: {random.choice(benefits)}"
    
    async def _green_hat_analysis(self, idea: Idea, problem: CreativeProblem) -> str:
        """Creative alternatives"""
        alternatives = [
            "Consider hybrid approach",
            "Explore parallel implementations",
            "Test with pilot program",
            "Combine with existing solutions"
        ]
        return f"Alternative: {random.choice(alternatives)}"
    
    async def _blue_hat_analysis(self, idea: Idea, problem: CreativeProblem) -> str:
        """Process control analysis"""
        next_steps = [
            "Prototype and test",
            "Gather stakeholder feedback",
            "Conduct feasibility study",
            "Develop implementation roadmap"
        ]
        return f"Next step: {random.choice(next_steps)}"


class DesignThinkingEngine:
    """Implements design thinking methodology"""
    
    def __init__(self):
        self.stages = [
            "empathize",
            "define",
            "ideate",
            "prototype",
            "test"
        ]
        self.insights = defaultdict(list)
    
    async def empathize_stage(
        self,
        problem: CreativeProblem
    ) -> Dict[str, Any]:
        """Understand user needs and context"""
        
        empathy_map = {
            "user_needs": self._identify_needs(problem),
            "pain_points": self._identify_pain_points(problem),
            "emotions": ["frustrated", "hopeful", "confused", "motivated"],
            "context": problem.context,
            "stakeholders": self._identify_stakeholders(problem)
        }
        
        self.insights["empathize"] = empathy_map
        return empathy_map
    
    def _identify_needs(self, problem: CreativeProblem) -> List[str]:
        """Identify user needs"""
        base_needs = ["efficiency", "simplicity", "reliability", "flexibility"]
        
        # Add problem-specific needs
        if "speed" in problem.description.lower():
            base_needs.append("performance")
        if "easy" in problem.description.lower():
            base_needs.append("usability")
        
        return base_needs[:3]
    
    def _identify_pain_points(self, problem: CreativeProblem) -> List[str]:
        """Identify pain points"""
        pain_points = []
        
        for constraint in problem.constraints:
            pain_points.append(f"Limited by {constraint}")
        
        pain_points.append("Current solution inadequate")
        
        return pain_points
    
    def _identify_stakeholders(self, problem: CreativeProblem) -> List[str]:
        """Identify stakeholders"""
        return ["end users", "administrators", "developers", "management"]
    
    async def define_stage(
        self,
        empathy_insights: Dict[str, Any]
    ) -> str:
        """Define the problem clearly"""
        
        needs = empathy_insights.get("user_needs", [])
        pain_points = empathy_insights.get("pain_points", [])
        
        problem_statement = (
            f"Users need {' and '.join(needs[:2])} "
            f"because they face {pain_points[0] if pain_points else 'challenges'}"
        )
        
        self.insights["define"] = problem_statement
        return problem_statement
    
    async def ideate_stage(
        self,
        problem_statement: str,
        quantity: int = 5
    ) -> List[str]:
        """Generate solution ideas"""
        
        ideas = []
        
        techniques = [
            "How might we",
            "What if",
            "In what ways might we"
        ]
        
        for i in range(quantity):
            technique = techniques[i % len(techniques)]
            idea = f"{technique} {problem_statement.lower()}"
            ideas.append(idea)
        
        self.insights["ideate"] = ideas
        return ideas
    
    async def prototype_stage(
        self,
        selected_idea: str
    ) -> Dict[str, Any]:
        """Create prototype plan"""
        
        prototype = {
            "concept": selected_idea,
            "components": [
                "User interface mockup",
                "Core functionality demo",
                "Integration points"
            ],
            "timeline": "2 weeks",
            "resources_needed": [
                "Development environment",
                "Test data",
                "User feedback tools"
            ],
            "success_metrics": [
                "User task completion rate",
                "Performance benchmarks",
                "Error rate"
            ]
        }
        
        self.insights["prototype"] = prototype
        return prototype
    
    async def test_stage(
        self,
        prototype: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Plan testing approach"""
        
        test_plan = {
            "test_types": [
                "User acceptance testing",
                "Performance testing",
                "Integration testing"
            ],
            "test_scenarios": [
                "Happy path workflow",
                "Edge cases",
                "Error conditions"
            ],
            "feedback_methods": [
                "User interviews",
                "Analytics tracking",
                "A/B testing"
            ],
            "iteration_plan": "Weekly sprints with continuous improvement"
        }
        
        self.insights["test"] = test_plan
        return test_plan


class CreativeThinkingEngine:
    """Main creative thinking coordinator"""
    
    def __init__(self):
        self.idea_generator = IdeaGenerator()
        self.evaluator = CreativeEvaluator()
        self.six_hats = SixHatsAnalyzer()
        self.design_thinking = DesignThinkingEngine()
        
        self.solution_history = []
    
    async def solve_creatively(
        self,
        problem_description: str,
        constraints: Optional[List[str]] = None,
        goals: Optional[List[str]] = None,
        strategy: Optional[CreativeStrategy] = None
    ) -> CreativeSolution:
        """Solve problem using creative thinking"""
        
        logger.info(f"Creative problem solving: {problem_description[:50]}...")
        
        # Create problem object
        problem = CreativeProblem(
            description=problem_description,
            constraints=constraints or [],
            goals=goals or ["solve efficiently", "minimize cost", "maximize impact"],
            context={},
            domain="general"
        )
        
        # Generate ideas using multiple strategies
        all_ideas = []
        
        if strategy:
            strategies = [strategy]
        else:
            strategies = [
                CreativeStrategy.BRAINSTORMING,
                CreativeStrategy.LATERAL_THINKING,
                CreativeStrategy.SCAMPER
            ]
        
        for strat in strategies:
            if strat == CreativeStrategy.BRAINSTORMING:
                ideas = await self.idea_generator.brainstorm(problem, quantity=5)
            elif strat == CreativeStrategy.LATERAL_THINKING:
                ideas = await self.idea_generator.lateral_thinking(problem)
            elif strat == CreativeStrategy.SCAMPER:
                ideas = await self.idea_generator.scamper_technique(problem)
            elif strat == CreativeStrategy.ANALOGICAL_THINKING:
                ideas = await self.idea_generator.analogical_creation(
                    problem,
                    ["nature", "music", "architecture"]
                )
            else:
                ideas = await self.idea_generator.brainstorm(problem, quantity=3)
            
            all_ideas.extend(ideas)
        
        # Evaluate ideas
        evaluated = await self.evaluator.evaluate_ideas(all_ideas, problem)
        
        # Select best idea
        if evaluated:
            selected_idea, score = evaluated[0]
            
            # Refine selected idea
            selected_idea = await self.evaluator.refine_idea(selected_idea)
            
            # Analyze with Six Hats
            perspectives = await self.six_hats.analyze_with_all_hats(
                selected_idea,
                problem
            )
            
            # Create implementation plan
            implementation_plan = await self._create_implementation_plan(
                selected_idea,
                problem
            )
        else:
            selected_idea = None
            implementation_plan = []
        
        # Calculate innovation level
        innovation_level = self._calculate_innovation_level(all_ideas)
        
        solution = CreativeSolution(
            problem=problem,
            ideas=all_ideas[:10],  # Top 10 ideas
            selected_idea=selected_idea,
            implementation_plan=implementation_plan,
            innovation_level=innovation_level,
            confidence=score if evaluated else 0.0
        )
        
        self.solution_history.append(solution)
        
        return solution
    
    async def _create_implementation_plan(
        self,
        idea: Idea,
        problem: CreativeProblem
    ) -> List[str]:
        """Create implementation plan for idea"""
        
        plan = [
            "1. Validate concept with stakeholders",
            "2. Create detailed design specification",
            "3. Develop proof of concept",
            "4. Test with small user group",
            "5. Iterate based on feedback",
            "6. Scale implementation",
            "7. Monitor and optimize"
        ]
        
        # Customize based on idea characteristics
        if idea.feasibility_score < 0.5:
            plan.insert(2, "2a. Conduct feasibility study")
        
        if idea.originality_score > 0.8:
            plan.insert(1, "1a. File for intellectual property protection")
        
        return plan
    
    def _calculate_innovation_level(self, ideas: List[Idea]) -> float:
        """Calculate overall innovation level"""
        
        if not ideas:
            return 0.0
        
        avg_originality = np.mean([idea.originality_score for idea in ideas])
        max_originality = max(idea.originality_score for idea in ideas)
        
        # Weight maximum more than average
        innovation = 0.4 * avg_originality + 0.6 * max_originality
        
        return innovation
    
    async def design_thinking_process(
        self,
        problem_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Complete design thinking process"""
        
        problem = CreativeProblem(
            description=problem_description,
            constraints=[],
            goals=["user-centered solution"],
            context=context or {},
            domain="design"
        )
        
        # Empathize
        empathy_insights = await self.design_thinking.empathize_stage(problem)
        
        # Define
        problem_statement = await self.design_thinking.define_stage(empathy_insights)
        
        # Ideate
        ideas = await self.design_thinking.ideate_stage(problem_statement)
        
        # Prototype
        prototype = await self.design_thinking.prototype_stage(ideas[0] if ideas else "")
        
        # Test
        test_plan = await self.design_thinking.test_stage(prototype)
        
        return {
            "process": "design_thinking",
            "stages": {
                "empathize": empathy_insights,
                "define": problem_statement,
                "ideate": ideas,
                "prototype": prototype,
                "test": test_plan
            },
            "insights": dict(self.design_thinking.insights)
        }
    
    def get_creative_capabilities(self) -> Dict[str, Any]:
        """Get creative thinking capabilities"""
        
        return {
            "strategies": [s.value for s in CreativeStrategy],
            "thinking_hats": [h.value for h in ThinkingHat],
            "design_thinking_stages": self.design_thinking.stages,
            "total_ideas_generated": len(self.idea_generator.idea_database),
            "solution_count": len(self.solution_history)
        }
"""
Reasoning Module Implementation for NitroAGI NEXUS
Handles symbolic AI, logic programming, and advanced reasoning
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from nitroagi.core.base import (
    AIModule,
    ModuleCapability,
    ModuleRequest,
    ModuleResponse,
    ProcessingContext
)
from nitroagi.core.exceptions import ModuleException
from nitroagi.utils.logging import get_logger

# Import reasoning engines
from .engines import (
    LogicEngine,
    CausalReasoner,
    KnowledgeGraph,
    ProblemSolver,
    InferenceEngine
)


class ReasoningTask(Enum):
    """Types of reasoning tasks."""
    LOGICAL_INFERENCE = "logical_inference"
    CAUSAL_ANALYSIS = "causal_analysis"
    PROBLEM_SOLVING = "problem_solving"
    KNOWLEDGE_QUERY = "knowledge_query"
    FACT_CHECKING = "fact_checking"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    PLANNING = "planning"
    DECISION_MAKING = "decision_making"
    CONSTRAINT_SATISFACTION = "constraint_satisfaction"
    ABSTRACT_REASONING = "abstract_reasoning"


@dataclass
class ReasoningResult:
    """Result from reasoning process."""
    task: ReasoningTask
    conclusion: Optional[str] = None
    proof: Optional[List[str]] = None
    confidence: float = 0.0
    facts_used: List[str] = None
    rules_applied: List[str] = None
    causal_chain: Optional[List[Dict[str, Any]]] = None
    solution: Optional[Any] = None
    explanation: Optional[str] = None
    metadata: Dict[str, Any] = None
    processing_time_ms: float = 0.0


class ReasoningModule(AIModule):
    """
    Reasoning module for NitroAGI NEXUS.
    Provides symbolic AI and logical reasoning capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the reasoning module.
        
        Args:
            config: Module configuration
        """
        super().__init__(config)
        self.logger = get_logger(__name__)
        
        # Initialize reasoning engines
        self.logic_engine = LogicEngine()
        self.causal_reasoner = CausalReasoner()
        self.knowledge_graph = KnowledgeGraph()
        self.problem_solver = ProblemSolver()
        self.inference_engine = InferenceEngine()
        
        # Configuration
        self.max_inference_depth = config.get("max_inference_depth", 10)
        self.confidence_threshold = config.get("confidence_threshold", 0.6)
        self.use_probabilistic = config.get("use_probabilistic", True)
        self.enable_learning = config.get("enable_learning", True)
        
        # Knowledge base
        self.knowledge_base = {
            "facts": [],
            "rules": [],
            "constraints": [],
            "causal_models": []
        }
    
    async def initialize(self) -> bool:
        """
        Initialize the reasoning module and load knowledge base.
        
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing Reasoning Module...")
            
            # Initialize engines
            await self.logic_engine.initialize()
            await self.causal_reasoner.initialize()
            await self.knowledge_graph.initialize()
            await self.problem_solver.initialize()
            await self.inference_engine.initialize()
            
            # Load default knowledge base
            await self._load_knowledge_base()
            
            self._initialized = True
            self.logger.info("Reasoning Module initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Reasoning Module: {e}")
            self._initialized = False
            return False
    
    async def process(self, request: ModuleRequest) -> ModuleResponse:
        """
        Process a reasoning request.
        
        Args:
            request: The reasoning request
            
        Returns:
            ModuleResponse with reasoning results
        """
        if not self._initialized:
            return ModuleResponse(
                request_id=request.id,
                module_name=self.name,
                status="error",
                error="Reasoning module not initialized"
            )
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Extract problem and context
            problem = self._extract_problem(request.data)
            context = self._extract_context(request)
            
            # Determine reasoning task
            task = self._determine_task(request, problem)
            
            # Process reasoning task
            result = await self._process_reasoning_task(problem, task, context)
            
            # Build response
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            result.processing_time_ms = processing_time
            
            return ModuleResponse(
                request_id=request.id,
                module_name=self.name,
                status="success",
                data=self._format_result(result),
                processing_time_ms=processing_time,
                confidence_score=result.confidence
            )
            
        except Exception as e:
            self.logger.error(f"Reasoning failed: {e}", exc_info=True)
            return ModuleResponse(
                request_id=request.id,
                module_name=self.name,
                status="error",
                error=str(e),
                processing_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000
            )
    
    def _extract_problem(self, data: Any) -> Dict[str, Any]:
        """
        Extract problem statement from request data.
        
        Args:
            data: Request data
            
        Returns:
            Problem dictionary
        """
        if isinstance(data, str):
            return {"query": data, "type": "text"}
        
        elif isinstance(data, dict):
            return {
                "query": data.get("query", data.get("problem", "")),
                "facts": data.get("facts", []),
                "rules": data.get("rules", []),
                "constraints": data.get("constraints", []),
                "goal": data.get("goal"),
                "hypothesis": data.get("hypothesis"),
                "type": data.get("type", "general")
            }
        
        else:
            return {"query": str(data), "type": "unknown"}
    
    def _extract_context(self, request: ModuleRequest) -> Dict[str, Any]:
        """
        Extract context from request.
        
        Args:
            request: Module request
            
        Returns:
            Context dictionary
        """
        context = {
            "timestamp": datetime.utcnow(),
            "conversation_id": request.context.conversation_id,
            "metadata": request.context.metadata
        }
        
        # Add any domain-specific context
        if isinstance(request.data, dict):
            context["domain"] = request.data.get("domain", "general")
            context["variables"] = request.data.get("variables", {})
        
        return context
    
    def _determine_task(self, request: ModuleRequest, problem: Dict[str, Any]) -> ReasoningTask:
        """
        Determine the reasoning task from request.
        
        Args:
            request: Module request
            problem: Extracted problem
            
        Returns:
            ReasoningTask enum value
        """
        # Check capabilities
        if ModuleCapability.LOGICAL_REASONING in request.required_capabilities:
            return ReasoningTask.LOGICAL_INFERENCE
        elif ModuleCapability.CAUSAL_INFERENCE in request.required_capabilities:
            return ReasoningTask.CAUSAL_ANALYSIS
        elif ModuleCapability.PLANNING in request.required_capabilities:
            return ReasoningTask.PLANNING
        
        # Check problem type
        query = problem.get("query", "").lower()
        
        if "cause" in query or "why" in query or "effect" in query:
            return ReasoningTask.CAUSAL_ANALYSIS
        elif "prove" in query or "deduce" in query or "infer" in query:
            return ReasoningTask.LOGICAL_INFERENCE
        elif "solve" in query or "find" in query or "calculate" in query:
            return ReasoningTask.PROBLEM_SOLVING
        elif "plan" in query or "steps" in query or "how to" in query:
            return ReasoningTask.PLANNING
        elif "decide" in query or "choose" in query or "best" in query:
            return ReasoningTask.DECISION_MAKING
        elif "check" in query or "verify" in query or "true" in query:
            return ReasoningTask.FACT_CHECKING
        elif "hypothesis" in problem or "test" in query:
            return ReasoningTask.HYPOTHESIS_TESTING
        elif problem.get("constraints"):
            return ReasoningTask.CONSTRAINT_SATISFACTION
        
        # Default to logical inference
        return ReasoningTask.LOGICAL_INFERENCE
    
    async def _process_reasoning_task(
        self,
        problem: Dict[str, Any],
        task: ReasoningTask,
        context: Dict[str, Any]
    ) -> ReasoningResult:
        """
        Process specific reasoning task.
        
        Args:
            problem: Problem to solve
            task: Reasoning task type
            context: Processing context
            
        Returns:
            ReasoningResult with conclusions
        """
        result = ReasoningResult(task=task, metadata={})
        
        if task == ReasoningTask.LOGICAL_INFERENCE:
            # Perform logical inference
            inference = await self.logic_engine.infer(
                facts=problem.get("facts", []) + self.knowledge_base["facts"],
                rules=problem.get("rules", []) + self.knowledge_base["rules"],
                query=problem.get("query"),
                max_depth=self.max_inference_depth
            )
            
            result.conclusion = inference["conclusion"]
            result.proof = inference.get("proof", [])
            result.facts_used = inference.get("facts_used", [])
            result.rules_applied = inference.get("rules_applied", [])
            result.confidence = inference.get("confidence", 0.8)
            result.explanation = self._generate_explanation(inference)
            
        elif task == ReasoningTask.CAUSAL_ANALYSIS:
            # Analyze causal relationships
            causal = await self.causal_reasoner.analyze(
                events=problem.get("facts", []),
                query=problem.get("query"),
                causal_models=self.knowledge_base.get("causal_models", [])
            )
            
            result.conclusion = causal["conclusion"]
            result.causal_chain = causal.get("causal_chain", [])
            result.confidence = causal.get("confidence", 0.75)
            result.explanation = causal.get("explanation")
            
        elif task == ReasoningTask.PROBLEM_SOLVING:
            # Solve problem
            solution = await self.problem_solver.solve(
                problem=problem.get("query"),
                constraints=problem.get("constraints", []),
                variables=context.get("variables", {})
            )
            
            result.solution = solution["solution"]
            result.conclusion = f"Solution found: {solution['solution']}"
            result.confidence = solution.get("confidence", 0.85)
            result.metadata["steps"] = solution.get("steps", [])
            result.explanation = solution.get("explanation")
            
        elif task == ReasoningTask.PLANNING:
            # Create plan
            plan = await self._create_plan(problem, context)
            
            result.solution = plan["steps"]
            result.conclusion = f"Plan created with {len(plan['steps'])} steps"
            result.confidence = plan.get("confidence", 0.8)
            result.metadata["plan"] = plan
            result.explanation = self._explain_plan(plan)
            
        elif task == ReasoningTask.DECISION_MAKING:
            # Make decision
            decision = await self._make_decision(problem, context)
            
            result.conclusion = decision["choice"]
            result.confidence = decision.get("confidence", 0.7)
            result.metadata["alternatives"] = decision.get("alternatives", [])
            result.metadata["criteria"] = decision.get("criteria", [])
            result.explanation = decision.get("reasoning")
            
        elif task == ReasoningTask.FACT_CHECKING:
            # Check facts
            check = await self._check_facts(problem)
            
            result.conclusion = "True" if check["verified"] else "False"
            result.confidence = check.get("confidence", 0.9)
            result.facts_used = check.get("supporting_facts", [])
            result.explanation = check.get("explanation")
            
        elif task == ReasoningTask.HYPOTHESIS_TESTING:
            # Test hypothesis
            test = await self._test_hypothesis(problem, context)
            
            result.conclusion = test["result"]
            result.confidence = test.get("confidence", 0.75)
            result.metadata["evidence"] = test.get("evidence", [])
            result.metadata["p_value"] = test.get("p_value")
            result.explanation = test.get("explanation")
            
        elif task == ReasoningTask.CONSTRAINT_SATISFACTION:
            # Satisfy constraints
            csp = await self._satisfy_constraints(problem)
            
            result.solution = csp["solution"]
            result.conclusion = "Constraints satisfied" if csp["satisfied"] else "No solution found"
            result.confidence = csp.get("confidence", 0.8)
            result.metadata["assignments"] = csp.get("assignments", {})
            
        elif task == ReasoningTask.ABSTRACT_REASONING:
            # Abstract reasoning
            abstract = await self._abstract_reasoning(problem, context)
            
            result.conclusion = abstract["conclusion"]
            result.confidence = abstract.get("confidence", 0.7)
            result.metadata["patterns"] = abstract.get("patterns", [])
            result.explanation = abstract.get("explanation")
        
        # Add knowledge graph query if needed
        if problem.get("query"):
            kg_result = await self.knowledge_graph.query(problem["query"])
            result.metadata["knowledge_graph"] = kg_result
        
        return result
    
    async def _create_plan(self, problem: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a plan for achieving a goal.
        
        Args:
            problem: Problem specification
            context: Planning context
            
        Returns:
            Plan dictionary
        """
        goal = problem.get("goal", problem.get("query", ""))
        constraints = problem.get("constraints", [])
        
        # Simple planning algorithm (would use more sophisticated methods in production)
        steps = []
        
        # Decompose goal into subgoals
        subgoals = await self._decompose_goal(goal)
        
        # Order subgoals based on dependencies
        ordered_subgoals = self._order_subgoals(subgoals, constraints)
        
        # Create steps for each subgoal
        for i, subgoal in enumerate(ordered_subgoals):
            steps.append({
                "step": i + 1,
                "action": subgoal["action"],
                "description": subgoal["description"],
                "preconditions": subgoal.get("preconditions", []),
                "effects": subgoal.get("effects", []),
                "estimated_time": subgoal.get("time", "unknown")
            })
        
        return {
            "goal": goal,
            "steps": steps,
            "total_steps": len(steps),
            "constraints_satisfied": True,
            "confidence": 0.8
        }
    
    async def _decompose_goal(self, goal: str) -> List[Dict[str, Any]]:
        """
        Decompose a goal into subgoals.
        
        Args:
            goal: Goal description
            
        Returns:
            List of subgoals
        """
        # Simple decomposition (would use AI planning techniques in production)
        subgoals = []
        
        # Analyze goal for key actions
        if "and" in goal.lower():
            parts = goal.split("and")
            for part in parts:
                subgoals.append({
                    "action": f"achieve_{len(subgoals)+1}",
                    "description": part.strip()
                })
        else:
            # Single goal
            subgoals.append({
                "action": "achieve_goal",
                "description": goal
            })
        
        return subgoals
    
    def _order_subgoals(
        self,
        subgoals: List[Dict[str, Any]],
        constraints: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Order subgoals based on dependencies.
        
        Args:
            subgoals: List of subgoals
            constraints: Constraints to consider
            
        Returns:
            Ordered list of subgoals
        """
        # Simple ordering (would use topological sort with dependencies in production)
        return subgoals
    
    async def _make_decision(self, problem: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a decision based on criteria.
        
        Args:
            problem: Decision problem
            context: Decision context
            
        Returns:
            Decision result
        """
        query = problem.get("query", "")
        alternatives = problem.get("alternatives", [])
        criteria = problem.get("criteria", ["efficiency", "cost", "quality"])
        
        # Simple multi-criteria decision making
        if not alternatives:
            # Extract alternatives from query
            alternatives = ["option_a", "option_b", "option_c"]
        
        # Score each alternative
        scores = {}
        for alt in alternatives:
            score = 0
            for criterion in criteria:
                # Simple scoring (would use proper MCDM methods in production)
                score += await self._score_alternative(alt, criterion)
            scores[alt] = score / len(criteria) if criteria else 0
        
        # Select best alternative
        best = max(scores, key=scores.get) if scores else alternatives[0]
        
        return {
            "choice": best,
            "alternatives": alternatives,
            "criteria": criteria,
            "scores": scores,
            "reasoning": f"Selected {best} based on {', '.join(criteria)}",
            "confidence": max(scores.values()) if scores else 0.5
        }
    
    async def _score_alternative(self, alternative: str, criterion: str) -> float:
        """
        Score an alternative on a criterion.
        
        Args:
            alternative: Alternative to score
            criterion: Criterion to use
            
        Returns:
            Score between 0 and 1
        """
        # Placeholder scoring (would use domain knowledge in production)
        import random
        return random.random()
    
    async def _check_facts(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check facts against knowledge base.
        
        Args:
            problem: Facts to check
            
        Returns:
            Fact checking result
        """
        query = problem.get("query", "")
        facts_to_check = problem.get("facts", [query])
        
        verified = True
        supporting_facts = []
        contradicting_facts = []
        
        for fact in facts_to_check:
            # Check against knowledge base
            if await self._is_fact_supported(fact):
                supporting_facts.append(fact)
            else:
                contradicting_facts.append(fact)
                verified = False
        
        return {
            "verified": verified,
            "supporting_facts": supporting_facts,
            "contradicting_facts": contradicting_facts,
            "confidence": 0.9 if verified else 0.3,
            "explanation": f"Fact {'verified' if verified else 'not verified'} based on knowledge base"
        }
    
    async def _is_fact_supported(self, fact: str) -> bool:
        """
        Check if a fact is supported by knowledge base.
        
        Args:
            fact: Fact to check
            
        Returns:
            True if supported
        """
        # Check in knowledge base
        for kb_fact in self.knowledge_base["facts"]:
            if fact.lower() in kb_fact.lower() or kb_fact.lower() in fact.lower():
                return True
        
        # Check using inference
        result = await self.inference_engine.check_entailment(
            self.knowledge_base["facts"],
            fact
        )
        
        return result.get("entailed", False)
    
    async def _test_hypothesis(
        self,
        problem: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test a hypothesis.
        
        Args:
            problem: Hypothesis to test
            context: Testing context
            
        Returns:
            Test result
        """
        hypothesis = problem.get("hypothesis", problem.get("query", ""))
        evidence = problem.get("facts", [])
        
        # Simple hypothesis testing
        supporting = 0
        contradicting = 0
        
        for fact in evidence:
            if await self._supports_hypothesis(fact, hypothesis):
                supporting += 1
            else:
                contradicting += 1
        
        total = supporting + contradicting
        if total > 0:
            confidence = supporting / total
            result = "Supported" if confidence > 0.5 else "Not supported"
        else:
            confidence = 0.5
            result = "Insufficient evidence"
        
        return {
            "hypothesis": hypothesis,
            "result": result,
            "evidence": evidence,
            "supporting_evidence": supporting,
            "contradicting_evidence": contradicting,
            "confidence": confidence,
            "p_value": 1 - confidence,  # Simplified
            "explanation": f"Hypothesis {result.lower()} with {supporting} supporting and {contradicting} contradicting evidence"
        }
    
    async def _supports_hypothesis(self, fact: str, hypothesis: str) -> bool:
        """
        Check if a fact supports a hypothesis.
        
        Args:
            fact: Fact to check
            hypothesis: Hypothesis to test
            
        Returns:
            True if fact supports hypothesis
        """
        # Simple check (would use more sophisticated methods in production)
        # Check for semantic similarity or logical connection
        common_words = set(fact.lower().split()) & set(hypothesis.lower().split())
        return len(common_words) > 2
    
    async def _satisfy_constraints(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve constraint satisfaction problem.
        
        Args:
            problem: CSP specification
            
        Returns:
            Solution to CSP
        """
        constraints = problem.get("constraints", [])
        variables = problem.get("variables", {})
        
        # Simple constraint satisfaction (would use proper CSP solver in production)
        solution = {}
        satisfied = True
        
        # Try to satisfy each constraint
        for constraint in constraints:
            if not await self._check_constraint(constraint, solution, variables):
                satisfied = False
                break
        
        return {
            "satisfied": satisfied,
            "solution": solution if satisfied else None,
            "assignments": solution,
            "constraints_checked": len(constraints),
            "confidence": 0.8 if satisfied else 0.2
        }
    
    async def _check_constraint(
        self,
        constraint: str,
        solution: Dict[str, Any],
        variables: Dict[str, Any]
    ) -> bool:
        """
        Check if a constraint is satisfied.
        
        Args:
            constraint: Constraint to check
            solution: Current solution
            variables: Variable domains
            
        Returns:
            True if constraint is satisfied
        """
        # Placeholder constraint checking
        return True
    
    async def _abstract_reasoning(
        self,
        problem: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform abstract reasoning.
        
        Args:
            problem: Abstract problem
            context: Reasoning context
            
        Returns:
            Abstract reasoning result
        """
        query = problem.get("query", "")
        
        # Identify patterns
        patterns = await self._identify_patterns(query)
        
        # Apply abstract rules
        conclusion = await self._apply_abstract_rules(patterns)
        
        return {
            "conclusion": conclusion,
            "patterns": patterns,
            "confidence": 0.7,
            "explanation": f"Identified {len(patterns)} patterns and applied abstract reasoning"
        }
    
    async def _identify_patterns(self, text: str) -> List[str]:
        """
        Identify patterns in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of identified patterns
        """
        patterns = []
        
        # Simple pattern identification
        if "if" in text.lower() and "then" in text.lower():
            patterns.append("conditional")
        if "all" in text.lower() or "every" in text.lower():
            patterns.append("universal")
        if "some" in text.lower() or "exists" in text.lower():
            patterns.append("existential")
        
        return patterns
    
    async def _apply_abstract_rules(self, patterns: List[str]) -> str:
        """
        Apply abstract rules based on patterns.
        
        Args:
            patterns: Identified patterns
            
        Returns:
            Conclusion from abstract reasoning
        """
        if "conditional" in patterns:
            return "Conditional reasoning applied"
        elif "universal" in patterns:
            return "Universal quantification identified"
        elif "existential" in patterns:
            return "Existential quantification identified"
        else:
            return "Abstract pattern recognized"
    
    def _generate_explanation(self, inference: Dict[str, Any]) -> str:
        """
        Generate natural language explanation.
        
        Args:
            inference: Inference result
            
        Returns:
            Explanation string
        """
        explanation = []
        
        if inference.get("proof"):
            explanation.append(f"Proof: {' → '.join(inference['proof'])}")
        
        if inference.get("facts_used"):
            explanation.append(f"Facts used: {', '.join(inference['facts_used'][:3])}")
        
        if inference.get("rules_applied"):
            explanation.append(f"Rules applied: {', '.join(inference['rules_applied'][:3])}")
        
        return " | ".join(explanation) if explanation else "Direct inference"
    
    def _explain_plan(self, plan: Dict[str, Any]) -> str:
        """
        Explain a plan in natural language.
        
        Args:
            plan: Plan to explain
            
        Returns:
            Explanation string
        """
        steps = plan.get("steps", [])
        if not steps:
            return "No plan created"
        
        explanation = f"Plan to achieve '{plan.get('goal', 'objective')}' in {len(steps)} steps: "
        step_descriptions = [f"{s['step']}. {s['description']}" for s in steps[:3]]
        explanation += "; ".join(step_descriptions)
        
        if len(steps) > 3:
            explanation += f"... and {len(steps) - 3} more steps"
        
        return explanation
    
    async def _load_knowledge_base(self):
        """Load default knowledge base."""
        # Load some default facts and rules
        self.knowledge_base["facts"] = [
            "All humans are mortal",
            "Socrates is human",
            "Birds can fly",
            "Penguins are birds",
            "Penguins cannot fly",
            "Water freezes at 0 degrees Celsius",
            "The sun rises in the east"
        ]
        
        self.knowledge_base["rules"] = [
            "IF X is human THEN X is mortal",
            "IF X can fly AND X is a bird THEN X has wings",
            "IF temperature < 0 THEN water is frozen",
            "IF A causes B AND B causes C THEN A causes C"
        ]
        
        self.knowledge_base["causal_models"] = [
            {"cause": "rain", "effect": "wet ground"},
            {"cause": "fire", "effect": "smoke"},
            {"cause": "study", "effect": "knowledge"}
        ]
    
    def _format_result(self, result: ReasoningResult) -> Dict[str, Any]:
        """
        Format reasoning result for response.
        
        Args:
            result: ReasoningResult object
            
        Returns:
            Formatted dictionary
        """
        output = {
            "task": result.task.value,
            "conclusion": result.conclusion,
            "confidence": result.confidence,
            "processing_time_ms": result.processing_time_ms
        }
        
        if result.proof:
            output["proof"] = result.proof
        
        if result.facts_used:
            output["facts_used"] = result.facts_used
        
        if result.rules_applied:
            output["rules_applied"] = result.rules_applied
        
        if result.causal_chain:
            output["causal_chain"] = result.causal_chain
        
        if result.solution:
            output["solution"] = result.solution
        
        if result.explanation:
            output["explanation"] = result.explanation
        
        if result.metadata:
            output["metadata"] = result.metadata
        
        return output
    
    def get_capabilities(self) -> List[ModuleCapability]:
        """
        Get module capabilities.
        
        Returns:
            List of supported capabilities
        """
        return [
            ModuleCapability.REASONING,
            ModuleCapability.LOGICAL_REASONING,
            ModuleCapability.CAUSAL_INFERENCE,
            ModuleCapability.PLANNING
        ]
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info("Cleaning up Reasoning Module resources")
        
        # Cleanup engines
        if hasattr(self, 'logic_engine'):
            await self.logic_engine.cleanup()
        if hasattr(self, 'knowledge_graph'):
            await self.knowledge_graph.cleanup()
        
        self._initialized = False
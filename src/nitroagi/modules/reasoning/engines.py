"""
Reasoning Engines for NitroAGI NEXUS
Specialized engines for different types of reasoning
"""

import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import re
import networkx as nx
from nitroagi.utils.logging import get_logger


@dataclass
class LogicalStatement:
    """Represents a logical statement."""
    subject: str
    predicate: str
    object: str
    negated: bool = False
    
    def __str__(self):
        neg = "NOT " if self.negated else ""
        return f"{neg}{self.subject} {self.predicate} {self.object}"


@dataclass
class Rule:
    """Represents a logical rule."""
    conditions: List[LogicalStatement]
    conclusion: LogicalStatement
    confidence: float = 1.0
    
    def __str__(self):
        conds = " AND ".join(str(c) for c in self.conditions)
        return f"IF {conds} THEN {self.conclusion}"


class LogicEngine:
    """
    Logic engine for formal reasoning.
    Implements forward chaining, backward chaining, and resolution.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.facts = set()
        self.rules = []
        self.inferred_facts = set()
    
    async def initialize(self):
        """Initialize the logic engine."""
        self.logger.info("Initializing LogicEngine")
    
    async def infer(
        self,
        facts: List[str],
        rules: List[str],
        query: str = None,
        max_depth: int = 10
    ) -> Dict[str, Any]:
        """
        Perform logical inference.
        
        Args:
            facts: List of facts
            rules: List of rules
            query: Query to answer
            max_depth: Maximum inference depth
            
        Returns:
            Inference results
        """
        # Parse facts and rules
        self.facts = set(facts)
        self.rules = self._parse_rules(rules)
        self.inferred_facts = set()
        
        # Forward chaining
        proof = await self._forward_chain(max_depth)
        
        # Check if query is satisfied
        conclusion = "Unknown"
        confidence = 0.5
        
        if query:
            if query in self.facts or query in self.inferred_facts:
                conclusion = f"{query} is TRUE"
                confidence = 0.95
            elif f"NOT {query}" in self.facts or f"NOT {query}" in self.inferred_facts:
                conclusion = f"{query} is FALSE"
                confidence = 0.95
            else:
                # Try backward chaining
                if await self._backward_chain(query, max_depth):
                    conclusion = f"{query} is TRUE"
                    confidence = 0.9
                else:
                    conclusion = f"{query} cannot be proven"
                    confidence = 0.3
        else:
            conclusion = f"Inferred {len(self.inferred_facts)} new facts"
            confidence = 0.85
        
        return {
            "conclusion": conclusion,
            "proof": proof[:10],  # Limit proof steps
            "facts_used": list(self.facts)[:10],
            "inferred_facts": list(self.inferred_facts)[:10],
            "rules_applied": [str(r) for r in self.rules[:5]],
            "confidence": confidence
        }
    
    def _parse_rules(self, rules: List[str]) -> List[Rule]:
        """
        Parse text rules into Rule objects.
        
        Args:
            rules: List of rule strings
            
        Returns:
            List of Rule objects
        """
        parsed_rules = []
        
        for rule_text in rules:
            # Simple parsing for IF-THEN rules
            if "IF" in rule_text and "THEN" in rule_text:
                parts = rule_text.split("THEN")
                if len(parts) == 2:
                    conditions_text = parts[0].replace("IF", "").strip()
                    conclusion_text = parts[1].strip()
                    
                    # Parse conditions (simplified)
                    conditions = []
                    for cond in conditions_text.split("AND"):
                        cond = cond.strip()
                        # Simple pattern matching
                        words = cond.split()
                        if len(words) >= 3:
                            conditions.append(LogicalStatement(
                                subject=words[0],
                                predicate=words[1],
                                object=" ".join(words[2:])
                            ))
                    
                    # Parse conclusion
                    words = conclusion_text.split()
                    if len(words) >= 3:
                        conclusion = LogicalStatement(
                            subject=words[0],
                            predicate=words[1],
                            object=" ".join(words[2:])
                        )
                        
                        parsed_rules.append(Rule(
                            conditions=conditions,
                            conclusion=conclusion
                        ))
        
        return parsed_rules
    
    async def _forward_chain(self, max_depth: int) -> List[str]:
        """
        Perform forward chaining inference.
        
        Args:
            max_depth: Maximum inference depth
            
        Returns:
            Proof steps
        """
        proof = []
        depth = 0
        
        while depth < max_depth:
            new_facts = set()
            
            for rule in self.rules:
                # Check if all conditions are satisfied
                if self._check_conditions(rule.conditions):
                    conclusion_str = str(rule.conclusion)
                    
                    if conclusion_str not in self.facts and conclusion_str not in self.inferred_facts:
                        new_facts.add(conclusion_str)
                        proof.append(f"Applied {rule} → {conclusion_str}")
            
            if not new_facts:
                break
            
            self.inferred_facts.update(new_facts)
            depth += 1
        
        return proof
    
    async def _backward_chain(self, goal: str, max_depth: int) -> bool:
        """
        Perform backward chaining inference.
        
        Args:
            goal: Goal to prove
            max_depth: Maximum search depth
            
        Returns:
            True if goal can be proven
        """
        if goal in self.facts or goal in self.inferred_facts:
            return True
        
        if max_depth <= 0:
            return False
        
        # Find rules that conclude the goal
        for rule in self.rules:
            if str(rule.conclusion) == goal:
                # Try to prove all conditions
                all_proven = True
                for condition in rule.conditions:
                    if not await self._backward_chain(str(condition), max_depth - 1):
                        all_proven = False
                        break
                
                if all_proven:
                    self.inferred_facts.add(goal)
                    return True
        
        return False
    
    def _check_conditions(self, conditions: List[LogicalStatement]) -> bool:
        """
        Check if all conditions are satisfied.
        
        Args:
            conditions: List of conditions to check
            
        Returns:
            True if all conditions are satisfied
        """
        for condition in conditions:
            condition_str = str(condition)
            if condition_str not in self.facts and condition_str not in self.inferred_facts:
                return False
        return True
    
    async def cleanup(self):
        """Clean up resources."""
        self.facts.clear()
        self.rules.clear()
        self.inferred_facts.clear()


class CausalReasoner:
    """
    Causal reasoning engine.
    Analyzes cause-effect relationships and causal chains.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.causal_graph = nx.DiGraph()
    
    async def initialize(self):
        """Initialize the causal reasoner."""
        self.logger.info("Initializing CausalReasoner")
    
    async def analyze(
        self,
        events: List[str],
        query: str = None,
        causal_models: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze causal relationships.
        
        Args:
            events: List of events
            query: Causal query
            causal_models: Known causal relationships
            
        Returns:
            Causal analysis results
        """
        # Build causal graph
        self._build_causal_graph(events, causal_models or [])
        
        # Analyze query
        conclusion = "No causal relationship found"
        causal_chain = []
        confidence = 0.5
        
        if query:
            # Extract cause and effect from query
            cause, effect = self._extract_cause_effect(query)
            
            if cause and effect:
                # Find causal path
                path = self._find_causal_path(cause, effect)
                
                if path:
                    causal_chain = [{"cause": path[i], "effect": path[i+1]} 
                                  for i in range(len(path)-1)]
                    conclusion = f"{cause} causes {effect} through {len(causal_chain)} steps"
                    confidence = 0.85
                else:
                    conclusion = f"No causal path found from {cause} to {effect}"
                    confidence = 0.2
        
        # Find all causal relationships
        all_causes = list(self.causal_graph.edges())
        
        return {
            "conclusion": conclusion,
            "causal_chain": causal_chain,
            "all_relationships": all_causes[:10],
            "confidence": confidence,
            "explanation": self._explain_causality(causal_chain)
        }
    
    def _build_causal_graph(self, events: List[str], causal_models: List[Dict[str, str]]):
        """
        Build causal graph from events and models.
        
        Args:
            events: List of events
            causal_models: Known causal relationships
        """
        self.causal_graph.clear()
        
        # Add known causal relationships
        for model in causal_models:
            if "cause" in model and "effect" in model:
                self.causal_graph.add_edge(model["cause"], model["effect"])
        
        # Infer causal relationships from events
        for i, event1 in enumerate(events):
            for event2 in events[i+1:]:
                if self._implies_causality(event1, event2):
                    self.causal_graph.add_edge(event1, event2)
    
    def _extract_cause_effect(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract cause and effect from query.
        
        Args:
            query: Causal query
            
        Returns:
            Tuple of (cause, effect)
        """
        # Simple pattern matching
        patterns = [
            r"does (.*) cause (.*)",
            r"why does (.*) lead to (.*)",
            r"what causes (.*)",
            r"(.*) causes (.*)",
            r"effect of (.*) on (.*)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                if len(match.groups()) == 2:
                    return match.group(1).strip(), match.group(2).strip()
                elif len(match.groups()) == 1:
                    return None, match.group(1).strip()
        
        return None, None
    
    def _find_causal_path(self, cause: str, effect: str) -> Optional[List[str]]:
        """
        Find causal path from cause to effect.
        
        Args:
            cause: Starting event
            effect: Target event
            
        Returns:
            Path if exists, None otherwise
        """
        try:
            path = nx.shortest_path(self.causal_graph, cause, effect)
            return path
        except nx.NetworkXNoPath:
            return None
    
    def _implies_causality(self, event1: str, event2: str) -> bool:
        """
        Check if event1 implies causality with event2.
        
        Args:
            event1: First event
            event2: Second event
            
        Returns:
            True if causal relationship is implied
        """
        # Simple heuristics
        causal_keywords = ["causes", "leads to", "results in", "produces"]
        
        for keyword in causal_keywords:
            if keyword in event1.lower() or keyword in event2.lower():
                return True
        
        # Temporal reasoning (simplified)
        if "before" in event2.lower() or "after" in event1.lower():
            return True
        
        return False
    
    def _explain_causality(self, causal_chain: List[Dict[str, str]]) -> str:
        """
        Explain causal chain in natural language.
        
        Args:
            causal_chain: Chain of causal relationships
            
        Returns:
            Explanation string
        """
        if not causal_chain:
            return "No causal relationship identified"
        
        if len(causal_chain) == 1:
            return f"{causal_chain[0]['cause']} directly causes {causal_chain[0]['effect']}"
        
        explanation = "Causal chain: "
        for i, link in enumerate(causal_chain):
            if i == 0:
                explanation += f"{link['cause']} → {link['effect']}"
            else:
                explanation += f" → {link['effect']}"
        
        return explanation


class KnowledgeGraph:
    """
    Knowledge graph for semantic reasoning.
    Stores and queries structured knowledge.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.graph = nx.MultiDiGraph()
        self.entity_types = {}
        self.relation_types = set()
    
    async def initialize(self):
        """Initialize the knowledge graph."""
        self.logger.info("Initializing KnowledgeGraph")
        await self._load_default_knowledge()
    
    async def _load_default_knowledge(self):
        """Load default knowledge into the graph."""
        # Add some default entities and relationships
        self.add_entity("Human", "class")
        self.add_entity("Mortal", "property")
        self.add_entity("Socrates", "instance", entity_type="Human")
        
        self.add_relationship("Human", "is", "Mortal")
        self.add_relationship("Socrates", "is_a", "Human")
    
    def add_entity(self, entity: str, category: str, **attributes):
        """
        Add entity to knowledge graph.
        
        Args:
            entity: Entity name
            category: Entity category
            **attributes: Additional attributes
        """
        self.graph.add_node(entity, category=category, **attributes)
        self.entity_types[entity] = category
    
    def add_relationship(self, subject: str, predicate: str, object: str, **attributes):
        """
        Add relationship to knowledge graph.
        
        Args:
            subject: Subject entity
            predicate: Relationship type
            object: Object entity
            **attributes: Additional attributes
        """
        self.graph.add_edge(subject, object, predicate=predicate, **attributes)
        self.relation_types.add(predicate)
    
    async def query(self, query: str) -> Dict[str, Any]:
        """
        Query the knowledge graph.
        
        Args:
            query: Query string
            
        Returns:
            Query results
        """
        results = []
        
        # Simple query parsing
        if "?" in query:
            # SPARQL-like query
            results = await self._sparql_query(query)
        else:
            # Natural language query
            results = await self._nl_query(query)
        
        return {
            "query": query,
            "results": results[:10],
            "num_results": len(results),
            "graph_size": self.graph.number_of_nodes()
        }
    
    async def _sparql_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Process SPARQL-like query.
        
        Args:
            query: SPARQL-like query
            
        Returns:
            Query results
        """
        results = []
        
        # Simple pattern matching for ?subject predicate ?object
        pattern = r"\?(\w+)\s+(\w+)\s+\?(\w+)"
        match = re.search(pattern, query)
        
        if match:
            var1, predicate, var2 = match.groups()
            
            # Find all edges with this predicate
            for u, v, data in self.graph.edges(data=True):
                if data.get("predicate") == predicate:
                    results.append({var1: u, var2: v, "predicate": predicate})
        
        return results
    
    async def _nl_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Process natural language query.
        
        Args:
            query: Natural language query
            
        Returns:
            Query results
        """
        results = []
        query_lower = query.lower()
        
        # Find entities mentioned in query
        for node in self.graph.nodes():
            if node.lower() in query_lower:
                # Get all relationships for this entity
                edges_out = list(self.graph.out_edges(node, data=True))
                edges_in = list(self.graph.in_edges(node, data=True))
                
                for u, v, data in edges_out:
                    results.append({
                        "subject": u,
                        "predicate": data.get("predicate", "related_to"),
                        "object": v
                    })
                
                for u, v, data in edges_in:
                    results.append({
                        "subject": u,
                        "predicate": data.get("predicate", "related_to"),
                        "object": v
                    })
        
        return results
    
    async def cleanup(self):
        """Clean up resources."""
        self.graph.clear()
        self.entity_types.clear()
        self.relation_types.clear()


class ProblemSolver:
    """
    General problem solver using search and optimization.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    async def initialize(self):
        """Initialize the problem solver."""
        self.logger.info("Initializing ProblemSolver")
    
    async def solve(
        self,
        problem: str,
        constraints: List[str] = None,
        variables: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Solve a problem.
        
        Args:
            problem: Problem description
            constraints: List of constraints
            variables: Problem variables
            
        Returns:
            Solution
        """
        # Identify problem type
        problem_type = self._identify_problem_type(problem)
        
        solution = None
        steps = []
        confidence = 0.5
        
        if problem_type == "optimization":
            solution, steps = await self._solve_optimization(problem, constraints, variables)
            confidence = 0.8
        elif problem_type == "search":
            solution, steps = await self._solve_search(problem, constraints, variables)
            confidence = 0.75
        elif problem_type == "mathematical":
            solution, steps = await self._solve_mathematical(problem, variables)
            confidence = 0.9
        else:
            solution = "Problem type not recognized"
            confidence = 0.3
        
        return {
            "problem": problem,
            "solution": solution,
            "steps": steps,
            "problem_type": problem_type,
            "confidence": confidence,
            "explanation": self._explain_solution(solution, steps)
        }
    
    def _identify_problem_type(self, problem: str) -> str:
        """
        Identify the type of problem.
        
        Args:
            problem: Problem description
            
        Returns:
            Problem type
        """
        problem_lower = problem.lower()
        
        if "maximize" in problem_lower or "minimize" in problem_lower:
            return "optimization"
        elif "find" in problem_lower or "search" in problem_lower:
            return "search"
        elif any(op in problem_lower for op in ["+", "-", "*", "/", "="]):
            return "mathematical"
        else:
            return "general"
    
    async def _solve_optimization(
        self,
        problem: str,
        constraints: List[str],
        variables: Dict[str, Any]
    ) -> Tuple[Any, List[str]]:
        """
        Solve optimization problem.
        
        Args:
            problem: Problem description
            constraints: Constraints
            variables: Variables
            
        Returns:
            Tuple of (solution, steps)
        """
        steps = []
        steps.append("1. Identify objective function")
        steps.append("2. Apply constraints")
        steps.append("3. Find optimal solution")
        
        # Placeholder solution
        solution = {"optimal_value": 42, "variables": variables or {}}
        
        return solution, steps
    
    async def _solve_search(
        self,
        problem: str,
        constraints: List[str],
        variables: Dict[str, Any]
    ) -> Tuple[Any, List[str]]:
        """
        Solve search problem.
        
        Args:
            problem: Problem description
            constraints: Constraints
            variables: Variables
            
        Returns:
            Tuple of (solution, steps)
        """
        steps = []
        steps.append("1. Define search space")
        steps.append("2. Apply search algorithm")
        steps.append("3. Verify solution")
        
        # Placeholder solution
        solution = {"found": True, "result": "Solution found"}
        
        return solution, steps
    
    async def _solve_mathematical(
        self,
        problem: str,
        variables: Dict[str, Any]
    ) -> Tuple[Any, List[str]]:
        """
        Solve mathematical problem.
        
        Args:
            problem: Problem description
            variables: Variables
            
        Returns:
            Tuple of (solution, steps)
        """
        steps = []
        
        # Try to evaluate simple expressions
        try:
            # Remove non-mathematical words
            expr = problem
            for word in ["solve", "calculate", "find", "what", "is", "the", "value", "of"]:
                expr = expr.lower().replace(word, "")
            
            expr = expr.strip()
            
            # Simple evaluation (unsafe in production!)
            if expr and all(c in "0123456789+-*/()= " for c in expr):
                if "=" in expr:
                    parts = expr.split("=")
                    if len(parts) == 2 and "?" not in parts[0]:
                        result = eval(parts[0])
                        solution = f"{parts[0].strip()} = {result}"
                else:
                    result = eval(expr)
                    solution = f"{expr} = {result}"
                
                steps.append(f"Evaluated: {expr}")
                steps.append(f"Result: {result}")
            else:
                solution = "Cannot evaluate expression"
        except:
            solution = "Mathematical evaluation failed"
        
        return solution, steps
    
    def _explain_solution(self, solution: Any, steps: List[str]) -> str:
        """
        Explain the solution.
        
        Args:
            solution: Solution found
            steps: Solution steps
            
        Returns:
            Explanation string
        """
        if not solution:
            return "No solution found"
        
        explanation = f"Solution: {solution}"
        if steps:
            explanation += f" achieved in {len(steps)} steps"
        
        return explanation


class InferenceEngine:
    """
    General inference engine for various reasoning tasks.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    async def initialize(self):
        """Initialize the inference engine."""
        self.logger.info("Initializing InferenceEngine")
    
    async def check_entailment(self, premises: List[str], conclusion: str) -> Dict[str, Any]:
        """
        Check if conclusion is entailed by premises.
        
        Args:
            premises: List of premises
            conclusion: Conclusion to check
            
        Returns:
            Entailment result
        """
        # Simple entailment checking
        entailed = False
        confidence = 0.5
        
        # Check direct entailment
        if conclusion in premises:
            entailed = True
            confidence = 1.0
        else:
            # Check semantic similarity
            for premise in premises:
                similarity = self._semantic_similarity(premise, conclusion)
                if similarity > 0.8:
                    entailed = True
                    confidence = similarity
                    break
        
        return {
            "entailed": entailed,
            "confidence": confidence,
            "premises_used": premises[:5]
        }
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    async def cleanup(self):
        """Clean up resources."""
        pass
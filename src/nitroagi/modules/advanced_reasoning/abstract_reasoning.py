"""
Abstract Reasoning Module for NitroAGI NEXUS
Handles pattern recognition, analogies, and abstract concept manipulation
"""

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import json
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    """Types of abstract reasoning"""
    PATTERN_RECOGNITION = "pattern_recognition"
    ANALOGY = "analogy"
    ABSTRACTION = "abstraction"
    INDUCTION = "induction"
    DEDUCTION = "deduction"
    ABDUCTION = "abduction"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    CONCEPTUAL = "conceptual"


@dataclass
class Pattern:
    """Represents an abstract pattern"""
    pattern_type: str
    elements: List[Any]
    rules: List[str]
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class Analogy:
    """Represents an analogy relationship"""
    source_domain: str
    target_domain: str
    mappings: Dict[str, str]
    similarity_score: float
    structural_alignment: Dict[str, Any]


@dataclass
class Concept:
    """Abstract concept representation"""
    name: str
    properties: Set[str]
    relations: Dict[str, List[str]]
    instances: List[Any]
    abstraction_level: int


class PatternRecognizer:
    """Identifies patterns in data and concepts"""
    
    def __init__(self):
        self.pattern_library = {}
        self.learned_patterns = []
        
    async def recognize_pattern(
        self, 
        data: List[Any],
        pattern_type: Optional[str] = None
    ) -> Optional[Pattern]:
        """Recognize patterns in data"""
        
        if pattern_type == "sequence":
            return await self._recognize_sequence_pattern(data)
        elif pattern_type == "spatial":
            return await self._recognize_spatial_pattern(data)
        elif pattern_type == "relational":
            return await self._recognize_relational_pattern(data)
        else:
            # Try all pattern types
            patterns = []
            for ptype in ["sequence", "spatial", "relational"]:
                pattern = await self.recognize_pattern(data, ptype)
                if pattern:
                    patterns.append(pattern)
            
            # Return highest confidence pattern
            if patterns:
                return max(patterns, key=lambda p: p.confidence)
            return None
    
    async def _recognize_sequence_pattern(self, data: List[Any]) -> Optional[Pattern]:
        """Recognize sequential patterns"""
        
        if len(data) < 3:
            return None
        
        # Check for arithmetic progression
        if all(isinstance(x, (int, float)) for x in data):
            differences = [data[i+1] - data[i] for i in range(len(data)-1)]
            if len(set(differences)) == 1:
                return Pattern(
                    pattern_type="arithmetic_sequence",
                    elements=data,
                    rules=[f"n+{differences[0]}"],
                    confidence=1.0,
                    metadata={"difference": differences[0]}
                )
            
            # Check for geometric progression
            if all(x != 0 for x in data):
                ratios = [data[i+1] / data[i] for i in range(len(data)-1)]
                if all(abs(r - ratios[0]) < 0.001 for r in ratios):
                    return Pattern(
                        pattern_type="geometric_sequence",
                        elements=data,
                        rules=[f"n*{ratios[0]}"],
                        confidence=1.0,
                        metadata={"ratio": ratios[0]}
                    )
        
        # Check for repeating pattern
        for pattern_len in range(1, len(data) // 2 + 1):
            pattern = data[:pattern_len]
            repeats = len(data) // pattern_len
            
            if data[:pattern_len * repeats] == pattern * repeats:
                return Pattern(
                    pattern_type="repeating",
                    elements=data,
                    rules=[f"repeat({pattern})"],
                    confidence=0.9,
                    metadata={"period": pattern_len, "pattern": pattern}
                )
        
        return None
    
    async def _recognize_spatial_pattern(self, data: List[Any]) -> Optional[Pattern]:
        """Recognize spatial patterns"""
        
        # Simplified spatial pattern recognition
        # In production, would use more sophisticated computer vision
        
        if not data:
            return None
        
        # Example: symmetry detection
        if len(data) > 1 and data == data[::-1]:
            return Pattern(
                pattern_type="symmetric",
                elements=data,
                rules=["mirror_symmetry"],
                confidence=0.95,
                metadata={"axis": "center"}
            )
        
        return None
    
    async def _recognize_relational_pattern(self, data: List[Any]) -> Optional[Pattern]:
        """Recognize relational patterns between elements"""
        
        if len(data) < 2:
            return None
        
        # Extract relationships
        relationships = []
        for i in range(len(data) - 1):
            if isinstance(data[i], dict) and isinstance(data[i+1], dict):
                common_keys = set(data[i].keys()) & set(data[i+1].keys())
                if common_keys:
                    relationships.append({
                        "type": "shared_properties",
                        "properties": list(common_keys)
                    })
        
        if relationships:
            return Pattern(
                pattern_type="relational",
                elements=data,
                rules=[str(r) for r in relationships],
                confidence=0.7,
                metadata={"relationships": relationships}
            )
        
        return None
    
    def extend_pattern(self, pattern: Pattern, n: int = 1) -> List[Any]:
        """Extend a pattern by n elements"""
        
        if pattern.pattern_type == "arithmetic_sequence":
            diff = pattern.metadata["difference"]
            last = pattern.elements[-1]
            return [last + diff * (i+1) for i in range(n)]
        
        elif pattern.pattern_type == "geometric_sequence":
            ratio = pattern.metadata["ratio"]
            last = pattern.elements[-1]
            return [last * (ratio ** (i+1)) for i in range(n)]
        
        elif pattern.pattern_type == "repeating":
            pattern_seq = pattern.metadata["pattern"]
            extended = []
            for i in range(n):
                extended.append(pattern_seq[i % len(pattern_seq)])
            return extended
        
        return []


class AnalogyEngine:
    """Handles analogical reasoning"""
    
    def __init__(self):
        self.analogy_database = {}
        self.domain_mappings = defaultdict(list)
    
    async def find_analogy(
        self,
        source: Dict[str, Any],
        target_domain: str
    ) -> Optional[Analogy]:
        """Find analogies between source and target domain"""
        
        # Extract source structure
        source_structure = self._extract_structure(source)
        
        # Search for similar structures in target domain
        if target_domain in self.domain_mappings:
            candidates = self.domain_mappings[target_domain]
            
            best_match = None
            best_score = 0.0
            
            for candidate in candidates:
                score = self._compute_similarity(source_structure, candidate)
                if score > best_score:
                    best_score = score
                    best_match = candidate
            
            if best_match and best_score > 0.5:
                mappings = self._create_mappings(source_structure, best_match)
                return Analogy(
                    source_domain=source.get("domain", "unknown"),
                    target_domain=target_domain,
                    mappings=mappings,
                    similarity_score=best_score,
                    structural_alignment={"source": source_structure, "target": best_match}
                )
        
        return None
    
    def _extract_structure(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structural relationships from object"""
        
        structure = {
            "entities": [],
            "relations": [],
            "properties": {}
        }
        
        # Extract entities
        for key, value in obj.items():
            if isinstance(value, dict):
                structure["entities"].append(key)
                structure["properties"][key] = list(value.keys())
            elif isinstance(value, list):
                structure["relations"].append({
                    "type": "collection",
                    "name": key,
                    "size": len(value)
                })
        
        return structure
    
    def _compute_similarity(
        self, 
        struct1: Dict[str, Any], 
        struct2: Dict[str, Any]
    ) -> float:
        """Compute structural similarity between two structures"""
        
        # Simple similarity based on common elements
        entities1 = set(struct1.get("entities", []))
        entities2 = set(struct2.get("entities", []))
        
        if not entities1 or not entities2:
            return 0.0
        
        common = len(entities1 & entities2)
        total = len(entities1 | entities2)
        
        entity_sim = common / total if total > 0 else 0
        
        # Relation similarity
        rel1_types = {r["type"] for r in struct1.get("relations", [])}
        rel2_types = {r["type"] for r in struct2.get("relations", [])}
        
        rel_sim = len(rel1_types & rel2_types) / max(len(rel1_types | rel2_types), 1)
        
        return 0.7 * entity_sim + 0.3 * rel_sim
    
    def _create_mappings(
        self,
        source: Dict[str, Any],
        target: Dict[str, Any]
    ) -> Dict[str, str]:
        """Create element mappings between structures"""
        
        mappings = {}
        
        # Map entities
        source_entities = source.get("entities", [])
        target_entities = target.get("entities", [])
        
        for i, s_entity in enumerate(source_entities):
            if i < len(target_entities):
                mappings[s_entity] = target_entities[i]
        
        return mappings
    
    async def apply_analogy(
        self,
        analogy: Analogy,
        problem: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply analogy to solve a problem"""
        
        solution = {}
        
        # Map problem elements using analogy
        for key, value in problem.items():
            if key in analogy.mappings:
                solution[analogy.mappings[key]] = value
            else:
                solution[key] = value
        
        # Apply structural transformations
        solution["applied_analogy"] = {
            "source": analogy.source_domain,
            "target": analogy.target_domain,
            "confidence": analogy.similarity_score
        }
        
        return solution


class ConceptualReasoner:
    """Handles abstract concept manipulation"""
    
    def __init__(self):
        self.concept_hierarchy = {}
        self.concept_relations = defaultdict(list)
    
    async def abstract_concept(
        self,
        instances: List[Dict[str, Any]]
    ) -> Concept:
        """Abstract a concept from instances"""
        
        if not instances:
            return None
        
        # Find common properties
        common_props = set(instances[0].keys())
        for instance in instances[1:]:
            common_props &= set(instance.keys())
        
        # Find common relations
        relations = defaultdict(list)
        for instance in instances:
            for prop in common_props:
                if isinstance(instance[prop], list):
                    relations[prop].extend(instance[prop])
        
        # Determine abstraction level
        abstraction_level = self._compute_abstraction_level(instances)
        
        concept = Concept(
            name=f"concept_{len(self.concept_hierarchy)}",
            properties=common_props,
            relations=dict(relations),
            instances=instances,
            abstraction_level=abstraction_level
        )
        
        # Store in hierarchy
        self.concept_hierarchy[concept.name] = concept
        
        return concept
    
    def _compute_abstraction_level(self, instances: List[Dict[str, Any]]) -> int:
        """Compute abstraction level of concept"""
        
        # Higher variance = higher abstraction
        if not instances:
            return 0
        
        # Count unique values per property
        prop_variance = {}
        for prop in instances[0].keys():
            unique_values = set()
            for instance in instances:
                if prop in instance:
                    unique_values.add(str(instance[prop]))
            prop_variance[prop] = len(unique_values)
        
        avg_variance = np.mean(list(prop_variance.values()))
        
        # Map to abstraction level (0-10)
        return min(10, int(avg_variance))
    
    async def specialize_concept(
        self,
        concept: Concept,
        constraints: Dict[str, Any]
    ) -> Concept:
        """Specialize a concept with additional constraints"""
        
        # Filter instances that meet constraints
        filtered_instances = []
        for instance in concept.instances:
            meets_constraints = True
            for key, value in constraints.items():
                if key not in instance or instance[key] != value:
                    meets_constraints = False
                    break
            if meets_constraints:
                filtered_instances.append(instance)
        
        # Create specialized concept
        specialized = Concept(
            name=f"{concept.name}_specialized",
            properties=concept.properties | set(constraints.keys()),
            relations=concept.relations,
            instances=filtered_instances,
            abstraction_level=max(0, concept.abstraction_level - 1)
        )
        
        # Add to hierarchy as child
        self.concept_relations[concept.name].append(specialized.name)
        self.concept_hierarchy[specialized.name] = specialized
        
        return specialized
    
    async def generalize_concepts(
        self,
        concepts: List[Concept]
    ) -> Concept:
        """Generalize multiple concepts into higher abstraction"""
        
        if not concepts:
            return None
        
        # Find common properties across concepts
        common_props = set(concepts[0].properties)
        for concept in concepts[1:]:
            common_props &= concept.properties
        
        # Combine all instances
        all_instances = []
        for concept in concepts:
            all_instances.extend(concept.instances)
        
        # Create generalized concept
        generalized = Concept(
            name=f"generalized_{len(self.concept_hierarchy)}",
            properties=common_props,
            relations={},
            instances=all_instances,
            abstraction_level=max(c.abstraction_level for c in concepts) + 1
        )
        
        # Update hierarchy
        self.concept_hierarchy[generalized.name] = generalized
        for concept in concepts:
            self.concept_relations[generalized.name].append(concept.name)
        
        return generalized


class InductiveReasoner:
    """Performs inductive reasoning"""
    
    def __init__(self):
        self.hypotheses = []
        self.evidence = []
    
    async def induce_rule(
        self,
        observations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Induce general rule from observations"""
        
        if len(observations) < 2:
            return None
        
        # Find patterns in observations
        patterns = []
        
        # Check for common structure
        common_keys = set(observations[0].keys())
        for obs in observations[1:]:
            common_keys &= set(obs.keys())
        
        # Analyze values for patterns
        for key in common_keys:
            values = [obs[key] for obs in observations]
            
            # Check if all values follow a pattern
            if all(isinstance(v, (int, float)) for v in values):
                # Numerical pattern
                if len(set(values)) == 1:
                    patterns.append({
                        "type": "constant",
                        "property": key,
                        "value": values[0]
                    })
                elif self._is_increasing(values):
                    patterns.append({
                        "type": "increasing",
                        "property": key
                    })
                elif self._is_decreasing(values):
                    patterns.append({
                        "type": "decreasing",
                        "property": key
                    })
            elif all(isinstance(v, str) for v in values):
                # String pattern
                if len(set(values)) == 1:
                    patterns.append({
                        "type": "constant_string",
                        "property": key,
                        "value": values[0]
                    })
        
        # Generate hypothesis
        hypothesis = {
            "rule": self._generate_rule(patterns),
            "confidence": self._calculate_confidence(patterns, observations),
            "patterns": patterns,
            "supporting_evidence": len(observations)
        }
        
        self.hypotheses.append(hypothesis)
        self.evidence.extend(observations)
        
        return hypothesis
    
    def _is_increasing(self, values: List[float]) -> bool:
        """Check if values are increasing"""
        return all(values[i] <= values[i+1] for i in range(len(values)-1))
    
    def _is_decreasing(self, values: List[float]) -> bool:
        """Check if values are decreasing"""
        return all(values[i] >= values[i+1] for i in range(len(values)-1))
    
    def _generate_rule(self, patterns: List[Dict[str, Any]]) -> str:
        """Generate rule description from patterns"""
        
        if not patterns:
            return "No clear pattern detected"
        
        rules = []
        for pattern in patterns:
            if pattern["type"] == "constant":
                rules.append(f"{pattern['property']} = {pattern['value']}")
            elif pattern["type"] == "increasing":
                rules.append(f"{pattern['property']} increases")
            elif pattern["type"] == "decreasing":
                rules.append(f"{pattern['property']} decreases")
            elif pattern["type"] == "constant_string":
                rules.append(f"{pattern['property']} = '{pattern['value']}'")
        
        return " AND ".join(rules)
    
    def _calculate_confidence(
        self,
        patterns: List[Dict[str, Any]],
        observations: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence in induced rule"""
        
        if not patterns or len(observations) < 2:
            return 0.0
        
        # Base confidence on number of observations
        obs_confidence = min(1.0, len(observations) / 10.0)
        
        # Pattern consistency
        pattern_confidence = len(patterns) / max(len(observations[0].keys()), 1)
        
        return 0.6 * obs_confidence + 0.4 * pattern_confidence


class AbstractReasoner:
    """Main abstract reasoning coordinator"""
    
    def __init__(self):
        self.pattern_recognizer = PatternRecognizer()
        self.analogy_engine = AnalogyEngine()
        self.conceptual_reasoner = ConceptualReasoner()
        self.inductive_reasoner = InductiveReasoner()
        
        self.reasoning_history = []
        
    async def reason(
        self,
        input_data: Dict[str, Any],
        reasoning_type: ReasoningType = None
    ) -> Dict[str, Any]:
        """Perform abstract reasoning on input"""
        
        logger.info(f"Abstract reasoning: {reasoning_type}")
        
        result = {
            "input": input_data,
            "reasoning_type": reasoning_type.value if reasoning_type else "auto",
            "results": []
        }
        
        if reasoning_type == ReasoningType.PATTERN_RECOGNITION:
            pattern = await self.pattern_recognizer.recognize_pattern(
                input_data.get("data", [])
            )
            if pattern:
                result["results"].append({
                    "type": "pattern",
                    "pattern": pattern.__dict__,
                    "next_elements": self.pattern_recognizer.extend_pattern(pattern, 3)
                })
        
        elif reasoning_type == ReasoningType.ANALOGY:
            analogy = await self.analogy_engine.find_analogy(
                input_data.get("source", {}),
                input_data.get("target_domain", "")
            )
            if analogy:
                result["results"].append({
                    "type": "analogy",
                    "analogy": analogy.__dict__
                })
        
        elif reasoning_type == ReasoningType.ABSTRACTION:
            concept = await self.conceptual_reasoner.abstract_concept(
                input_data.get("instances", [])
            )
            if concept:
                result["results"].append({
                    "type": "concept",
                    "concept": {
                        "name": concept.name,
                        "properties": list(concept.properties),
                        "abstraction_level": concept.abstraction_level
                    }
                })
        
        elif reasoning_type == ReasoningType.INDUCTION:
            hypothesis = await self.inductive_reasoner.induce_rule(
                input_data.get("observations", [])
            )
            if hypothesis:
                result["results"].append({
                    "type": "hypothesis",
                    "hypothesis": hypothesis
                })
        
        else:
            # Auto-detect and apply multiple reasoning types
            tasks = []
            
            if "data" in input_data:
                tasks.append(self.pattern_recognizer.recognize_pattern(input_data["data"]))
            
            if "instances" in input_data:
                tasks.append(self.conceptual_reasoner.abstract_concept(input_data["instances"]))
            
            if "observations" in input_data:
                tasks.append(self.inductive_reasoner.induce_rule(input_data["observations"]))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for res in results:
                if res and not isinstance(res, Exception):
                    if isinstance(res, Pattern):
                        result["results"].append({
                            "type": "pattern",
                            "pattern": res.__dict__
                        })
                    elif isinstance(res, Concept):
                        result["results"].append({
                            "type": "concept",
                            "concept": {
                                "name": res.name,
                                "properties": list(res.properties)
                            }
                        })
                    elif isinstance(res, dict):
                        result["results"].append(res)
        
        # Store in history
        self.reasoning_history.append(result)
        
        return result
    
    async def solve_abstract_problem(
        self,
        problem: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Solve abstract problem using multiple reasoning strategies"""
        
        solution = {
            "problem": problem,
            "approaches": [],
            "solution": None,
            "confidence": 0.0
        }
        
        # Try pattern-based approach
        if "sequence" in problem:
            pattern = await self.pattern_recognizer.recognize_pattern(
                problem["sequence"]
            )
            if pattern:
                next_elements = self.pattern_recognizer.extend_pattern(pattern, 1)
                solution["approaches"].append({
                    "method": "pattern_recognition",
                    "result": next_elements[0] if next_elements else None
                })
        
        # Try analogy-based approach
        if "similar_problem" in problem:
            analogy = await self.analogy_engine.find_analogy(
                problem["similar_problem"],
                problem.get("domain", "general")
            )
            if analogy:
                analogical_solution = await self.analogy_engine.apply_analogy(
                    analogy,
                    problem
                )
                solution["approaches"].append({
                    "method": "analogy",
                    "result": analogical_solution
                })
        
        # Try inductive approach
        if "examples" in problem:
            hypothesis = await self.inductive_reasoner.induce_rule(
                problem["examples"]
            )
            if hypothesis:
                solution["approaches"].append({
                    "method": "induction",
                    "result": hypothesis["rule"]
                })
        
        # Select best solution
        if solution["approaches"]:
            # For now, use first successful approach
            solution["solution"] = solution["approaches"][0]["result"]
            solution["confidence"] = 0.8
        
        return solution
    
    def get_reasoning_capabilities(self) -> Dict[str, List[str]]:
        """Get available reasoning capabilities"""
        
        return {
            "pattern_types": ["sequence", "spatial", "relational"],
            "reasoning_methods": [r.value for r in ReasoningType],
            "abstraction_levels": list(range(11)),
            "concept_count": len(self.conceptual_reasoner.concept_hierarchy),
            "hypothesis_count": len(self.inductive_reasoner.hypotheses)
        }
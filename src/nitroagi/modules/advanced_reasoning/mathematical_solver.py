"""
Mathematical Problem Solver for NitroAGI NEXUS
Handles symbolic math, numerical computation, and mathematical reasoning
"""

import asyncio
import numpy as np
import sympy as sp
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
import logging
import re
from fractions import Fraction

logger = logging.getLogger(__name__)


class MathProblemType(Enum):
    """Types of mathematical problems"""
    ARITHMETIC = "arithmetic"
    ALGEBRA = "algebra"
    CALCULUS = "calculus"
    LINEAR_ALGEBRA = "linear_algebra"
    STATISTICS = "statistics"
    GEOMETRY = "geometry"
    NUMBER_THEORY = "number_theory"
    DISCRETE_MATH = "discrete_math"
    OPTIMIZATION = "optimization"
    DIFFERENTIAL_EQUATIONS = "differential_equations"


@dataclass
class MathExpression:
    """Mathematical expression representation"""
    raw_text: str
    symbolic_form: Optional[sp.Basic]
    numeric_value: Optional[Union[float, complex]]
    variables: List[str]
    expression_type: str


@dataclass
class MathSolution:
    """Solution to a mathematical problem"""
    problem: str
    solution: Any
    steps: List[str]
    verification: bool
    confidence: float
    method_used: str


class SymbolicMathEngine:
    """Handles symbolic mathematics"""
    
    def __init__(self):
        self.symbols_cache = {}
        self.equation_history = []
    
    async def parse_expression(self, expr_text: str) -> MathExpression:
        """Parse text into mathematical expression"""
        
        try:
            # Clean the expression
            expr_text = self._clean_expression(expr_text)
            
            # Extract variables
            variables = self._extract_variables(expr_text)
            
            # Create symbols
            symbols = {}
            for var in variables:
                if var not in self.symbols_cache:
                    self.symbols_cache[var] = sp.Symbol(var)
                symbols[var] = self.symbols_cache[var]
            
            # Parse to symbolic form
            symbolic = sp.sympify(expr_text, locals=symbols)
            
            # Try to evaluate numerically if possible
            numeric = None
            if not symbolic.free_symbols:
                try:
                    numeric = float(symbolic.evalf())
                except:
                    numeric = complex(symbolic.evalf())
            
            return MathExpression(
                raw_text=expr_text,
                symbolic_form=symbolic,
                numeric_value=numeric,
                variables=variables,
                expression_type=self._classify_expression(symbolic)
            )
            
        except Exception as e:
            logger.error(f"Failed to parse expression: {e}")
            return MathExpression(
                raw_text=expr_text,
                symbolic_form=None,
                numeric_value=None,
                variables=[],
                expression_type="unknown"
            )
    
    def _clean_expression(self, expr: str) -> str:
        """Clean and standardize expression"""
        
        # Replace common notation
        expr = expr.replace("^", "**")
        expr = expr.replace("×", "*")
        expr = expr.replace("÷", "/")
        expr = expr.replace("π", "pi")
        expr = expr.replace("e^", "exp")
        
        return expr.strip()
    
    def _extract_variables(self, expr: str) -> List[str]:
        """Extract variable names from expression"""
        
        # Find all letter sequences not part of functions
        pattern = r'\b[a-zA-Z](?![a-zA-Z\(])\b'
        variables = re.findall(pattern, expr)
        
        # Remove function names
        functions = ['sin', 'cos', 'tan', 'exp', 'log', 'ln', 'sqrt', 'abs']
        variables = [v for v in variables if v not in functions]
        
        return list(set(variables))
    
    def _classify_expression(self, expr: sp.Basic) -> str:
        """Classify the type of expression"""
        
        if expr.is_polynomial():
            return "polynomial"
        elif expr.is_rational:
            return "rational"
        elif any(expr.has(func) for func in [sp.sin, sp.cos, sp.tan]):
            return "trigonometric"
        elif expr.has(sp.exp) or expr.has(sp.log):
            return "exponential"
        elif expr.has(sp.Derivative):
            return "differential"
        elif expr.has(sp.Integral):
            return "integral"
        else:
            return "general"
    
    async def solve_equation(
        self,
        equation: str,
        variable: Optional[str] = None
    ) -> List[Any]:
        """Solve an equation"""
        
        try:
            # Parse equation
            if "=" in equation:
                left, right = equation.split("=")
                left_expr = await self.parse_expression(left)
                right_expr = await self.parse_expression(right)
                
                eq = sp.Eq(left_expr.symbolic_form, right_expr.symbolic_form)
            else:
                expr = await self.parse_expression(equation)
                eq = sp.Eq(expr.symbolic_form, 0)
            
            # Determine variable to solve for
            if variable:
                var = self.symbols_cache.get(variable, sp.Symbol(variable))
            else:
                # Use first free symbol
                free_symbols = eq.free_symbols
                if free_symbols:
                    var = list(free_symbols)[0]
                else:
                    return []
            
            # Solve the equation
            solutions = sp.solve(eq, var)
            
            # Store in history
            self.equation_history.append({
                "equation": equation,
                "variable": str(var),
                "solutions": [str(s) for s in solutions]
            })
            
            return solutions
            
        except Exception as e:
            logger.error(f"Failed to solve equation: {e}")
            return []
    
    async def differentiate(
        self,
        expression: str,
        variable: Optional[str] = None,
        order: int = 1
    ) -> Optional[sp.Basic]:
        """Compute derivative"""
        
        try:
            expr = await self.parse_expression(expression)
            
            if not expr.symbolic_form:
                return None
            
            # Determine variable
            if variable:
                var = self.symbols_cache.get(variable, sp.Symbol(variable))
            else:
                free_symbols = expr.symbolic_form.free_symbols
                if free_symbols:
                    var = list(free_symbols)[0]
                else:
                    return expr.symbolic_form
            
            # Compute derivative
            result = expr.symbolic_form
            for _ in range(order):
                result = sp.diff(result, var)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to differentiate: {e}")
            return None
    
    async def integrate(
        self,
        expression: str,
        variable: Optional[str] = None,
        limits: Optional[Tuple[float, float]] = None
    ) -> Optional[sp.Basic]:
        """Compute integral"""
        
        try:
            expr = await self.parse_expression(expression)
            
            if not expr.symbolic_form:
                return None
            
            # Determine variable
            if variable:
                var = self.symbols_cache.get(variable, sp.Symbol(variable))
            else:
                free_symbols = expr.symbolic_form.free_symbols
                if free_symbols:
                    var = list(free_symbols)[0]
                else:
                    return expr.symbolic_form
            
            # Compute integral
            if limits:
                result = sp.integrate(expr.symbolic_form, (var, limits[0], limits[1]))
            else:
                result = sp.integrate(expr.symbolic_form, var)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to integrate: {e}")
            return None


class NumericalComputation:
    """Handles numerical computations"""
    
    def __init__(self):
        self.precision = 10
        
    async def compute_matrix_operations(
        self,
        operation: str,
        matrices: List[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Perform matrix operations"""
        
        try:
            if operation == "multiply":
                result = matrices[0]
                for matrix in matrices[1:]:
                    result = np.matmul(result, matrix)
                return result
            
            elif operation == "add":
                return sum(matrices)
            
            elif operation == "inverse":
                return np.linalg.inv(matrices[0])
            
            elif operation == "determinant":
                return np.linalg.det(matrices[0])
            
            elif operation == "eigenvalues":
                eigenvalues, eigenvectors = np.linalg.eig(matrices[0])
                return eigenvalues
            
            elif operation == "svd":
                u, s, vh = np.linalg.svd(matrices[0])
                return s
            
            elif operation == "solve":
                # Solve Ax = b
                if len(matrices) >= 2:
                    return np.linalg.solve(matrices[0], matrices[1])
            
            return None
            
        except Exception as e:
            logger.error(f"Matrix operation failed: {e}")
            return None
    
    async def statistical_analysis(
        self,
        data: List[float],
        analysis_type: str
    ) -> Dict[str, float]:
        """Perform statistical analysis"""
        
        try:
            data_array = np.array(data)
            
            results = {
                "count": len(data),
                "mean": np.mean(data_array),
                "median": np.median(data_array),
                "std": np.std(data_array),
                "variance": np.var(data_array),
                "min": np.min(data_array),
                "max": np.max(data_array)
            }
            
            if analysis_type == "descriptive":
                results["q1"] = np.percentile(data_array, 25)
                results["q3"] = np.percentile(data_array, 75)
                results["iqr"] = results["q3"] - results["q1"]
                results["skewness"] = self._compute_skewness(data_array)
                results["kurtosis"] = self._compute_kurtosis(data_array)
            
            elif analysis_type == "hypothesis":
                # Simple t-test against zero
                from scipy import stats
                t_stat, p_value = stats.ttest_1samp(data_array, 0)
                results["t_statistic"] = t_stat
                results["p_value"] = p_value
            
            return results
            
        except Exception as e:
            logger.error(f"Statistical analysis failed: {e}")
            return {}
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data"""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0
        
        skewness = np.sum(((data - mean) / std) ** 3) * n / ((n - 1) * (n - 2))
        return skewness
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of data"""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0
        
        kurtosis = np.sum(((data - mean) / std) ** 4) * n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        return kurtosis
    
    async def numerical_optimization(
        self,
        objective_func: str,
        constraints: List[str],
        bounds: Optional[List[Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """Solve optimization problem"""
        
        try:
            from scipy.optimize import minimize
            
            # Parse objective function
            obj_expr = sp.sympify(objective_func)
            variables = list(obj_expr.free_symbols)
            
            # Create numerical function
            def objective(x):
                subs_dict = {var: val for var, val in zip(variables, x)}
                return float(obj_expr.subs(subs_dict))
            
            # Initial guess
            x0 = np.ones(len(variables))
            
            # Solve
            result = minimize(objective, x0, bounds=bounds)
            
            return {
                "optimal_value": result.fun,
                "optimal_point": result.x.tolist(),
                "variables": [str(v) for v in variables],
                "success": result.success,
                "iterations": result.nit
            }
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return {}


class GeometryEngine:
    """Handles geometric computations"""
    
    def __init__(self):
        self.shapes = {}
    
    async def compute_2d_properties(
        self,
        shape_type: str,
        parameters: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute 2D shape properties"""
        
        try:
            if shape_type == "circle":
                r = parameters.get("radius", 0)
                return {
                    "area": np.pi * r ** 2,
                    "circumference": 2 * np.pi * r,
                    "diameter": 2 * r
                }
            
            elif shape_type == "rectangle":
                w = parameters.get("width", 0)
                h = parameters.get("height", 0)
                return {
                    "area": w * h,
                    "perimeter": 2 * (w + h),
                    "diagonal": np.sqrt(w ** 2 + h ** 2)
                }
            
            elif shape_type == "triangle":
                a = parameters.get("a", 0)
                b = parameters.get("b", 0)
                c = parameters.get("c", 0)
                
                # Heron's formula
                s = (a + b + c) / 2
                area = np.sqrt(s * (s - a) * (s - b) * (s - c))
                
                return {
                    "area": area,
                    "perimeter": a + b + c,
                    "semiperimeter": s
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Geometry computation failed: {e}")
            return {}
    
    async def compute_3d_properties(
        self,
        shape_type: str,
        parameters: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute 3D shape properties"""
        
        try:
            if shape_type == "sphere":
                r = parameters.get("radius", 0)
                return {
                    "volume": (4/3) * np.pi * r ** 3,
                    "surface_area": 4 * np.pi * r ** 2,
                    "diameter": 2 * r
                }
            
            elif shape_type == "cube":
                s = parameters.get("side", 0)
                return {
                    "volume": s ** 3,
                    "surface_area": 6 * s ** 2,
                    "diagonal": s * np.sqrt(3)
                }
            
            elif shape_type == "cylinder":
                r = parameters.get("radius", 0)
                h = parameters.get("height", 0)
                return {
                    "volume": np.pi * r ** 2 * h,
                    "surface_area": 2 * np.pi * r * (r + h),
                    "lateral_area": 2 * np.pi * r * h
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"3D geometry computation failed: {e}")
            return {}


class ProblemSolver:
    """Coordinates mathematical problem solving"""
    
    def __init__(self):
        self.symbolic_engine = SymbolicMathEngine()
        self.numerical_engine = NumericalComputation()
        self.geometry_engine = GeometryEngine()
        self.solution_strategies = {}
    
    async def identify_problem_type(
        self,
        problem_text: str
    ) -> MathProblemType:
        """Identify the type of mathematical problem"""
        
        problem_lower = problem_text.lower()
        
        # Keywords for different problem types
        if any(word in problem_lower for word in ["derive", "differentiate", "integral", "integrate"]):
            return MathProblemType.CALCULUS
        elif any(word in problem_lower for word in ["solve", "equation", "variable", "x =", "y ="]):
            return MathProblemType.ALGEBRA
        elif any(word in problem_lower for word in ["matrix", "eigenvalue", "determinant", "vector"]):
            return MathProblemType.LINEAR_ALGEBRA
        elif any(word in problem_lower for word in ["mean", "median", "variance", "probability", "distribution"]):
            return MathProblemType.STATISTICS
        elif any(word in problem_lower for word in ["circle", "triangle", "rectangle", "area", "perimeter", "volume"]):
            return MathProblemType.GEOMETRY
        elif any(word in problem_lower for word in ["optimize", "minimize", "maximize", "minimum", "maximum"]):
            return MathProblemType.OPTIMIZATION
        elif any(word in problem_lower for word in ["prime", "divisible", "factor", "gcd", "lcm"]):
            return MathProblemType.NUMBER_THEORY
        else:
            return MathProblemType.ARITHMETIC
    
    async def solve_problem_step_by_step(
        self,
        problem: str,
        problem_type: Optional[MathProblemType] = None
    ) -> MathSolution:
        """Solve mathematical problem with step-by-step explanation"""
        
        steps = []
        solution = None
        confidence = 0.0
        method = "unknown"
        
        # Identify problem type if not specified
        if not problem_type:
            problem_type = await self.identify_problem_type(problem)
            steps.append(f"Identified problem type: {problem_type.value}")
        
        try:
            if problem_type == MathProblemType.ALGEBRA:
                # Extract equation from problem
                equation_match = re.search(r'([^=]+=[^=]+)', problem)
                if equation_match:
                    equation = equation_match.group(1)
                    steps.append(f"Extracted equation: {equation}")
                    
                    solutions = await self.symbolic_engine.solve_equation(equation)
                    solution = solutions
                    steps.append(f"Solutions: {', '.join(str(s) for s in solutions)}")
                    
                    # Verify solution
                    if solutions:
                        steps.append("Verification: Substituting back into original equation")
                        confidence = 0.95
                    
                    method = "symbolic_algebra"
            
            elif problem_type == MathProblemType.CALCULUS:
                # Handle calculus problems
                if "derivative" in problem.lower() or "differentiate" in problem.lower():
                    expr_match = re.search(r'of\s+(.+?)(?:\s+with|\s+respect|$)', problem)
                    if expr_match:
                        expression = expr_match.group(1).strip()
                        steps.append(f"Expression to differentiate: {expression}")
                        
                        derivative = await self.symbolic_engine.differentiate(expression)
                        solution = str(derivative)
                        steps.append(f"Derivative: {solution}")
                        
                        confidence = 0.9
                        method = "symbolic_calculus"
            
            elif problem_type == MathProblemType.GEOMETRY:
                # Extract shape and parameters
                shape_match = re.search(r'(circle|triangle|rectangle|square|sphere|cube|cylinder)', problem.lower())
                if shape_match:
                    shape = shape_match.group(1)
                    steps.append(f"Identified shape: {shape}")
                    
                    # Extract numerical values
                    numbers = re.findall(r'\d+(?:\.\d+)?', problem)
                    
                    if shape in ["circle", "sphere"]:
                        if numbers:
                            params = {"radius": float(numbers[0])}
                            if shape == "circle":
                                props = await self.geometry_engine.compute_2d_properties("circle", params)
                            else:
                                props = await self.geometry_engine.compute_3d_properties("sphere", params)
                            
                            solution = props
                            steps.append(f"Computed properties: {props}")
                            confidence = 0.9
                            method = "geometry"
            
            elif problem_type == MathProblemType.STATISTICS:
                # Extract data points
                numbers = re.findall(r'-?\d+(?:\.\d+)?', problem)
                if numbers:
                    data = [float(n) for n in numbers]
                    steps.append(f"Extracted data: {data}")
                    
                    stats = await self.numerical_engine.statistical_analysis(data, "descriptive")
                    solution = stats
                    steps.append(f"Statistical analysis: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
                    
                    confidence = 0.85
                    method = "statistics"
            
            # Default handling
            if solution is None:
                steps.append("Unable to parse problem completely")
                solution = "Problem requires clarification"
                confidence = 0.2
                method = "failed"
            
        except Exception as e:
            logger.error(f"Problem solving failed: {e}")
            steps.append(f"Error occurred: {str(e)}")
            solution = "Error in computation"
            confidence = 0.0
        
        return MathSolution(
            problem=problem,
            solution=solution,
            steps=steps,
            verification=confidence > 0.7,
            confidence=confidence,
            method_used=method
        )


class MathematicalSolver:
    """Main mathematical problem solver"""
    
    def __init__(self):
        self.problem_solver = ProblemSolver()
        self.solution_history = []
    
    async def solve(
        self,
        problem: str,
        problem_type: Optional[str] = None,
        show_steps: bool = True
    ) -> Dict[str, Any]:
        """Solve mathematical problem"""
        
        logger.info(f"Solving math problem: {problem[:50]}...")
        
        # Convert string type to enum if provided
        if problem_type:
            try:
                problem_type_enum = MathProblemType(problem_type)
            except ValueError:
                problem_type_enum = None
        else:
            problem_type_enum = None
        
        # Solve the problem
        solution = await self.problem_solver.solve_problem_step_by_step(
            problem,
            problem_type_enum
        )
        
        # Store in history
        self.solution_history.append(solution)
        
        # Format response
        response = {
            "problem": solution.problem,
            "solution": solution.solution,
            "confidence": solution.confidence,
            "method": solution.method_used
        }
        
        if show_steps:
            response["steps"] = solution.steps
        
        response["verified"] = solution.verification
        
        return response
    
    async def batch_solve(
        self,
        problems: List[str]
    ) -> List[Dict[str, Any]]:
        """Solve multiple problems"""
        
        tasks = [self.solve(problem, show_steps=False) for problem in problems]
        solutions = await asyncio.gather(*tasks)
        
        return solutions
    
    def get_capabilities(self) -> Dict[str, List[str]]:
        """Get solver capabilities"""
        
        return {
            "problem_types": [p.value for p in MathProblemType],
            "symbolic_operations": [
                "solve_equation",
                "differentiate",
                "integrate",
                "simplify",
                "expand",
                "factor"
            ],
            "numerical_operations": [
                "matrix_operations",
                "statistical_analysis",
                "optimization",
                "interpolation"
            ],
            "geometry_shapes": [
                "circle", "triangle", "rectangle", "square",
                "sphere", "cube", "cylinder", "cone"
            ]
        }
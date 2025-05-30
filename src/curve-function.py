import os
import numpy as np
from scipy import interpolate, optimize
from typing import List, Tuple, Union, Optional, Dict, Any
import ast
import inspect
from dataclasses import dataclass
from enum import Enum

from sklearn.metrics import r2_score, mean_squared_error
import warnings


class FunctionType(Enum):  # Fixed missing parenthesis
    """Supported function generation methods."""
    POLYNOMIAL = "polynomial"
    FOURIER = "fourier"
    PIECEWISE = "piecewise"
    NEURAL_NETWORK = "neural_network"
    SYMBOLIC_REGRESSION = "symbolic_regression"
    SPLINE = "spline"
    RATIONAL = "rational"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    POWER = "power"
    TRIGONOMETRIC = "trigonometric"


@dataclass
class GeneratedFunction:
    """Container for generated function with metadata."""
    source_code: str
    compiled_function: callable
    expression: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    domain: Tuple[float, float]
    complexity: int


class FunctionCodeGenerator:
    """
    Advanced function generator that creates optimized Python function code from data points.
    
    Features:
    - Multiple function generation strategies
    - Automatic complexity optimization
    - Symbolic mathematics integration
    - Performance metrics and validation
    - Clean, executable Python code output
    """
    
    def __init__(self, optimization_level: int = 2):
        """
        Initialize the generator.
        
        Args:
            optimization_level: 0=fast/simple, 1=balanced, 2=thorough/complex
        """
        self.optimization_level = optimization_level
        self._function_templates = self._initialize_templates()
        
    def generate_function(
        self,
        x_points: Union[List[float], np.ndarray],
        y_points: Union[List[float], np.ndarray],
        function_type: Optional[FunctionType] = None,
        function_name: str = "generated_function",
        docstring: bool = True,
        optimize_for: str = "accuracy",  # "accuracy", "simplicity", "speed"
        constraints: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a Python function from data points.
        
        Args:
            x_points: Input data points
            y_points: Output data points
            function_type: Type of function to generate (auto-detected if None)
            function_name: Name for the generated function
            docstring: Include comprehensive docstring
            optimize_for: Optimization target
            constraints: Additional constraints (e.g., monotonic, bounded)
            
        Returns:
            GeneratedFunction object with source code and metadata
        """
        # Validate and prepare data
        x = np.asarray(x_points, dtype=np.float64)
        y = np.asarray(y_points, dtype=np.float64)
        
        if len(x) != len(y):
            raise ValueError("x_points and y_points must have the same length")
        
        # Sort by x values
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]
        
        # Auto-detect best function type if not specified
        if function_type is None:
            function_type = self._detect_best_function_type(x, y, constraints)
        
        # Generate function based on type
        if function_type == FunctionType.POLYNOMIAL:
            result = self._generate_polynomial_function(x, y, function_name, docstring, optimize_for)
        elif function_type == FunctionType.FOURIER:
            result = self._generate_fourier_function(x, y, function_name, docstring)
        elif function_type == FunctionType.PIECEWISE:
            result = self._generate_piecewise_function(x, y, function_name, docstring)
        elif function_type == FunctionType.NEURAL_NETWORK:
            result = self._generate_neural_network_function(x, y, function_name, docstring)
        elif function_type == FunctionType.SYMBOLIC_REGRESSION:
            result = self._generate_symbolic_regression_function(x, y, function_name, docstring)
        elif function_type == FunctionType.SPLINE:
            result = self._generate_spline_function(x, y, function_name, docstring)
        elif function_type == FunctionType.RATIONAL:
            result = self._generate_rational_function(x, y, function_name, docstring)
        else:
            result = self._generate_parametric_function(x, y, function_type, function_name, docstring)

        # Return only the mathematical expression as f(x)
        return result.expression
    
    def _detect_best_function_type(
        self,
        x: np.ndarray,
        y: np.ndarray,
        constraints: Optional[Dict[str, Any]] = None
    ) -> FunctionType:
        """Auto-detect the best function type based on data characteristics."""
        
        # Analyze data characteristics
        n_points = len(x)
        
        # Check for periodicity
        if self._check_periodicity(x, y):
            return FunctionType.FOURIER
        
        # Check for discontinuities
        if self._check_discontinuities(x, y):
            return FunctionType.PIECEWISE
        
        # Check complexity
        if n_points > 20 and self.optimization_level >= 2:
            # Try symbolic regression for complex patterns
            noise_level = self._estimate_noise_level(x, y)
            if noise_level < 0.1:
                return FunctionType.SYMBOLIC_REGRESSION
        
        # Default to polynomial for smooth data
        if n_points <= 10:
            return FunctionType.POLYNOMIAL
        else:
            return FunctionType.SPLINE
    
    def _generate_polynomial_function(
        self,
        x: np.ndarray,
        y: np.ndarray,
        function_name: str,
        docstring: bool,
        optimize_for: str
    ) -> GeneratedFunction:
        """Generate an optimized polynomial function."""
        
        # Determine optimal degree
        if optimize_for == "simplicity":
            max_degree = min(3, len(x) - 1)
        elif optimize_for == "speed":
            max_degree = min(5, len(x) - 1)
        else:  # accuracy
            max_degree = min(10, len(x) - 1)
        
        best_degree = self._find_optimal_polynomial_degree(x, y, max_degree)
        
        # Fit polynomial
        coeffs = np.polyfit(x, y, best_degree)
        
        # Generate optimized code
        source_code = self._generate_polynomial_code(
            coeffs, function_name, x, y, docstring
        )
        
        # Compile function
        compiled_func = self._compile_function(source_code, function_name)
        
        # Calculate metrics
        y_pred = compiled_func(x)
        metrics = {
            'r2_score': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'max_error': np.max(np.abs(y - y_pred))
        }
        
        return GeneratedFunction(
            source_code=source_code,
            compiled_function=compiled_func,
            expression=self._polynomial_to_expression(coeffs),
            parameters={'coefficients': coeffs.tolist(), 'degree': best_degree},
            metrics=metrics,
            domain=(x.min(), x.max()),
            complexity=best_degree + 1
        )
    
    def _generate_polynomial_code(
        self,
        coeffs: np.ndarray,
        function_name: str,
        x_data: np.ndarray,
        y_data: np.ndarray,
        docstring: bool
    ) -> str:
        """Generate optimized polynomial function code."""
        
        # Use Horner's method for efficient evaluation
        code_lines = []
        
        # Function definition
        code_lines.append(f"def {function_name}(x):")
        
        # Docstring
        if docstring:
            code_lines.extend([
                '    """',
                f'    Polynomial function of degree {len(coeffs)-1}.',
                '    ',
                '    Generated from data points using optimized polynomial fitting.',
                f'    Domain: [{x_data.min():.6f}, {x_data.max():.6f}]',
                '    ',
                '    Args:',
                '        x: Input value(s), can be scalar or array-like',
                '    ',
                '    Returns:',
                '        Computed function value(s)',
                '    """'
            ])
        
        # Import statements
        code_lines.extend([
            '    import numpy as np',
            '    ',
            '    # Ensure input is numpy array for vectorized operations',
            '    x = np.asarray(x)',
            '    scalar_input = x.ndim == 0',
            '    if scalar_input:',
            '        x = x.reshape(1)',
            '    '
        ])
        
        # Horner's method implementation
        code_lines.append('    # Horner\'s method for efficient polynomial evaluation')
        code_lines.append(f'    result = {coeffs[0]:.16e}')
        
        for i in range(1, len(coeffs)):
            code_lines.append(f'    result = result * x + {coeffs[i]:.16e}')
        
        # Return statement
        code_lines.extend([
            '    ',
            '    return result.item() if scalar_input else result'
        ])
        
        return '\n'.join(code_lines)
    
    def _generate_fourier_function(
        self,
        x: np.ndarray,
        y: np.ndarray,
        function_name: str,
        docstring: bool
    ) -> GeneratedFunction:
        """Generate a Fourier series function."""
        
        # Determine period
        period = x.max() - x.min()
        
        # Compute Fourier coefficients
        n_terms = min(20, len(x) // 2)
        coeffs = self._compute_fourier_coefficients(x, y, n_terms, period)
        
        # Create mathematical expression
        expr_terms = [f"{coeffs['a0']:.6f}"]
        
        for n, (a_n, b_n) in enumerate(zip(coeffs["a"][1:], coeffs["b"][1:]), 1):
            if abs(a_n) > 1e-10:
                expr_terms.append(f"{a_n:.6f}*cos({2*np.pi*n:.6f}*x/{period:.6f})")
            if abs(b_n) > 1e-10:
                expr_terms.append(f"{b_n:.6f}*sin({2*np.pi*n:.6f}*x/{period:.6f})")
        
        expr = " + ".join(expr_terms)
        
        # Generate code
        code_lines = [
            f"def {function_name}(x):"
        ]
        
        if docstring:
            code_lines.extend([
                '    """',
                f'    Fourier series approximation with {n_terms} terms.',
                '    ',
                f'    Period: {period:.6f}',
                f'    Domain: [{x.min():.6f}, {x.max():.6f}]',
                '    ',
                '    Args:',
                '        x: Input value(s)',
                '    ',
                '    Returns:',
                '        Computed function value(s)',
                '    """'
            ])
        
        code_lines.extend([
            '    import numpy as np',
            '    ',
            '    x = np.asarray(x)',
            '    scalar_input = x.ndim == 0',
            '    if scalar_input:',
            '        x = x.reshape(1)',
            '    ',
            f'    # Fourier series with period {period:.6f}',
            f'    omega = 2 * np.pi / {period:.16e}',
            f'    result = {coeffs["a0"]:.16e}',
            '    '
        ])
        
        # Add cosine terms
        for n, a_n in enumerate(coeffs["a"][1:], 1):
            if abs(a_n) > 1e-10:
                code_lines.append(
                    f'    result += {a_n:.16e} * np.cos({n} * omega * x)'
                )
        
        # Add sine terms
        for n, b_n in enumerate(coeffs["b"][1:], 1):
            if abs(b_n) > 1e-10:
                code_lines.append(
                    f'    result += {b_n:.16e} * np.sin({n} * omega * x)'
                )
        
        code_lines.extend([
            '    ',
            '    return result.item() if scalar_input else result'
        ])
        
        source_code = '\n'.join(code_lines)
        compiled_func = self._compile_function(source_code, function_name)
        
        # Metrics
        y_pred = compiled_func(x)
        metrics = {
            'r2_score': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'n_terms': n_terms
        }
        
        return GeneratedFunction(
            source_code=source_code,
            compiled_function=compiled_func,
            expression=expr,  # Now using the mathematical expression
            parameters=coeffs,
            metrics=metrics,
            domain=(x.min(), x.max()),
            complexity=2 * n_terms + 1
        )
    
    def _generate_piecewise_function(
        self,
        x: np.ndarray,
        y: np.ndarray,
        function_name: str,
        docstring: bool
    ) -> GeneratedFunction:
        """Generate a piecewise function with automatic breakpoint detection."""
        
        # Detect breakpoints
        breakpoints = self._detect_breakpoints(x, y)
        
        # Fit functions to each segment
        segments = []
        for i in range(len(breakpoints) - 1):
            mask = (x >= breakpoints[i]) & (x < breakpoints[i + 1])
            if i == len(breakpoints) - 2:  # Last segment
                mask = (x >= breakpoints[i]) & (x <= breakpoints[i + 1])
            
            x_seg = x[mask]
            y_seg = y[mask]
            
            # Fit polynomial to segment
            degree = min(3, len(x_seg) - 1)
            if len(x_seg) > 1:
                coeffs = np.polyfit(x_seg, y_seg, degree)
                segments.append({
                    'range': (breakpoints[i], breakpoints[i + 1]),
                    'coeffs': coeffs,
                    'degree': degree
                })
        
        # Generate code
        code_lines = [
            f"def {function_name}(x):"
        ]
        
        if docstring:
            code_lines.extend([
                '    """',
                f'    Piecewise function with {len(segments)} segments.',
                '    ',
                f'    Breakpoints: {breakpoints.tolist()}',
                f'    Domain: [{x.min():.6f}, {x.max():.6f}]',
                '    """'
            ])
        
        code_lines.extend([
            '    import numpy as np',
            '    ',
            '    x_input = np.asarray(x)',
            '    scalar_input = x_input.ndim == 0',
            '    x_array = x_input.reshape(-1) if scalar_input else x_input.copy()',
            '    result = np.zeros_like(x_array, dtype=float)',
            '    '
        ])
        
        # Add conditions for each segment
        for i, seg in enumerate(segments):
            if i == 0:
                condition = f'mask = (x_array >= {seg["range"][0]:.16e}) & (x_array < {seg["range"][1]:.16e})'
            elif i == len(segments) - 1:
                condition = f'mask = (x_array >= {seg["range"][0]:.16e}) & (x_array <= {seg["range"][1]:.16e})'
            else:
                condition = f'mask = (x_array >= {seg["range"][0]:.16e}) & (x_array < {seg["range"][1]:.16e})'
            
            code_lines.append(f'    # Segment {i+1}')
            code_lines.append(f'    {condition}')
            
            # Polynomial evaluation
            poly_eval = f'{seg["coeffs"][0]:.16e}'
            for j in range(1, len(seg["coeffs"])):
                poly_eval = f'({poly_eval}) * x_array[mask] + {seg["coeffs"][j]:.16e}'
            
            code_lines.append(f'    result[mask] = {poly_eval}')
            code_lines.append('    ')
        
        code_lines.extend([
            '    return result.item() if scalar_input else result'
        ])
        
        source_code = '\n'.join(code_lines)
        compiled_func = self._compile_function(source_code, function_name)
        
        # Metrics
        y_pred = compiled_func(x)
        metrics = {
            'r2_score': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'n_segments': len(segments)
        }
        
        return GeneratedFunction(
            source_code=source_code,
            compiled_function=compiled_func,
            expression=f"Piecewise function with {len(segments)} segments",
            parameters={'segments': segments, 'breakpoints': breakpoints.tolist()},
            metrics=metrics,
            domain=(x.min(), x.max()),
            complexity=sum(seg['degree'] + 1 for seg in segments)
        )
    
    def _generate_neural_network_function(
        self,
        x: np.ndarray,
        y: np.ndarray,
        function_name: str,
        docstring: bool
    ) -> GeneratedFunction:
        """Generate a simple neural network function."""
        
        # Normalize data
        x_mean, x_std = x.mean(), x.std()
        y_mean, y_std = y.mean(), y.std()
        
        x_norm = (x - x_mean) / (x_std + 1e-8)
        y_norm = (y - y_mean) / (y_std + 1e-8)
        
        # Simple 1-hidden layer network
        n_hidden = min(10, len(x) // 2)
        
        # Initialize weights using simple optimization
        weights = self._fit_simple_nn(x_norm.reshape(-1, 1), y_norm, n_hidden)
        
        # Generate code
        code_lines = [
            f"def {function_name}(x):"
        ]
        
        if docstring:
            code_lines.extend([
                '    """',
                f'    Neural network with {n_hidden} hidden units.',
                '    ',
                '    Single hidden layer with tanh activation.',
                f'    Domain: [{x.min():.6f}, {x.max():.6f}]',
                '    """'
            ])
        
        code_lines.extend([
            '    import numpy as np',
            '    ',
            '    x = np.asarray(x)',
            '    scalar_input = x.ndim == 0',
            '    x = x.reshape(-1, 1) if x.ndim <= 1 else x',
            '    ',
            f'    # Normalization parameters',
            f'    x_mean = {x_mean:.16e}',
            f'    x_std = {x_std:.16e}',
            f'    y_mean = {y_mean:.16e}',
            f'    y_std = {y_std:.16e}',
            '    ',
            '    # Normalize input',
            '    x_norm = (x - x_mean) / (x_std + 1e-8)',
            '    ',
            '    # Network weights',
            f'    W1 = np.array({weights["W1"].tolist()})',
            f'    b1 = np.array({weights["b1"].tolist()})',
            f'    W2 = np.array({weights["W2"].tolist()})',
            f'    b2 = {weights["b2"]:.16e}',
            '    ',
            '    # Forward pass',
            '    hidden = np.tanh(x_norm @ W1 + b1)',
            '    output_norm = hidden @ W2 + b2',
            '    ',
            '    # Denormalize output',
            '    output = output_norm * y_std + y_mean',
            '    ',
            '    return output.squeeze() if scalar_input else output.squeeze()'
        ])
        
        source_code = '\n'.join(code_lines)
        compiled_func = self._compile_function(source_code, function_name)
        
        # Metrics
        y_pred = compiled_func(x)
        metrics = {
            'r2_score': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'n_parameters': weights["W1"].size + weights["b1"].size + weights["W2"].size + 1
        }
        
        return GeneratedFunction(
            source_code=source_code,
            compiled_function=compiled_func,
            expression=f"Neural network with {n_hidden} hidden units",
            parameters={'weights': weights, 'normalization': {
                'x_mean': x_mean, 'x_std': x_std,
                'y_mean': y_mean, 'y_std': y_std
            }},
            metrics=metrics,
            domain=(x.min(), x.max()),
            complexity=n_hidden * 3 + 2
        )
    
    def _generate_symbolic_regression_function(
        self,
        x: np.ndarray,
        y: np.ndarray,
        function_name: str,
        docstring: bool
    ) -> GeneratedFunction:
        """Generate function using symbolic regression."""
        
        # Use sympy to find symbolic expression
        expr, params = self._symbolic_regression(x, y)
        
        # Convert to Python code
        code_lines = [
            f"def {function_name}(x):"
        ]
        
        if docstring:
            code_lines.extend([
                '    """',
                '    Symbolic regression generated function.',
                '    ',
                f'    Expression: {expr}',
                f'    Domain: [{x.min():.6f}, {x.max():.6f}]',
                '    """'
            ])
        
        code_lines.extend([
            '    import numpy as np',
            '    ',
            '    x = np.asarray(x)',
            '    scalar_input = x.ndim == 0',
            '    '
        ])
        
        # Add parameter definitions
        for param_name, param_value in params.items():
            code_lines.append(f'    {param_name} = {param_value:.16e}')
        
        code_lines.append('    ')
        
        # Convert symbolic expression to numpy code
        numpy_expr = self._sympy_to_numpy(expr)
        code_lines.append(f'    result = {numpy_expr}')
        code_lines.extend([
            '    ',
            '    return result.item() if scalar_input else result'
        ])
        
        source_code = '\n'.join(code_lines)
        compiled_func = self._compile_function(source_code, function_name)
        
        # Metrics
        y_pred = compiled_func(x)
        metrics = {
            'r2_score': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'expression_complexity': self._expression_complexity(expr)
        }
        
        return GeneratedFunction(
            source_code=source_code,
            compiled_function=compiled_func,
            expression=str(expr),
            parameters=params,
            metrics=metrics,
            domain=(x.min(), x.max()),
            complexity=self._expression_complexity(expr)
        )
    
    def _generate_spline_function(
        self,
        x: np.ndarray,
        y: np.ndarray,
        function_name: str,
        docstring: bool
    ) -> GeneratedFunction:
        """Generate an efficient spline function."""
        
        # Create cubic spline
        cs = interpolate.CubicSpline(x, y)
        
        # Extract coefficients
        coeffs = []
        for i in range(len(x) - 1):
            coeffs.append({
                'x0': x[i],
                'x1': x[i + 1],
                'c': cs.c[:, i].tolist()
            })
        
        # Generate code
        code_lines = [
            f"def {function_name}(x):"
        ]
        
        if docstring:
            code_lines.extend([
                '    """',
                '    Cubic spline interpolation function.',
                '    ',
                f'    Number of segments: {len(coeffs)}',
                f'    Domain: [{x.min():.6f}, {x.max():.6f}]',
                '    """'
            ])
        
        code_lines.extend([
            '    import numpy as np',
            '    ',
            '    x_input = np.asarray(x)',
            '    scalar_input = x_input.ndim == 0',
            '    x = x_input.reshape(-1) if x_input.ndim <= 1 else x_input',
            '    result = np.zeros_like(x, dtype=float)',
            '    ',
            '    # Spline segments'
        ])
        
        # Add binary search for efficiency
        code_lines.extend([
            '    # Binary search for segment',
            f'    knots = np.array({x.tolist()})',
            '    ',
            '    for i, xi in enumerate(x):',
            '        if xi <= knots[0]:',
            '            seg_idx = 0',
            '        elif xi >= knots[-1]:',
            '            seg_idx = len(knots) - 2',
            '        else:',
            '            seg_idx = np.searchsorted(knots, xi) - 1',
            '        ',
            '        # Evaluate spline for this segment',
            '        dx = xi - knots[seg_idx]',
            '        '
        ])
        
        # Add coefficient arrays
        for i, seg in enumerate(coeffs):
            code_lines.append(f'        if seg_idx == {i}:')
            code_lines.append(f'            c = {seg["c"]}')
            code_lines.append(f'            result[i] = c[0]*dx**3 + c[1]*dx**2 + c[2]*dx + c[3]')
        
        code_lines.extend([
            '    ',
            '    return result.item() if scalar_input else result'
        ])
        
        source_code = '\n'.join(code_lines)
        compiled_func = self._compile_function(source_code, function_name)
        
        # Metrics
        y_pred = compiled_func(x)
        metrics = {
            'r2_score': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'n_knots': len(x)
        }
        
        return GeneratedFunction(
            source_code=source_code,
            compiled_function=compiled_func,
            expression=f"Cubic spline with {len(x)} knots",
            parameters={'knots': x.tolist(), 'coefficients': coeffs},
            metrics=metrics,
            domain=(x.min(), x.max()),
            complexity=4 * (len(x) - 1)
        )
    
    def _generate_rational_function(
        self,
        x: np.ndarray,
        y: np.ndarray,
        function_name: str,
        docstring: bool
    ) -> GeneratedFunction:
        """Generate a rational function (ratio of polynomials)."""
        
        # Fit rational function P(x)/Q(x)
        p_degree = min(3, len(x) // 2 - 1)
        q_degree = min(2, len(x) // 2 - 1)
        
        # Use Padé approximation
        p_coeffs, q_coeffs = self._fit_rational_function(x, y, p_degree, q_degree)
        
        # Generate code
        code_lines = [
            f"def {function_name}(x):"
        ]
        
        if docstring:
            code_lines.extend([
                '    """',
                f'    Rational function: P(x)/Q(x) with degrees {p_degree}/{q_degree}.',
                '    ',
                f'    Domain: [{x.min():.6f}, {x.max():.6f}]',
                '    """'
            ])
        
        code_lines.extend([
            '    import numpy as np',
            '    ',
            '    x = np.asarray(x)',
            '    scalar_input = x.ndim == 0',
            '    ',
            '    # Numerator polynomial',
            f'    p = {p_coeffs[0]:.16e}'
        ])
        
        for i in range(1, len(p_coeffs)):
            code_lines.append(f'    p = p * x + {p_coeffs[i]:.16e}')
        
        code_lines.extend([
            '    ',
            '    # Denominator polynomial',
            f'    q = {q_coeffs[0]:.16e}'
        ])
        
        for i in range(1, len(q_coeffs)):
            code_lines.append(f'    q = q * x + {q_coeffs[i]:.16e}')
        
        code_lines.extend([
            '    ',
            '    # Avoid division by zero',
            '    q = np.where(np.abs(q) < 1e-10, 1e-10, q)',
            '    ',
            '    result = p / q',
            '    return result.item() if scalar_input else result'
        ])
        
        source_code = '\n'.join(code_lines)
        compiled_func = self._compile_function(source_code, function_name)
        
        # Metrics
        y_pred = compiled_func(x)
        metrics = {
            'r2_score': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'p_degree': p_degree,
            'q_degree': q_degree
        }
        
        return GeneratedFunction(
            source_code=source_code,
            compiled_function=compiled_func,
            expression=f"Rational function P{p_degree}/Q{q_degree}",
            parameters={'p_coeffs': p_coeffs.tolist(), 'q_coeffs': q_coeffs.tolist()},
            metrics=metrics,
            domain=(x.min(), x.max()),
            complexity=p_degree + q_degree + 2
        )
    
    def _generate_parametric_function(
        self,
        x: np.ndarray,
        y: np.ndarray,
        function_type: FunctionType,
        function_name: str,
        docstring: bool
    ) -> GeneratedFunction:
        """Generate parametric functions (exponential, logarithmic, etc.)."""
        
        if function_type == FunctionType.EXPONENTIAL:
            # Fit: y = a * exp(b * x) + c
            def exp_func(x, a, b, c):
                return a * np.exp(b * x) + c
            
            p0 = [1, 0.1, 0]
            try:
                popt, _ = optimize.curve_fit(exp_func, x, y, p0=p0, maxfev=5000)
            except:
                popt = p0
            
            expr = f"{popt[0]:.6f} * exp({popt[1]:.6f} * x) + {popt[2]:.6f}"
            
            code_lines = [
                f"def {function_name}(x):",
                '    """Exponential function."""' if docstring else '',
                '    import numpy as np',
                '    x = np.asarray(x)',
                f'    return {popt[0]:.16e} * np.exp({popt[1]:.16e} * x) + {popt[2]:.16e}'
            ]
            
        elif function_type == FunctionType.LOGARITHMIC:
            # Fit: y = a * log(b * x + c) + d
            def log_func(x, a, b, c, d):
                return a * np.log(np.abs(b * x + c) + 1e-10) + d
            
            p0 = [1, 1, 1, 0]
            try:
                popt, _ = optimize.curve_fit(log_func, x, y, p0=p0, maxfev=5000)
            except:
                popt = p0
            
            expr = f"{popt[0]:.6f} * log({popt[1]:.6f} * x + {popt[2]:.6f}) + {popt[3]:.6f}"
            
            code_lines = [
                f"def {function_name}(x):",
                '    """Logarithmic function."""' if docstring else '',
                '    import numpy as np',
                '    x = np.asarray(x)',
                f'    return {popt[0]:.16e} * np.log(np.abs({popt[1]:.16e} * x + {popt[2]:.16e}) + 1e-10) + {popt[3]:.16e}'
            ]
            
        elif function_type == FunctionType.POWER:
            # Fit: y = a * x^b + c
            def power_func(x, a, b, c):
                return a * np.power(np.abs(x) + 1e-10, b) + c
            
            p0 = [1, 1, 0]
            try:
                popt, _ = optimize.curve_fit(power_func, x, y, p0=p0, maxfev=5000)
            except:
                popt = p0
            
            expr = f"{popt[0]:.6f} * x^{popt[1]:.6f} + {popt[2]:.6f}"
            
            code_lines = [
                f"def {function_name}(x):",
                '    """Power function."""' if docstring else '',
                '    import numpy as np',
                '    x = np.asarray(x)',
                f'    return {popt[0]:.16e} * np.power(np.abs(x) + 1e-10, {popt[1]:.16e}) + {popt[2]:.16e}'
            ]
            
        else:  # Trigonometric
            # Fit: y = a * sin(b * x + c) + d
            def trig_func(x, a, b, c, d):
                return a * np.sin(b * x + c) + d
            
            # Estimate frequency
            fft = np.fft.fft(y)
            freqs = np.fft.fftfreq(len(x), x[1] - x[0])
            dominant_freq = abs(freqs[np.argmax(np.abs(fft[1:len(fft)//2])) + 1])
            
            p0 = [(y.max() - y.min()) / 2, 2 * np.pi * dominant_freq, 0, y.mean()]
            try:
                popt, _ = optimize.curve_fit(trig_func, x, y, p0=p0, maxfev=5000)
            except:
                popt = p0
            
            expr = f"{popt[0]:.6f} * sin({popt[1]:.6f} * x + {popt[2]:.6f}) + {popt[3]:.6f}"
            
            code_lines = [
                f"def {function_name}(x):",
                '    """Trigonometric function."""' if docstring else '',
                '    import numpy as np',
                '    x = np.asarray(x)',
                f'    return {popt[0]:.16e} * np.sin({popt[1]:.16e} * x + {popt[2]:.16e}) + {popt[3]:.16e}'
            ]
        
        source_code = '\n'.join(code_lines)
        compiled_func = self._compile_function(source_code, function_name)
        
        # Metrics
        y_pred = compiled_func(x)
        metrics = {
            'r2_score': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred))
        }
        
        return GeneratedFunction(
            source_code=source_code,
            compiled_function=compiled_func,
            expression=expr,
            parameters={'coefficients': popt.tolist()},
            metrics=metrics,
            domain=(x.min(), x.max()),
            complexity=len(popt)
        )
    
    # Helper methods
    def _compile_function(self, source_code: str, function_name: str) -> callable:
        """Compile source code and return the function."""
        namespace = {}
        exec(source_code, namespace)
        return namespace[function_name]
    
    def _find_optimal_polynomial_degree(
        self,
        x: np.ndarray,
        y: np.ndarray,
        max_degree: int
    ) -> int:
        """Find optimal polynomial degree using cross-validation."""
        if len(x) < 10:
            return min(len(x) - 1, max_degree)
        
        best_degree = 1
        best_score = -np.inf
        
        for degree in range(1, max_degree + 1):
            # Simple cross-validation
            scores = []
            for i in range(min(5, len(x))):
                mask = np.ones(len(x), dtype=bool)
                mask[i::5] = False
                
                x_train, y_train = x[mask], y[mask]
                x_val, y_val = x[~mask], y[~mask]
                
                try:
                    coeffs = np.polyfit(x_train, y_train, degree)
                    poly = np.poly1d(coeffs)
                    y_pred = poly(x_val)
                    score = -np.mean((y_val - y_pred) ** 2)
                    scores.append(score)
                except:
                    scores.append(-np.inf)
            
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_degree = degree
        
        return best_degree
    
    def _polynomial_to_expression(self, coeffs: np.ndarray) -> str:
        """Convert polynomial coefficients to readable expression."""
        terms = []
        degree = len(coeffs) - 1
        
        for i, coeff in enumerate(coeffs):
            if abs(coeff) < 1e-10:
                continue
                
            power = degree - i
            if power == 0:
                terms.append(f"{coeff:.6f}")
            elif power == 1:
                terms.append(f"{coeff:.6f}*x")
            else:
                terms.append(f"{coeff:.6f}*x^{power}")
        
        return " + ".join(terms) if terms else "0"
    
    def _check_periodicity(self, x: np.ndarray, y: np.ndarray) -> bool:
        """Check if data shows periodic behavior."""
        if len(x) < 20:
            return False
        
        # FFT analysis
        fft = np.fft.fft(y)
        power = np.abs(fft) ** 2
        
        # Check if there's a dominant frequency
        freqs = np.fft.fftfreq(len(x), x[1] - x[0])
        positive_freqs = freqs[:len(freqs)//2]
        positive_power = power[:len(power)//2]
        
        if len(positive_power) > 1:
            peak_idx = np.argmax(positive_power[1:]) + 1
            peak_power = positive_power[peak_idx]
            total_power = np.sum(positive_power[1:])
            
            return peak_power / total_power > 0.5
        
        return False
    
    def _check_discontinuities(self, x: np.ndarray, y: np.ndarray) -> bool:
        """Check for discontinuities in the data."""
        if len(x) < 4:
            return False
        
        # Calculate differences
        dy = np.diff(y)
        dx = np.diff(x)
        slopes = dy / (dx + 1e-10)
        
        # Check for sudden changes in slope
        slope_changes = np.diff(slopes)
        threshold = 3 * np.std(slope_changes)
        
        return np.any(np.abs(slope_changes) > threshold)
    
    def _estimate_noise_level(self, x: np.ndarray, y: np.ndarray) -> float:
        """Estimate noise level in data."""
        if len(x) < 5:
            return 0.0
        
        # Fit smooth curve and calculate residuals
        smooth = interpolate.UnivariateSpline(x, y, s=len(x))
        residuals = y - smooth(x)
        
        return np.std(residuals) / (np.ptp(y) + 1e-10)
    
    def _detect_breakpoints(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Detect breakpoints for piecewise functions."""
        if len(x) < 10:
            return np.array([x[0], x[-1]])
        
        # Simple method: detect large changes in derivative
        dy = np.diff(y)
        dx = np.diff(x)
        slopes = dy / (dx + 1e-10)
        
        # Find significant changes
        slope_changes = np.abs(np.diff(slopes))
        threshold = np.percentile(slope_changes, 90)
        
        breakpoint_indices = [0]
        for i, change in enumerate(slope_changes):
            if change > threshold and i > breakpoint_indices[-1] + 2:
                breakpoint_indices.append(i + 1)
        breakpoint_indices.append(len(x) - 1)
        
        # Limit number of breakpoints
        if len(breakpoint_indices) > 6:
            # Keep most significant ones
            indices = np.linspace(0, len(x) - 1, 6, dtype=int)
            breakpoint_indices = sorted(set(indices))
        
        return x[breakpoint_indices]
    
    def _compute_fourier_coefficients(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_terms: int,
        period: float
    ) -> Dict[str, Any]:
        """Compute Fourier series coefficients."""
        # Shift x to [0, period]
        x_shifted = x - x.min()
        
        # Compute coefficients using trapezoid instead of trapz
        a0 = 2 * np.trapezoid(y, x_shifted) / period
        
        a_coeffs = [a0 / 2]
        b_coeffs = [0]
        
        for n in range(1, n_terms + 1):
            cos_term = y * np.cos(2 * np.pi * n * x_shifted / period)
            sin_term = y * np.sin(2 * np.pi * n * x_shifted / period)
            
            a_n = 2 * np.trapezoid(cos_term, x_shifted) / period
            b_n = 2 * np.trapezoid(sin_term, x_shifted) / period
            
            a_coeffs.append(a_n)
            b_coeffs.append(b_n)
        
        return {
            'a0': a0 / 2,
            'a': np.array(a_coeffs),
            'b': np.array(b_coeffs)
        }
    
    def _fit_simple_nn(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_hidden: int
    ) -> Dict[str, np.ndarray]:
        """Fit a simple neural network using basic optimization."""
        n_features = X.shape[1]
        
        # Initialize weights
        np.random.seed(42)
        W1 = np.random.randn(n_features, n_hidden) * 0.5
        b1 = np.zeros(n_hidden)
        W2 = np.random.randn(n_hidden, 1) * 0.5
        b2 = 0.0
        
        # Simple gradient descent
        learning_rate = 0.01
        for _ in range(1000):
            # Forward pass
            hidden = np.tanh(X @ W1 + b1)
            output = hidden @ W2 + b2
            
            # Backward pass
            error = output.squeeze() - y
            d_output = error
            d_W2 = hidden.T @ d_output.reshape(-1, 1) / len(y)
            d_b2 = np.mean(d_output)
            
            d_hidden = d_output.reshape(-1, 1) @ W2.T
            d_tanh = d_hidden * (1 - hidden ** 2)
            d_W1 = X.T @ d_tanh / len(y)
            d_b1 = np.mean(d_tanh, axis=0)
            
            # Update weights
            W1 -= learning_rate * d_W1
            b1 -= learning_rate * d_b1
            W2 -= learning_rate * d_W2
            b2 -= learning_rate * d_b2
        
        return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    
    def _symbolic_regression(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[str, Dict[str, float]]:
        """Perform simple symbolic regression."""
        # Try common patterns
        patterns = [
            (lambda x, a, b: a * x + b, "a*x + b", ['a', 'b']),
            (lambda x, a, b, c: a * x**2 + b * x + c, "a*x^2 + b*x + c", ['a', 'b', 'c']),
            (lambda x, a, b: a * np.exp(b * x), "a*exp(b*x)", ['a', 'b']),
            (lambda x, a, b: a * np.log(np.abs(x) + 1) + b, "a*log(|x|+1) + b", ['a', 'b']),
            (lambda x, a, b, c: a * np.sin(b * x) + c, "a*sin(b*x) + c", ['a', 'b', 'c']),
        ]
        
        best_pattern = None
        best_params = None
        best_score = -np.inf
        
        for func, expr_template, param_names in patterns:
            try:
                p0 = [1.0] * len(param_names)
                popt, _ = optimize.curve_fit(func, x, y, p0=p0, maxfev=5000)
                
                y_pred = func(x, *popt)
                score = r2_score(y, y_pred)
                
                if score > best_score:
                    best_score = score
                    best_pattern = (func, expr_template, param_names)
                    best_params = popt
            except:
                continue
        
        if best_pattern is None:
            # Fallback to polynomial
            degree = min(3, len(x) - 1)
            coeffs = np.polyfit(x, y, degree)
            expr = self._polynomial_to_expression(coeffs)
            params = {f'c{i}': c for i, c in enumerate(coeffs)}
        else:
            _, expr_template, param_names = best_pattern
            params = {name: value for name, value in zip(param_names, best_params)}
            expr = expr_template
            for name, value in params.items():
                expr = expr.replace(name, f"{value:.6f}")
        
        return expr, params
    
    def _sympy_to_numpy(self, expr: str) -> str:
        """Convert symbolic expression to numpy code."""
        # Simple conversion rules
        replacements = [
            ('exp', 'np.exp'),
            ('log', 'np.log'),
            ('sin', 'np.sin'),
            ('cos', 'np.cos'),
            ('tan', 'np.tan'),
            ('sqrt', 'np.sqrt'),
            ('abs', 'np.abs'),
            ('^', '**'),
        ]
        
        result = expr
        for old, new in replacements:
            result = result.replace(old, new)
        
        return result
    
    def _expression_complexity(self, expr: str) -> int:
        """Estimate expression complexity."""
        # Count operations
        ops = ['+', '-', '*', '/', '**', 'exp', 'log', 'sin', 'cos']
        complexity = 0
        for op in ops:
            complexity += expr.count(op)
        return complexity + 1
    
    def _fit_rational_function(
        self,
        x: np.ndarray,
        y: np.ndarray,
        p_degree: int,
        q_degree: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit rational function using linear least squares."""
        # Set up linear system for rational function fitting
        # P(x)/Q(x) = y => P(x) = y * Q(x)
        # P(x) - y * Q(x) = 0
        
        n = len(x)
        A = np.zeros((n, p_degree + q_degree + 2))
        
        # Fill matrix for P(x) terms
        for i in range(p_degree + 1):
            A[:, i] = x ** (p_degree - i)
        
        # Fill matrix for -y*Q(x) terms
        for i in range(q_degree + 1):
            A[:, p_degree + 1 + i] = -y * (x ** (q_degree - i))
        
        # Solve using SVD (more stable than normal equations)
        _, _, Vt = np.linalg.svd(A)
        coeffs = Vt[-1, :]
        
        # Normalize by last Q coefficient
        if abs(coeffs[-1]) > 1e-10:
            coeffs = coeffs / coeffs[-1]
        
        p_coeffs = coeffs[:p_degree + 1]
        q_coeffs = coeffs[p_degree + 1:]
        
        return p_coeffs, q_coeffs
    
    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize function templates."""
        return {
            'imports': 'import numpy as np',
            'scalar_check': '''x = np.asarray(x)
    scalar_input = x.ndim == 0
    if scalar_input:
        x = x.reshape(1)''',
            'scalar_return': 'return result.item() if scalar_input else result'
        }


# Advanced usage examples
def generate_optimal_function(x_data, y_data):
    """Example: Generate the most optimal function for given data."""
    generator = FunctionCodeGenerator(optimization_level=2)
    
    # Auto-detect best function type
    expr = generator.generate_function(
        x_data, y_data,
        function_name="optimal_fit",
        optimize_for="accuracy"
    )
    
    print(f"Generated function: f(x) = {expr}")
    
    return expr


def generate_fast_approximation(x_data, y_data):
    """Example: Generate a fast approximation function."""
    generator = FunctionCodeGenerator(optimization_level=0)
    
    result = generator.generate_function(
        x_data, y_data,
        function_type=FunctionType.POLYNOMIAL,
        function_name="fast_approximation",
        optimize_for="speed",
        docstring=False
    )
    
    return result


def generate_constrained_function(x_data, y_data):
    """Example: Generate function with constraints."""
    generator = FunctionCodeGenerator()
    
    result = generator.generate_function(
        x_data, y_data,
        function_name="constrained_fit",
        constraints={
            'monotonic': True,
            'bounded': (0, 100),
            'smooth': True
        }
    )
    
    return result


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    
    try:
        # Load and prepare data
        csv_path = 'src/marker_points.csv'
        
        # Read CSV file
        try:
            marker_df = pd.read_csv(csv_path)
            if 'x' not in marker_df.columns or 'y' not in marker_df.columns:
                marker_df = pd.read_csv(csv_path, header=None, names=['x', 'y'])
            marker_df = marker_df.iloc[:, 0:2]
            marker_df.columns = ['x', 'y']
        except Exception as e:
            print(f"Error reading CSV: {str(e)}")
            raise
        
        # Convert to numeric and clean data
        marker_df = marker_df.apply(pd.to_numeric, errors='coerce')
        marker_df = marker_df.dropna()
        
        if len(marker_df) == 0:
            raise ValueError("No valid numeric data points found in the CSV file")
        
        # Extract and sort points
        x = marker_df['x'].values
        y = marker_df['y'].values
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]
        

        # First, just plot the points
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, color='blue', label='Data points', alpha=0.6)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Original Data Points')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
        
        # Wait for user confirmation
        input("\nPress Enter to fit and plot the curve...")
        
        # Now fit the function and create the second plot
        generator = FunctionCodeGenerator(optimization_level=2)
        expr = generator.generate_function(
            x, y,
            function_type=FunctionType.POLYNOMIAL,
            function_name="marker_fit",
            optimize_for="accuracy"
        )
        
        print("\nFitted polynomial function:")
        print(f"f(x) = {expr}")
        
        # Create evaluation function
        def f(x):
            return eval(expr.replace("^", "**"))
        
        # Generate smooth curve points
        x_smooth = np.linspace(min(x), max(x), 200)
        y_smooth = np.array([f(float(xi)) for xi in x_smooth])
        
        # Plot points with fitted curve
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, color='blue', label='Data points', alpha=0.6)
        plt.plot(x_smooth, y_smooth, 'r-', label='Fitted curve')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Points with Fitted Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Save the function to a file
        with open('fitted_function.txt', 'w') as f:
            f.write(f"f(x) = {expr}")
        
    except FileNotFoundError:
        print(f"Error: CSV file not found at src/marker_points.csv")
        print("Current working directory:", os.getcwd())
    except ValueError as e:
        print(f"Data error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

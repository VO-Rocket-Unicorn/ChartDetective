from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware #totalk between different ports
from pydantic import BaseModel #helps in checking the data types of the input
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, UnivariateSpline, BarycentricInterpolator, interp1d
import io
import base64
from typing import List, Optional, Tuple, Dict, Any
import matplotlib
from sympy import false
matplotlib.use('Agg')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Point(BaseModel):
    x: float
    y: float

class AxisInfo(BaseModel):
    x_name: str
    y_name: str
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
# ADDED: Global storage for interpolators - exactly like scipy tutorial
stored_interpolators: Dict[int, Any] = {}

class SingleCurveRequest(BaseModel):
    points: List[Point]
    seriesIndex: int
    color: Optional[str] = None
    label: Optional[str] = None
    axis_info: AxisInfo
    fitting_method: Optional[str] = "linear"
    is_traced: Optional[bool] = False

class CurveResponse(BaseModel):
    equation: str
    python_function: str
    plot_image: str
    coefficients: List[float]
    degree: int
    seriesIndex: int
    axis_info: AxisInfo
    fitting_method: str
    r_squared: float
    point_count: int# ADDED: New evaluation request/response models
class EvaluationRequest(BaseModel):
    seriesIndex: int
    x_value: float

class EvaluationResponse(BaseModel):
    y_value: float
    is_within_bounds: bool
    x_min: float
    x_max: float
    is_extrapolation: bool
    warning: Optional[str] = None

def fit_linear_interpolation(x_data, y_data):
    """
    Linear interpolation using scipy.interpolate.interp1d
    """
    # Sort data by x values
    sorted_indices = np.argsort(x_data)
    x_sorted = np.array(x_data)[sorted_indices]
    y_sorted = np.array(y_data)[sorted_indices]
    
    # Create linear interpolator
    # linear_interp = interp1d(x_sorted, y_sorted, kind='linear', 
    #                         bounds_error=False, fill_value='extrapolate')
    linear_interp = interp1d(x_sorted, y_sorted, kind='linear', 
                            bounds_error=False, fill_value=np.nan) # # bounds_error=True to raise error if out of range of x_data
    
    equation = f"Piecewise Linear Interpolation through {len(x_data)} points"
    
    # For coefficients, return slope and intercept of first segment
    if len(x_data) >= 2:
        slope = (y_sorted[1] - y_sorted[0]) / (x_sorted[1] - x_sorted[0])
        intercept = y_sorted[0] - slope * x_sorted[0]
        coefficients = [intercept, slope]
    else:
        coefficients = [0.0, 0.0]
    
    return linear_interp, coefficients, equation, 1

def smart_lagrange_interpolation(x_data, y_data):
    """Smart Lagrange interpolation that avoids Runge's phenomenon"""
    n = len(x_data)
    
    # For small datasets, use exact Lagrange
    if n <= 6:
        interpolator = BarycentricInterpolator(x_data, y_data)
        coeffs = np.polyfit(x_data, y_data, n-1)
        poly_func = np.poly1d(coeffs)
        equation = f"Lagrange Polynomial (degree {n-1}): f(x) = {str(poly_func)}"
        return interpolator, coeffs.tolist(), equation, n-1
    else:
        # For larger datasets, use piecewise approach
        return piecewise_lagrange_interpolation(x_data, y_data)

def piecewise_lagrange_interpolation(x_data, y_data):
    """Piecewise Lagrange interpolation to avoid oscillations"""
    n = len(x_data)
    sorted_indices = np.argsort(x_data)
    x_sorted = np.array(x_data)[sorted_indices]
    y_sorted = np.array(y_data)[sorted_indices]
    
    spline = CubicSpline(x_sorted, y_sorted)
    equation = f"Piecewise Lagrange (Cubic Spline) through {n} points"
    coefficients = [0.0] * min(n, 8)
    
    return spline, coefficients, equation, 3

def fit_cubic_spline(x_data, y_data):
    """Standard cubic spline interpolation"""
    sorted_indices = np.argsort(x_data)
    x_sorted = np.array(x_data)[sorted_indices]
    y_sorted = np.array(y_data)[sorted_indices]
    
    spline = CubicSpline(x_sorted, y_sorted)
    equation = f"Cubic Spline Interpolation through {len(x_data)} points"
    coefficients = spline.c.flatten()[:8] if len(spline.c.flatten()) > 8 else spline.c.flatten()
    
    return spline, coefficients.tolist(), equation, 3

def fit_smoothing_spline(x_data, y_data):
    """Smoothing spline with automatic parameter selection"""
    sorted_indices = np.argsort(x_data)
    x_sorted = np.array(x_data)[sorted_indices]
    y_sorted = np.array(y_data)[sorted_indices]
    
    spline = UnivariateSpline(x_sorted, y_sorted, s=None)
    equation = f"Smoothing Spline (auto-smoothed) through {len(x_data)} points"
    coefficients = [0.0] * min(len(x_data), 8)
    
    return spline, coefficients, equation, 3

def fit_standard_polynomial(x_data, y_data):
    """Standard polynomial fitting with degree limitation"""
    n = len(x_data)
    max_degree = min(n-1, 8)
    
    coefficients = np.polyfit(x_data, y_data, max_degree)
    poly_func = np.poly1d(coefficients)
    equation = f"Polynomial (degree {max_degree}): f(x) = {str(poly_func)}"
    
    return poly_func, coefficients.tolist(), equation, max_degree

def calculate_r_squared(y_true, y_pred):
    """Calculate R-squared value"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

def generate_plot_for_curve(x_data, y_data, fitted_func, equation, color, label, axis_info, fitting_method, r_squared, point_count):
    """Generate high-quality plot with proper visualization"""
    try:
        plt.figure(figsize=(12, 8))
        
        # Use chart's axis ranges
        x_range = axis_info.x_range
        y_range = axis_info.y_range
        
        # Generate smooth curve points
        x_smooth = np.linspace(x_range[0], x_range[1], 1000)
        
        try:
            if hasattr(fitted_func, '__call__'):
                y_smooth = fitted_func(x_smooth)
            else:
                y_smooth = [fitted_func(x) for x in x_smooth]
        except:
            # Fallback for edge cases
            x_smooth = np.linspace(min(x_data), max(x_data), 500)
            y_smooth = fitted_func(x_smooth)
        
        # Plot the fitted curve
        plt.plot(x_smooth, y_smooth, color=color or 'blue', linewidth=3, 
                label=f'{fitting_method.replace("_", " ").title()} Fit (R² = {r_squared:.4f})', alpha=0.9)
        
        # Plot sample of original data points (if too many, subsample for visibility)
        if len(x_data) > 50:
            # Subsample for visualization
            step = len(x_data) // 25
            x_plot = x_data[::step]
            y_plot = y_data[::step]
            plt.scatter(x_plot, y_plot, color='red', s=50, zorder=5, 
                       label=f'Data Points (showing {len(x_plot)} of {len(x_data)})', 
                       edgecolors='darkred', linewidth=1, alpha=0.7)
        else:
            plt.scatter(x_data, y_data, color='red', s=100, zorder=5, 
                       label='Data Points', edgecolors='darkred', linewidth=2, alpha=0.9)
        
        # Enhanced plot styling
        plt.xlabel(axis_info.x_name, fontsize=14, fontweight='bold')
        plt.ylabel(axis_info.y_name, fontsize=14, fontweight='bold')
        plt.title(f'{label} - {fitting_method.replace("_", " ").title()} Interpolation\nR² = {r_squared:.4f} ({point_count} points)', 
                 fontsize=14, fontweight='bold', pad=20)
        
        # Set axis limits
        plt.xlim(x_range[0], x_range[1])
        plt.ylim(y_range[0], y_range[1])
        
        plt.legend(fontsize=12, framealpha=0.9)
        plt.grid(True, alpha=0.4, linestyle='--')
        
        # Add enhanced info box
        info_text = f'Method: {fitting_method.replace("_", " ").title()}\nPoints: {point_count}\nR²: {r_squared:.2f}\n{axis_info.x_name} vs {axis_info.y_name}'
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        # Improve layout
        plt.tight_layout()
        
        # Save with high quality
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        buffer.seek(0)
        
        plot_data = buffer.getvalue()
        plot_base64 = base64.b64encode(plot_data).decode()
        
        return f"data:image/png;base64,{plot_base64}"
        
    except Exception as e:
        print(f"Plot generation error: {e}")
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

def generate_python_code(fitted_func, coefficients, x_data, y_data, fitting_method, series_index, label, axis_info, r_squared, point_count):
    """Generate comprehensive Python code"""
    
    if fitting_method == "linear":
        # Linear interpolation implementation
        function_code = f"""
import numpy as np
from numpy import nan
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def {label.lower().replace(' ', '_')}_linear_function(x):
    '''
    Linear Interpolation for {label}
    
    Uses piecewise linear interpolation between consecutive data points.
    This method connects adjacent points with straight lines.
    
    Chart Axis: {axis_info.x_name} vs {axis_info.y_name}
    Method: {fitting_method.replace('_', ' ').title()}
    R-squared: {r_squared:.6f}
    Data points: {point_count}
    Point density: High (traced curve data)
    '''
    x_data = np.array({x_data[:50]})  # First 50 points for brevity
    y_data = np.array({y_data[:50]})
    
    # Full dataset (truncated in display)
    x_full = np.array({x_data})
    y_full = np.array({y_data})
    
    # Create linear interpolator
    linear_interp = interp1d(x_full, y_full, kind='linear', 
                            bounds_error=False, fill_value=nan)  
    
    if isinstance(x, (list, np.ndarray)):
        return linear_interp(x)
    else:
        return float(linear_interp(x))

# Enhanced plotting with chart axis scales
def plot_{label.lower().replace(' ', '_')}_linear():
    x_vals = np.linspace({axis_info.x_range[0]:.2f}, {axis_info.x_range[1]:.2f}, 1000)
    y_vals = {label.lower().replace(' ', '_')}_linear_function(x_vals)
    
    plt.figure(figsize=(12, 8))
    plt.plot(x_vals, y_vals, 'b-', linewidth=3, label='{label} - Linear Interpolation', alpha=0.9)
    
    # Plot sample of data points
    x_sample = np.array({x_data[::max(1, len(x_data)//50)]})
    y_sample = np.array({y_data[::max(1, len(y_data)//50)]})
    plt.scatter(x_sample, y_sample, color='red', s=50, label='Sample Data Points', zorder=5,
               edgecolors='darkred', linewidth=1)
    
    plt.xlabel('{axis_info.x_name}', fontsize=14, fontweight='bold')
    plt.ylabel('{axis_info.y_name}', fontsize=14, fontweight='bold')
    plt.title('{label} - Linear Interpolation (R² = {r_squared:.4f}, {point_count} points)', fontsize=14, fontweight='bold')
    
    plt.xlim({axis_info.x_range[0]:.2f}, {axis_info.x_range[1]:.2f})
    plt.ylim({axis_info.y_range[0]:.2f}, {axis_info.y_range[1]:.2f})
    
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.4, linestyle='--')
    plt.tight_layout()
    plt.show()

# Evaluation function
def evaluate_{label.lower().replace(' ', '_')}(x_values):
    '''
    Evaluate the {label} linear interpolation at given x values
    Valid range: {axis_info.x_range[0]:.2f} to {axis_info.x_range[1]:.2f}
    '''
    if isinstance(x_values, (int, float)):
        return {label.lower().replace(' ', '_')}_linear_function(x_values)
    else:
        return [{label.lower().replace(' ', '_')}_linear_function(x) for x in x_values]

# Run the plotting function
if __name__ == "__main__":
    print("Running {label} Linear Interpolation...")
    plot_{label.lower().replace(' ', '_')}_linear()
    print("\\nTesting evaluation at x=2:")
    print(f"f(2) = {{evaluate_{label.lower().replace(' ', '_')}(2):.4f}}")
"""
    
    elif "lagrange" in fitting_method.lower():
        # Lagrange-specific implementation
        function_code = f"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BarycentricInterpolator

def {label.lower().replace(' ', '_')}_lagrange_function(x):
    '''
    Lagrange Interpolation for {label}
    
    Theory: Constructs unique polynomial of degree ≤ N-1 through N points
    Using barycentric form for numerical stability with {point_count} traced points
    
    Chart Axis: {axis_info.x_name} vs {axis_info.y_name}
    Method: {fitting_method.replace('_', ' ').title()}
    R-squared: {r_squared:.6f}
    Data points: {point_count} (high density from curve tracing)
    '''
    # Sample of data points (full dataset too large to display)
    x_sample = np.array({x_data[::max(1, len(x_data)//100)]})
    y_sample = np.array({y_data[::max(1, len(y_data)//100)]})
    
    # Full dataset for interpolation
    x_data = np.array({x_data})
    y_data = np.array({y_data})
    
    # Create barycentric interpolator
    interpolator = BarycentricInterpolator(x_data, y_data)
    
    if isinstance(x, (list, np.ndarray)):
        return interpolator(x)
    else:
        return float(interpolator(x))

# Enhanced plotting function
def plot_{label.lower().replace(' ', '_')}_lagrange():
    x_vals = np.linspace({axis_info.x_range[0]:.2f}, {axis_info.x_range[1]:.2f}, 1000)
    y_vals = {label.lower().replace(' ', '_')}_lagrange_function(x_vals)
    
    plt.figure(figsize=(12, 8))
    plt.plot(x_vals, y_vals, 'b-', linewidth=3, label='{label} - Lagrange Interpolation', alpha=0.9)
    
    # Plot sample of traced points
    x_sample = np.array({x_data[::max(1, len(x_data)//50)]})
    y_sample = np.array({y_data[::max(1, len(y_data)//50)]})
    plt.scatter(x_sample, y_sample, color='red', s=30, label='Traced Data Sample', zorder=5,
               edgecolors='darkred', linewidth=1, alpha=0.7)
    
    plt.xlabel('{axis_info.x_name}', fontsize=14, fontweight='bold')
    plt.ylabel('{axis_info.y_name}', fontsize=14, fontweight='bold')
    plt.title('{label} - Lagrange Interpolation (R² = {r_squared:.4f}, {point_count} points)', fontsize=14, fontweight='bold')
    
    plt.xlim({axis_info.x_range[0]:.2f}, {axis_info.x_range[1]:.2f})
    plt.ylim({axis_info.y_range[0]:.2f}, {axis_info.y_range[1]:.2f})
    
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.4, linestyle='--')
    plt.tight_layout()
    plt.show()

# Run the plotting function
if __name__ == "__main__":
    print("Running {label} Lagrange Interpolation with {point_count} traced points...")
    plot_{label.lower().replace(' ', '_')}_lagrange()
"""
    
    else:  # Spline methods
        function_code = f"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, UnivariateSpline

def {label.lower().replace(' ', '_')}_spline_function():
    '''
    {fitting_method.replace('_', ' ').title()} Interpolation for {label}
    Chart Axis: {axis_info.x_name} vs {axis_info.y_name}
    R-squared: {r_squared:.6f}
    Data points: {point_count} (traced curve data)
    '''
    x_data = np.array({x_data})
    y_data = np.array({y_data})
    
    # Sort data for spline fitting
    sorted_indices = np.argsort(x_data)
    x_sorted = x_data[sorted_indices]
    y_sorted = y_data[sorted_indices]
    
    # Create spline
    {'spline = CubicSpline(x_sorted, y_sorted)' if 'cubic' in fitting_method else 'spline = UnivariateSpline(x_sorted, y_sorted, s=None)'}
    
    return spline

# Enhanced plotting function
def plot_{label.lower().replace(' ', '_')}_spline():
    spline_func = {label.lower().replace(' ', '_')}_spline_function()
    x_vals = np.linspace({axis_info.x_range[0]:.2f}, {axis_info.x_range[1]:.2f}, 1000)
    y_vals = spline_func(x_vals)
    
    plt.figure(figsize=(12, 8))
    plt.plot(x_vals, y_vals, 'b-', linewidth=3, label='{label} - {fitting_method.replace("_", " ").title()}', alpha=0.9)
    
    # Plot sample of data points
    x_sample = np.array({x_data[::max(1, len(x_data)//50)]})
    y_sample = np.array({y_data[::max(1, len(y_data)//50)]})
    plt.scatter(x_sample, y_sample, color='red', s=50, label='Sample Data Points', zorder=5,
               edgecolors='darkred', linewidth=1)
    
    plt.xlabel('{axis_info.x_name}', fontsize=14, fontweight='bold')
    plt.ylabel('{axis_info.y_name}', fontsize=14, fontweight='bold')
    plt.title('{label} - {fitting_method.replace("_", " ").title()} (R² = {r_squared:.4f}, {point_count} points)', fontsize=14, fontweight='bold')
    
    plt.xlim({axis_info.x_range[0]:.2f}, {axis_info.x_range[1]:.2f})
    plt.ylim({axis_info.y_range[0]:.2f}, {axis_info.y_range[1]:.2f})
    
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.4, linestyle='--')
    plt.tight_layout()
    plt.show()

# Run the plotting function
if __name__ == "__main__":
    print("Running {label} {fitting_method.replace('_', ' ').title()} with {point_count} points...")
    plot_{label.lower().replace(' ', '_')}_spline()
"""
    
    return function_code

@app.post("/fit-single-curve", response_model=CurveResponse)
async def fit_single_curve(request: SingleCurveRequest):
    try:
        if len(request.points) < 2:
            raise HTTPException(status_code=400, detail="At least 2 points required")
        
        # Extract coordinates
        x_data = [p.x for p in request.points]
        y_data = [p.y for p in request.points]
        
        # Choose fitting method
        fitting_method = request.fitting_method
        
        if fitting_method == "linear":
            fitted_func, coefficients, equation, degree = fit_linear_interpolation(x_data, y_data)
        elif fitting_method == "smart_lagrange":
            fitted_func, coefficients, equation, degree = smart_lagrange_interpolation(x_data, y_data)
        elif fitting_method == "piecewise_lagrange":
            fitted_func, coefficients, equation, degree = piecewise_lagrange_interpolation(x_data, y_data)
        elif fitting_method == "cubic_spline":
            fitted_func, coefficients, equation, degree = fit_cubic_spline(x_data, y_data)
        elif fitting_method == "smoothing_spline":
            fitted_func, coefficients, equation, degree = fit_smoothing_spline(x_data, y_data)
        elif fitting_method == "polynomial":
            fitted_func, coefficients, equation, degree = fit_standard_polynomial(x_data, y_data)
        else:
            fitted_func, coefficients, equation, degree = fit_linear_interpolation(x_data, y_data)
            # ADDED: Store the interpolator for later evaluation - key addition!

        stored_interpolators[request.seriesIndex] = {   
        'interpolator': fitted_func,  # This is the 'f' from scipy tutorial
        'x_range': [min(x_data), max(x_data)],
        'method': fitting_method
        }

        # Calculate R-squared
        y_pred = fitted_func(np.array(x_data))
        r_squared = calculate_r_squared(np.array(y_data), y_pred)
        
        # Generate plot
        plot_image = generate_plot_for_curve(
            x_data, y_data, fitted_func, equation, 
            request.color, request.label, request.axis_info, fitting_method, r_squared, len(x_data)
        )
        
        # Generate Python code
        python_function = generate_python_code(
            fitted_func, coefficients, x_data, y_data, fitting_method,
            request.seriesIndex, request.label or f"Series_{request.seriesIndex}", 
            request.axis_info, r_squared, len(x_data)
        )
        
        return CurveResponse(
            equation=equation,
            python_function=python_function,
            plot_image=plot_image,
            coefficients=coefficients,
            degree=degree,
            seriesIndex=request.seriesIndex,
            axis_info=request.axis_info,
            fitting_method=fitting_method or "linear",
            r_squared=float(r_squared),
            point_count=len(x_data)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Curve fitting failed: {str(e)}")
# ADDED: New evaluation endpoint - exactly like scipy tutorial
@app.post("/evaluate-curve", response_model=EvaluationResponse)
async def evaluate_curve(request: EvaluationRequest):
    """Evaluate interpolator exactly like scipy tutorial: y_new = f(x_new)"""
    try:
        # Check if interpolator exists
        if request.seriesIndex not in stored_interpolators:
            raise HTTPException(
                status_code=404, 
                detail=f"Values not found for the series {request.seriesIndex}. Please fit the curve first."
            )
        
        stored_data = stored_interpolators[request.seriesIndex]
        f = stored_data['interpolator']  # This is our 'f' from the tutorial
        x_range = stored_data['x_range']
        
        # Single line evaluation exactly like the tutorial: y_new = f(x_new)
        x_new = request.x_value
        y_new = float(f(x_new))
        
        # Check if result is NaN (scipy returns NaN for out-of-bounds)
        warning = None
        if np.isnan(y_new):
            warning = f"X value {x_new:.3f} is outside the range [{x_range[0]:.3f}, {x_range[1]:.3f}]"
            y_new = 0.0  # or handle as needed
        
        # Boundary checking
        is_within_bounds = x_range[0] <= x_new <= x_range[1]
        
        return EvaluationResponse(
            y_value=y_new,
            is_within_bounds=is_within_bounds,
            x_min=x_range[0],
            x_max=x_range[1],
            is_extrapolation=not is_within_bounds,
            warning=warning
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
    
@app.get("/debug/interpolators")
async def debug_interpolators():
    """Debug endpoint to check stored interpolators"""
    return {
        "stored_series": list(stored_interpolators.keys()),
        "count": len(stored_interpolators),
        "message": "These are the interpolators available for evaluation"
    }

@app.get("/")
async def root():
    return {"message": "Enhanced Curve Fitting API with Linear Interpolation and Traced Data Support"}

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { ChartExtractorState } from './ChartExtractor';
import { ChartElementType } from '../datastructure/chartelements/ChartElement';
import SubChartElement from '../datastructure/chartelements/SubChartElement';
import AxisCoordinate2D from '../datastructure/AxisCoordinate2D';

interface Point {
    x: number;
    y: number;
}

interface CurveData {
    points: Point[];
    seriesIndex: number;
    color: string;
    label: string;
    isTraced: boolean;
}

interface CurveResult {
    equation: string;
    python_function: string;
    plot_image: string;
    coefficients: number[];
    degree: number;
    seriesIndex: number;
    axis_info: {
        x_name: string;
        y_name: string;
        x_range: [number, number];
        y_range: [number, number];
    };
    fitting_method: string;
    r_squared: number;
    point_count: number;
}

interface CurveDisplayProps {
    state: ChartExtractorState;
}

const CurveDisplay: React.FC<CurveDisplayProps> = ({ state }) => {
    const [curveResults, setCurveResults] = useState<Map<number, CurveResult>>(new Map());
    const [loading, setLoading] = useState<Set<number>>(new Set());
    const [selectedCurve, setSelectedCurve] = useState<number | null>(null);
    const [autoGenerate, setAutoGenerate] = useState(true);
    const [fittingMethod, setFittingMethod] = useState<string>('linear');
    const [useTracedData, setUseTracedData] = useState(false);
    const [curveEvaluations, setCurveEvaluations] = useState<Map<number, { x: string, y: string, error: string }>>(new Map());
    // Convert AxisCoordinate2D to simple arrays like ReconstructedChart
    const axisCoordToArray = (coord: AxisCoordinate2D[]): { x: number[], y: number[] } => {
        const x = [];
        const y = [];
        for (const pt of coord) {
            x.push(pt.x.value);
            y.push(pt.y.value);
        }
        return { x, y };
    };

    // Get dense traced data points from ReconstructedChart rendering
    const getTracedDataPoints = (serie: any, seriesIndex: number): Point[] => {
        const points: Point[] = [];

        // Get the coordinate data
        const coords = axisCoordToArray(serie.data);

        if (coords.x.length < 2) return [];

        // Sort by x-coordinate
        const sortedIndices = coords.x.map((_, i) => i).sort((a, b) => coords.x[a] - coords.x[b]);
        const sortedX = sortedIndices.map(i => coords.x[i]);
        const sortedY = sortedIndices.map(i => coords.y[i]);

        // Generate dense points along the curve (similar to ReconstructedChart rendering)
        const densityFactor = 10; // Increase for more points

        for (let i = 0; i < sortedX.length - 1; i++) {
            const x1 = sortedX[i];
            const y1 = sortedY[i];
            const x2 = sortedX[i + 1];
            const y2 = sortedY[i + 1];

            // Add the current point
            points.push({ x: x1, y: y1 });

            // Add interpolated points between current and next
            for (let j = 1; j < densityFactor; j++) {
                const t = j / densityFactor;
                const interpX = x1 + t * (x2 - x1);
                const interpY = y1 + t * (y2 - y1);
                points.push({ x: interpX, y: interpY });
            }
        }

        // Add the last point
        if (sortedX.length > 0) {
            points.push({ x: sortedX[sortedX.length - 1], y: sortedY[sortedY.length - 1] });
        }

        return points;
    };

    // Get axis information from the chart state
    const getAxisInfo = () => {
        const xAxis = state.dataTable.axisX;
        const yAxis = state.dataTable.axisY;

        // Get all data points to determine actual ranges
        let allXValues: number[] = [];
        let allYValues: number[] = [];

        for (const serie of state.dataTable.series) {
            if (!(serie instanceof SubChartElement)) {
                const coords = axisCoordToArray(serie.data);
                allXValues = allXValues.concat(coords.x);
                allYValues = allYValues.concat(coords.y);
            }
        }

        if (allXValues.length === 0) {
            return {
                x_name: 'X Axis',
                y_name: 'Y Axis',
                x_range: [0, 1] as [number, number],
                y_range: [0, 1] as [number, number]
            };
        }

        // Calculate ranges with some padding
        const xMin = Math.min(...allXValues);
        const xMax = Math.max(...allXValues);
        const yMin = Math.min(...allYValues);
        const yMax = Math.max(...allYValues);

        const xPadding = (xMax - xMin) * 0.05;
        const yPadding = (yMax - yMin) * 0.05;

        return {
            x_name: xAxis.name || 'X Axis',
            y_name: yAxis.name || 'Y Axis',
            x_range: [xMin - xPadding, xMax + xPadding] as [number, number],
            y_range: [yMin - yPadding, yMax + yPadding] as [number, number]
        };
    };

    // Extract curves with option for traced data
    const extractCurvesFromState = (): CurveData[] => {
        const curves: CurveData[] = [];

        if (state.dataTable && state.dataTable.series) {
            for (let seriesIndex = 0; seriesIndex < state.dataTable.series.length; seriesIndex++) {
                const serie = state.dataTable.series[seriesIndex];

                if (!(serie instanceof SubChartElement)) {
                    let points: Point[];
                    let isTraced = false;

                    if (useTracedData) {
                        // Use dense traced data points
                        points = getTracedDataPoints(serie, seriesIndex);
                        isTraced = true;
                    } else {
                        // Use original extracted points
                        const coords = axisCoordToArray(serie.data);
                        points = coords.x.map((x, i) => ({ x: x, y: coords.y[i] }));
                    }

                    if (points.length >= 2) {
                        let color = serie.getMainColor();
                        if (color === "#ffffff") color = "#CCCCCC";

                        curves.push({
                            points,
                            seriesIndex,
                            color,
                            label: serie.name || `Series ${seriesIndex + 1}`,
                            isTraced
                        });
                    }
                }
            }
        }

        return curves;
    };
    const evaluateIndividualCurve = async (seriesIndex: number, xValue: string) => {
        const x_new = parseFloat(xValue);
        if (isNaN(x_new)) {
            setCurveEvaluations(prev => new Map(prev).set(seriesIndex, { x: xValue, y: '', error: 'Invalid number' }));
            return;
        }

        try {
            const response = await axios.post('http://localhost:8000/evaluate-curve', {
                seriesIndex: seriesIndex,
                x_value: x_new
            });

            setCurveEvaluations(prev => new Map(prev).set(seriesIndex, {
                x: xValue,
                y: response.data.y_value.toFixed(6),
                error: response.data.warning || ''
            }));

        } catch (err: any) {
            setCurveEvaluations(prev => new Map(prev).set(seriesIndex, {
                x: xValue,
                y: '',
                error: err.response?.data?.detail || 'Evaluation failed'
            }));
        }
    };
    const generateCurveFunction = async (curveData: CurveData) => {
        try {
            setLoading(prev => new Set(prev).add(curveData.seriesIndex));

            const axisInfo = getAxisInfo();

            const response = await axios.post<CurveResult>('http://localhost:8000/fit-single-curve', {
                points: curveData.points,
                seriesIndex: curveData.seriesIndex,
                color: curveData.color,
                label: curveData.label,
                axis_info: axisInfo,
                fitting_method: fittingMethod,
                is_traced: curveData.isTraced
            });

            setCurveResults(prev => new Map(prev).set(curveData.seriesIndex, response.data));
        } catch (err: any) {
            console.error(`Curve function generation failed for series ${curveData.seriesIndex}:`, err);
        } finally {
            setLoading(prev => {
                const newSet = new Set(prev);
                newSet.delete(curveData.seriesIndex);
                return newSet;
            });
        }
    };

    // Auto-generate functions when curves are extracted
    useEffect(() => {
        if (autoGenerate) {
            const curves = extractCurvesFromState();
            curves.forEach(curve => {
                // Regenerate if method changed or no result exists
                if (!curveResults.has(curve.seriesIndex) ||
                    (curveResults.get(curve.seriesIndex)?.fitting_method !== fittingMethod)) {
                    generateCurveFunction(curve);
                }
            });
        }
    }, [state.dataTable.series, autoGenerate, fittingMethod, useTracedData]);

    const downloadPythonFunction = (result: CurveResult) => {
        const blob = new Blob([result.python_function], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `curve_${result.axis_info.x_name}_vs_${result.axis_info.y_name}_series_${result.seriesIndex + 1}_${result.fitting_method}_${result.point_count}pts.py`;
        a.click();
        URL.revokeObjectURL(url);
    };

    const curves = extractCurvesFromState();
    const axisInfo = getAxisInfo();

    return (
        <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            {/* Chart Axis Information
            <div style={{
                marginBottom: '15px',
                padding: '10px',
                backgroundColor: '#e3f2fd',
                borderRadius: '5px'
            }}>
                <h4>Chart Axis Information:</h4>
                <div><strong>X-Axis:</strong> {axisInfo.x_name} (Range: {axisInfo.x_range[0].toFixed(2)} to {axisInfo.x_range[1].toFixed(2)})</div>
                <div><strong>Y-Axis:</strong> {axisInfo.y_name} (Range: {axisInfo.y_range[0].toFixed(2)} to {axisInfo.y_range[1].toFixed(2)})</div>
            </div> */}

            {/* Control Panel */}
            <div style={{
                marginBottom: '15px',
                padding: '10px',
                backgroundColor: '#f8f9fa',
                borderRadius: '5px'
            }}>
                <div style={{ marginBottom: '10px' }}>
                    <strong>Extracted Curves: </strong>{curves.length}
                    {curves.length > 0 && (
                        <span style={{ marginLeft: '10px', color: '#666' }}>
                            (Avg points per curve: {Math.round(curves.reduce((sum, c) => sum + c.points.length, 0) / curves.length)})
                        </span>
                    )}
                </div>

                <div style={{ display: 'flex', alignItems: 'center', marginBottom: '10px', gap: '15px', flexWrap: 'wrap' }}>
                    <label>
                        <input
                            type="checkbox"
                            checked={autoGenerate}
                            onChange={(e) => setAutoGenerate(e.target.checked)}
                            style={{ marginRight: '5px' }}
                        />
                        Auto-generate functions
                    </label>

                    <label>
                        <input
                            type="checkbox"
                            checked={useTracedData}
                            onChange={(e) => setUseTracedData(e.target.checked)}
                            style={{ marginRight: '5px' }}
                        />
                        Use traced curve data (more points)
                    </label>

                    <label>
                        Fitting Method:
                        <select
                            value={fittingMethod}
                            onChange={(e) => setFittingMethod(e.target.value)}
                            style={{ marginLeft: '5px', padding: '4px' }}
                        >
                            <option value="linear">Linear Interpolation</option>
                            <option value="smart_lagrange">Smart Lagrange</option>
                            <option value="piecewise_lagrange">Piecewise Lagrange</option>
                            <option value="cubic_spline">Cubic Spline</option>
                            <option value="smoothing_spline">Smoothing Spline</option>
                            <option value="polynomial">Standard Polynomial</option>
                        </select>
                    </label>
                </div>
            </div>

            {/* Curve List with Results */}
            <div style={{ marginBottom: '15px' }}>
                <h4>Curve Functions:</h4>
                {curves.map((curve) => {
                    const result = curveResults.get(curve.seriesIndex);
                    const isLoading = loading.has(curve.seriesIndex);

                    return (
                        <div
                            key={curve.seriesIndex}
                            style={{
                                marginBottom: '20px',
                                padding: '15px',
                                border: selectedCurve === curve.seriesIndex ? '2px solid #ffc52e' : '1px solid #ddd',
                                borderRadius: '8px',
                                backgroundColor: selectedCurve === curve.seriesIndex ? '#f8f9ff' : '#fff',
                                cursor: 'pointer'
                            }}
                            onClick={() => setSelectedCurve(curve.seriesIndex)}
                        >
                            {/* Curve Header */}
                            <div style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center',
                                marginBottom: '10px'
                            }}>
                                <div style={{ display: 'flex', alignItems: 'center' }}>
                                    <div
                                        style={{
                                            width: '16px',
                                            height: '16px',
                                            backgroundColor: curve.color,
                                            marginRight: '10px',
                                            borderRadius: '3px'
                                        }}
                                    />
                                    <strong>{curve.label}</strong>
                                    <span style={{ marginLeft: '10px', color: '#666' }}>
                                        ({curve.points.length} {curve.isTraced ? 'traced' : 'extracted'} points)
                                    </span>
                                </div>

                                <div>
                                    {isLoading && (
                                        <span style={{ color: '#ffc107', marginRight: '10px' }}>⏳ Generating...</span>
                                    )}
                                    {result && (
                                        <span style={{ color: '#ffc52e', marginRight: '10px' }}>
                                            ✓ {result.fitting_method.replace('_', ' ')} (R²: {result.r_squared.toFixed(4)})
                                        </span>
                                    )}

                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            generateCurveFunction(curve);
                                        }}
                                        disabled={isLoading}
                                        style={{
                                            padding: '5px 10px',
                                            backgroundColor: isLoading ? '#ccc' : '#ffc52e',
                                            color: '#181818',
                                            border: 'none',
                                            borderRadius: '3px',
                                            cursor: isLoading ? 'not-allowed' : 'pointer',
                                            fontSize: '12px'
                                        }}
                                    >
                                        {isLoading ? 'Generating...' : 'Generate'}
                                    </button>
                                </div>
                            </div>

                            {/* Results Display */}
                            {result && (
                                <div>
                                    {/* Mathematical Function */}
                                    <div style={{ marginBottom: '15px' }}>
                                        <h5>Mathematical Function ({result.fitting_method.replace('_', ' ')}, {result.point_count} points, R² = {result.r_squared.toFixed(4)}):</h5>
                                        <div style={{
                                            backgroundColor: '#e9ecef',
                                            padding: '8px',
                                            borderRadius: '4px',
                                            fontFamily: 'monospace',
                                            fontSize: '13px',
                                            wordBreak: 'break-all'
                                        }}>
                                            {result.equation}
                                        </div>
                                    </div>

                                    {/* Plot Display */}
                                    <div style={{ marginBottom: '15px' }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                            <h5>Generated Plot (Using Chart Axis Scales):</h5>
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    downloadPythonFunction(result);
                                                }}
                                                style={{
                                                    padding: '4px 8px',
                                                    backgroundColor: '#ffc52e',
                                                    color: '#181818',
                                                    border: 'none',
                                                    borderRadius: '3px',
                                                    cursor: 'pointer',
                                                    fontSize: '11px'
                                                }}
                                            >
                                                Download Python Code
                                            </button>
                                        </div>
                                        <div style={{
                                            border: '1px solid #dee2e6',
                                            borderRadius: '6px',
                                            padding: '8px',
                                            backgroundColor: 'white'
                                        }}>
                                            <img
                                                src={result.plot_image}
                                                alt={`Curve for ${curve.label}`}
                                                style={{
                                                    width: '100%',
                                                    height: 'auto',
                                                    borderRadius: '4px'
                                                }}
                                            />
                                        </div>
                                    </div>

                                    {/* Python Function Preview */}
                                    <div>
                                        <h5>Python Function Preview:</h5>
                                        <pre style={{
                                            backgroundColor: '#f8f9fa',
                                            padding: '10px',
                                            borderRadius: '4px',
                                            fontSize: '11px',
                                            overflow: 'auto',
                                            maxHeight: '150px',
                                            border: '1px solid #dee2e6'
                                        }}>
                                            {result.python_function.split('\n').slice(0, 15).join('\n')}
                                            {result.python_function.split('\n').length > 15 && '\n... (truncated)'}
                                        </pre>
                                    </div>
                                    <div style={{
                                        backgroundColor: '#e2e8f0',
                                        padding: '15px',
                                        borderRadius: '8px',
                                    }}>


                                        <div style={{ color: '#a0aec0', marginBottom: '10px', fontSize: '13px' }}>
                                            {`Use the interpolator for ${curve.label}:`}
                                        </div>

                                        <div style={{
                                            fontFamily: 'monospace',
                                            backgroundColor: '#e2e8f0',
                                            color: '#1a202c',
                                            padding: '12px',
                                            borderRadius: '6px',
                                            marginBottom: '15px'
                                        }}>
                                            <div style={{ marginBottom: '8px' }}>
                                                X Value = <input
                                                    type="number"
                                                    value={curveEvaluations.get(curve.seriesIndex)?.x || ''}
                                                    onChange={(e) => {
                                                        const newValue = e.target.value;
                                                        setCurveEvaluations(prev => new Map(prev).set(curve.seriesIndex, {
                                                            x: newValue,
                                                            y: prev.get(curve.seriesIndex)?.y || '',
                                                            error: ''
                                                        }));
                                                    }}
                                                    placeholder="2.5"
                                                    style={{
                                                        width: '80px',
                                                        backgroundColor: '#e2e8f0',
                                                        color: '#1a202c',
                                                        border: '1px solid #718096',
                                                        padding: '4px 8px',
                                                        borderRadius: '4px',
                                                        fontFamily: 'monospace'
                                                    }}
                                                    onKeyPress={(e) => {
                                                        if (e.key === 'Enter') {
                                                            evaluateIndividualCurve(curve.seriesIndex, curveEvaluations.get(curve.seriesIndex)?.x || '');
                                                        }
                                                    }}
                                                />
                                            </div>
                                            <div>
                                                Y Value = <span style={{ color: '#181818' }}>
                                                    {curveEvaluations.get(curve.seriesIndex)?.y || '?'}
                                                </span>
                                            </div>
                                        </div>

                                        <button
                                            onClick={() => evaluateIndividualCurve(curve.seriesIndex, curveEvaluations.get(curve.seriesIndex)?.x || '')}
                                            disabled={!curveEvaluations.get(curve.seriesIndex)?.x}
                                            style={{
                                                padding: '6px 12px',
                                                backgroundColor: '#ffc52e',
                                                color: '#1a202c',
                                                border: 'none',
                                                borderRadius: '4px',
                                                cursor: curveEvaluations.get(curve.seriesIndex)?.x ? 'pointer' : 'not-allowed',
                                                fontFamily: 'monospace',
                                                fontSize: '13px',
                                                fontWeight: 'bold'
                                            }}
                                        >
                                            Execute
                                        </button>

                                        {curveEvaluations.get(curve.seriesIndex)?.error && (
                                            <div style={{
                                                color: '#fc8181',
                                                fontSize: '13px',
                                                marginTop: '8px',
                                                fontFamily: 'monospace'
                                            }}>
                                                Error: {curveEvaluations.get(curve.seriesIndex)?.error}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>

            {/* No Data Message */}
            {curves.length === 0 && (
                <div style={{
                    textAlign: 'center',
                    color: '#6c757d',
                    marginTop: '50px',
                    padding: '20px'
                }}>
                    <h4>No Curves Extracted</h4>
                    <p>Extract data series from the chart to automatically generate curve functions.</p>
                </div>
            )}
        </div>
    );
};

export default CurveDisplay;

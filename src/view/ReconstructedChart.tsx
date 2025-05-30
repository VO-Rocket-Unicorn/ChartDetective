// Import necessary React and chart-related libraries
import React from 'react';
import { ChartElementType } from '../datastructure/chartelements/ChartElement';
import { ChartExtractorState } from './ChartExtractor';
import Plot from 'react-plotly.js';
import AxisCoordinate2D from '../datastructure/AxisCoordinate2D';
import SubChartElement from '../datastructure/chartelements/SubChartElement';

// Define the props interface - what data this component expects from its parent
interface ReconstructedChartProps {
    state: ChartExtractorState;
    setState: any;
}

// Define structure for coordinate arrays with X and Y number arrays
interface CoordArray {
    x: number[],
    y: number[]
}

// Define the component's internal state structure for tracing functionality
interface TracingState {
    selectedSerieIndex: number;        // Which curve is currently selected for tracing
    tracingProgress: number;           // Progress from 0 to 100 percent
    isTracing: boolean;               // Whether tracing is currently active
    currentCoordinate: { x: number | null, y: number | null }; // Current X,Y position
}

// Main component class that extends React.Component with typed props and state
export default class ReconstructedChart extends React.Component<ReconstructedChartProps, TracingState> {
    // Static arrays defining visual styles for different chart elements
    static dashLineStyle = ['solid', 'dash', 'dot', 'longdash', 'dashdot', 'longdashdot'];
    static markerSymbols = ['circle', 'cross', 'square', 'diamond', 'triangle', 'pentagon', 'hexagon', 'octagon', 'star'];
    static barPatternStyle = ['', '/', '\\', 'x', '-', '|', '+', '.'];

    // Constructor initializes the component with default state values
    constructor(props: ReconstructedChartProps) {
        super(props); // Call parent constructor
        this.state = {
            selectedSerieIndex: 0,                    // Start with first series
            tracingProgress: 0,                       // Start at 0% progress
            isTracing: false,                         // Not tracing initially
            currentCoordinate: { x: null, y: null }   // No current position yet
        };
    }

    // Convert complex coordinate objects into simple X and Y arrays for plotting
    axisCoordToArray(coord: AxisCoordinate2D[]): CoordArray {
        const x = []; // Array to store X coordinates
        const y = []; // Array to store Y coordinates

        // Loop through each coordinate point and extract values
        for (const pt of coord) {
            x.push(pt.x.value); // Add X value to array
            y.push(pt.y.value); // Add Y value to array
        }

        return { x: x, y: y }; // Return object with both arrays
    }

    // Filter and return only series that can be traced (lines and scatter plots)
    getTraceableSeries() {
        return this.props.state.dataTable.series.filter(serie =>
            !(serie instanceof SubChartElement) &&                                    // Exclude sub-elements
            (serie.is(ChartElementType.LINE) || serie.is(ChartElementType.SCATTER))   // Only lines and scatter plots
        );
    }

    // Calculate current coordinate with smooth interpolation between data points
    getCurrentCoordinate(coords: CoordArray, progress: number): { x: number | null, y: number | null } {
        if (progress === 0 || coords.x.length === 0) {
            return { x: null, y: null };
        }

        // Use sorted coordinates for tracing
        const tracingCoords = this.getTracingCoordinates(coords);

        const totalPoints = tracingCoords.x.length;
        const exactPosition = (progress / 100) * (totalPoints - 1);
        const index = Math.floor(exactPosition);
        const fraction = exactPosition - index;

        if (index >= totalPoints - 1) {
            return {
                x: tracingCoords.x[totalPoints - 1],
                y: tracingCoords.y[totalPoints - 1]
            };
        }

        // Interpolate using sorted tracing data
        const x_interp = tracingCoords.x[index] + fraction * (tracingCoords.x[index + 1] - tracingCoords.x[index]);
        const y_interp = tracingCoords.y[index] + fraction * (tracingCoords.y[index + 1] - tracingCoords.y[index]);

        return { x: x_interp, y: y_interp };
    }
    // Create sorted coordinates specifically for tracing - SEPARATE FROM ORIGINAL DATA
    getTracingCoordinates(coords: CoordArray): CoordArray {
        if (coords.x.length === 0) {
            return { x: [], y: [] };
        }

        // Create array of indices and sort by X values
        const indices = Array.from({ length: coords.x.length }, (_, i) => i);
        indices.sort((a, b) => coords.x[a] - coords.x[b]);

        const sortedX = indices.map(i => coords.x[i]);
        const sortedY = indices.map(i => coords.y[i]);

        return { x: sortedX, y: sortedY };
    }
    // Create the traced portion of the curve with interpolated endpoint
    createTracedCurve(coords: CoordArray, progress: number) {
        if (progress === 0 || coords.x.length === 0) {
            return { x: [], y: [] };
        }

        // Use sorted coordinates for tracing
        const tracingCoords = this.getTracingCoordinates(coords);

        const totalPoints = tracingCoords.x.length;
        const exactPosition = (progress / 100) * (totalPoints - 1);
        const index = Math.floor(exactPosition);
        const fraction = exactPosition - index;

        // Get all complete points up to the current index from sorted data
        const tracedX = tracingCoords.x.slice(0, index + 1);
        const tracedY = tracingCoords.y.slice(0, index + 1);

        // Add interpolated point if we're between two points
        if (fraction > 0 && index < totalPoints - 1) {
            const x_interp = tracingCoords.x[index] + fraction * (tracingCoords.x[index + 1] - tracingCoords.x[index]);
            const y_interp = tracingCoords.y[index] + fraction * (tracingCoords.y[index + 1] - tracingCoords.y[index]);

            tracedX.push(x_interp);
            tracedY.push(y_interp);
        }

        return { x: tracedX, y: tracedY };
    }

    // Handle dropdown selection change - reset tracing when switching curves
    handleSerieChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
        const newIndex = parseInt(event.target.value); // Get selected series index
        this.setState({
            selectedSerieIndex: newIndex,                    // Update selected series
            tracingProgress: 0,                              // Reset progress to start
            currentCoordinate: { x: null, y: null }         // Clear current coordinates
        });
    }

    // Handle slider movement - update tracing progress and calculate new position
    handleProgressChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const progress = parseFloat(event.target.value); // Get slider value as number

        // Get the currently selected series data
        const selectedSerie = this.props.state.dataTable.series[this.state.selectedSerieIndex];
        if (selectedSerie && !(selectedSerie instanceof SubChartElement)) {
            const coords = this.axisCoordToArray(selectedSerie.data);        // Convert to coordinate arrays
            const currentCoord = this.getCurrentCoordinate(coords, progress); // Calculate current position

            // Update state with new progress and coordinates
            this.setState({
                tracingProgress: progress,
                currentCoordinate: currentCoord
            });
        } else {
            // If no valid series, just update progress
            this.setState({
                tracingProgress: progress
            });
        }
    }

    // Toggle tracing on/off (currently unused but available for future features)
    toggleTracing = () => {
        this.setState(prevState => ({
            isTracing: !prevState.isTracing
        }));
    }

    // Reset tracing to initial state
    resetTracing = () => {
        this.setState({
            tracingProgress: 0,                              // Reset to 0% progress
            currentCoordinate: { x: null, y: null }         // Clear coordinates
        });
    }

    // Format coordinate values for display - show integers as whole numbers, decimals with 4 places
    formatCoordinate = (value: number | null): string => {
        if (value === null) return 'N/A';                                          // Show N/A for null values
        return Number.isInteger(value) ? value.toString() : value.toFixed(4);      // Format based on type
    }

    // Main render method - creates the visual interface and chart
    render(): JSX.Element {
        const traces = [];                          // Array to hold all chart traces
        const styleCounter = {};                    // Track style usage for variety
        const traceableSeries = this.getTraceableSeries(); // Get series that can be traced
        const processedCoordinates = new Map();     // Store processed coordinates for reuse

        // Loop through all series to create chart traces
        for (let serieIndex = 0; serieIndex < this.props.state.dataTable.series.length; serieIndex++) {
            const serie = this.props.state.dataTable.series[serieIndex];

            // Only process main chart elements, not sub-elements
            if (!(serie instanceof SubChartElement)) {
                // Convert series data to coordinate arrays using the same method for consistency
                const coords = this.axisCoordToArray(serie.data);
                processedCoordinates.set(serieIndex, coords); // Store for potential reuse

                // Handle error bars if they exist
                let errorBar = null;
                if (serie.hasErrorBars()) {
                    const upperValue = this.axisCoordToArray(serie.upperErrorBar.data).y;  // Upper error values
                    const lowerValue = this.axisCoordToArray(serie.lowerErrorBar.data).y;  // Lower error values
                    const array = upperValue.map((v, i) => Math.abs(v - coords.y[i]));     // Calculate upper error
                    const arrayMinus = lowerValue.map((v, i) => Math.abs(v - coords.y[i])); // Calculate lower error

                    errorBar = {
                        type: 'data',
                        symmetric: false,
                        array: array,
                        arrayminus: arrayMinus
                    }
                }

                // Get series color and convert white to gray for visibility
                let color = serie.getMainColor();
                if (color === "#ffffff") color = "#CCCCCC";

                // Check if this series is currently selected for tracing
                const isSelectedForTracing = serieIndex === this.state.selectedSerieIndex &&
                    (serie.is(ChartElementType.LINE) || serie.is(ChartElementType.SCATTER));

                // Create the base trace object with the same coordinates used for plotting
                const baseTrace = {
                    x: coords.x,                                                    // X coordinates
                    y: coords.y,                                                    // Y coordinates
                    name: serie.name,                                               // Series name for legend
                    error_y: errorBar,                                              // Error bars if any
                    line: {
                        color: isSelectedForTracing ? `${color}40` : color,         // Semi-transparent if selected
                        width: isSelectedForTracing ? 2 : undefined                 // Thinner if selected
                    },
                    marker: {
                        color: isSelectedForTracing ? `${color}40` : color          // Semi-transparent if selected
                    },
                    showlegend: !isSelectedForTracing || this.state.tracingProgress === 0 // Hide legend when tracing
                };

                // Create style hash for consistent styling across similar series
                const styleHash = ChartElementType.LINE + "." + color;
                if (!(styleHash in styleCounter)) styleCounter[styleHash] = 0;
                const styleId = styleCounter[styleHash];

                // Configure trace based on chart element type
                if (serie.is(ChartElementType.LINE)) {
                    baseTrace['type'] = "scatter";                                                                      // Plotly scatter type
                    baseTrace['mode'] = "lines";                                                                        // Lines only mode
                    baseTrace.line['dash'] = ReconstructedChart.dashLineStyle[styleId % ReconstructedChart.dashLineStyle.length]; // Cycle through dash styles
                }

                if (serie.is(ChartElementType.SCATTER)) {
                    baseTrace['type'] = "scatter"                                                                       // Plotly scatter type
                    baseTrace['mode'] = "markers"                                                                       // Markers only mode
                    baseTrace.marker['symbol'] = ReconstructedChart.markerSymbols[styleId % ReconstructedChart.markerSymbols.length]; // Cycle through symbols
                }

                if (serie.is(ChartElementType.BAR)) {
                    baseTrace['type'] = "bar"                                                                           // Plotly bar type
                    baseTrace.marker['pattern'] = {
                        shape: ReconstructedChart.barPatternStyle[styleId % ReconstructedChart.barPatternStyle.length] // Cycle through patterns
                    };
                }

                if (serie.is(ChartElementType.BOX_PLOT)) {
                    baseTrace['type'] = "box";                                                                          // Plotly box type
                    baseTrace['median'] = coords.y;                                                                     // Median values
                    baseTrace['y'] = undefined;                                                                         // Clear Y for box plot
                    // Set box plot components if they exist
                    if (serie.linkedElements[5] !== null) baseTrace['lowerfence'] = this.axisCoordToArray(serie.linkedElements[5].data).y;
                    if (serie.linkedElements[4] !== null) baseTrace['upperfence'] = this.axisCoordToArray(serie.linkedElements[4].data).y;
                    if (serie.linkedElements[2] !== null) baseTrace['q1'] = this.axisCoordToArray(serie.linkedElements[2].data).y;
                    if (serie.linkedElements[3] !== null) baseTrace['q3'] = this.axisCoordToArray(serie.linkedElements[3].data).y;
                }

                traces.push(baseTrace); // Add base trace to traces array

                // Add traced curve if this series is selected and has progress
                if (isSelectedForTracing && this.state.tracingProgress > 0) {
                    const tracedCoords = this.createTracedCurve(coords, this.state.tracingProgress); // Get traced portion

                    // Create traced line trace - thick solid line in original color
                    const tracedTrace = {
                        x: tracedCoords.x,                          // Traced X coordinates
                        y: tracedCoords.y,                          // Traced Y coordinates
                        name: `${serie.name} (Traced)`,            // Name with traced indicator
                        type: "scatter",                            // Scatter type
                        mode: "lines",                              // Lines mode
                        line: {
                            color: color,                           // Original color (not transparent)
                            width: 4,                               // Thick 4px line
                            dash: 'solid'                           // Always solid
                        },
                        showlegend: true                            // Show in legend
                    };

                    traces.push(tracedTrace); // Add traced trace to array

                    // Add position marker at current location
                    const currentCoord = this.getCurrentCoordinate(coords, this.state.tracingProgress);
                    if (currentCoord.x !== null && currentCoord.y !== null) {
                        const currentPosTrace = {
                            x: [currentCoord.x],                    // Single X coordinate
                            y: [currentCoord.y],                    // Single Y coordinate
                            name: 'Current Position',               // Marker name
                            type: "scatter",                        // Scatter type
                            mode: "markers",                        // Markers mode
                            marker: {
                                color: color,                       // Same color as series
                                size: 12,                           // Large marker
                                symbol: 'circle',                  // Circle symbol
                                line: {
                                    color: 'white',                 // White border
                                    width: 2                        // Border width
                                }
                            },
                            showlegend: false,                      // Don't show in legend
                            hovertemplate: `<b>Interpolated Position</b><br>X: ${this.formatCoordinate(currentCoord.x)}<br>Y: ${this.formatCoordinate(currentCoord.y)}<extra></extra>` // Hover info
                        };

                        traces.push(currentPosTrace); // Add position marker to traces
                    }
                }

                styleCounter[styleHash] += 1; // Increment style counter for variety
            }
        }

        // Return the complete JSX structure for the component
        return (
            <div style={{ width: "100%", height: "100%" }}> {/* Main container */}
                {/* Control Panel */}
                <div style={{
                    padding: "10px",                    // Internal spacing
                    backgroundColor: "#f5f5f5",        // Light gray background
                    borderRadius: "5px",               // Rounded corners
                    marginBottom: "10px",              // Space below
                    display: "flex",                   // Flexbox layout
                    alignItems: "center",              // Center items vertically
                    gap: "15px",                       // Space between items
                    flexWrap: "wrap"                   // Wrap on small screens
                }}>
                    {/* Series Selection Dropdown */}
                    <div>
                        <label htmlFor="serie-select" style={{ marginRight: "5px", fontWeight: "bold" }}>
                            Select Curve:
                        </label>
                        <select
                            id="serie-select"
                            value={this.state.selectedSerieIndex}      // Controlled by state
                            onChange={this.handleSerieChange}          // Handle selection changes
                            style={{ padding: "5px", borderRadius: "3px" }}
                        >
                            {/* Create option for each traceable series */}
                            {traceableSeries.map((serie, index) => (
                                <option key={index} value={this.props.state.dataTable.series.indexOf(serie)}>
                                    {serie.name || `Series ${index + 1}`}  {/* Use series name or default */}
                                </option>
                            ))}
                        </select>
                    </div>

                    {/* Progress Slider Section */}
                    <div style={{ display: "flex", alignItems: "center", gap: "10px", flex: 1, minWidth: "200px" }}>
                        <label htmlFor="progress-slider" style={{ fontWeight: "bold" }}>
                            Trace Progress:
                        </label>
                        <input
                            id="progress-slider"
                            type="range"                                // Range slider input
                            min="0"                                     // Minimum value
                            max="100"                                   // Maximum value
                            step="0.5"                                  // Step size for precision
                            value={this.state.tracingProgress}         // Controlled by state
                            onChange={this.handleProgressChange}       // Handle slider changes
                            style={{ flex: 1 }}                       // Take remaining space
                        />
                        <span style={{ minWidth: "50px", fontWeight: "bold" }}>
                            {this.state.tracingProgress.toFixed(1)}%   {/* Display current percentage */}
                        </span>
                    </div>

                    {/* Coordinate Display Panel */}
                    <div style={{
                        backgroundColor: "#ffffff",        // White background
                        padding: "8px 12px",              // Internal padding
                        borderRadius: "5px",              // Rounded corners
                        border: "2px solid #ffc52e",      // Blue border
                        minWidth: "220px"                 // Minimum width
                    }}>
                        <div style={{ fontWeight: "bold", fontSize: "12px", color: "#666", marginBottom: "4px" }}>
                            Current X and Y values:        {/* Header text */}
                        </div>
                        <div style={{ fontFamily: "monospace", fontSize: "14px", marginBottom: "4px" }}>
                            <span style={{ fontWeight: "bold" }}>X:</span> {this.formatCoordinate(this.state.currentCoordinate.x)} |
                            <span style={{ fontWeight: "bold" }}> Y:</span> {this.formatCoordinate(this.state.currentCoordinate.y)}
                        </div>

                        {/* Show interpolation status when actively tracing */}
                        {this.state.currentCoordinate.x !== null && this.state.tracingProgress > 0 && (
                            <div style={{ fontSize: "11px", color: "#666", fontStyle: "italic" }}>
                                {(() => {
                                    const selectedSerie = this.props.state.dataTable.series[this.state.selectedSerieIndex];
                                    if (selectedSerie && !(selectedSerie instanceof SubChartElement)) {
                                        const coords = this.axisCoordToArray(selectedSerie.data);           // Get coordinates
                                        const totalPoints = coords.x.length;                               // Total points
                                        const exactPosition = (this.state.tracingProgress / 100) * (totalPoints - 1); // Exact position
                                        const index = Math.floor(exactPosition);                           // Integer part
                                        const fraction = exactPosition - index;                            // Decimal part

                                        if (fraction === 0) {
                                            return `At point ${index + 1} of ${totalPoints}`;              // Exactly at a point
                                        } else if (index < totalPoints - 1) {
                                            return `Between points ${index + 1} and ${index + 2} (${(fraction * 100).toFixed(1)}%)`; // Between points
                                        }
                                    }
                                    return '';
                                })()}
                            </div>
                        )}
                    </div>

                    {/* Reset Button */}
                    <button
                        onClick={this.resetTracing}                // Handle reset click
                        style={{
                            padding: "5px 10px",                   // Button padding
                            backgroundColor: "#ffc52e",            // Blue background
                            color: "white",                         // White text
                            border: "none",                         // No border
                            borderRadius: "3px",                   // Rounded corners
                            cursor: "pointer"                       // Pointer cursor
                        }}
                    >
                        Reset
                    </button>
                </div>

                {/* Chart Container */}
                <div style={{ height: "calc(100% - 80px)" }}> {/* Take remaining height minus control panel */}
                    <Plot
                        style={{ width: "100%", height: "100%" }}      // Full size
                        useResizeHandler={true}                         // Handle window resize
                        data={traces}                                   // Chart data traces
                        layout={{
                            autosize: true,                             // Auto-resize
                            title: this.props.state.dataTable.name,    // Chart title
                            xaxis: {
                                type: this.props.state.dataTable.axisX.isCategorical() ? "category" : "linear", // X-axis type
                                title: this.props.state.dataTable.axisX.name                                    // X-axis title
                            },
                            yaxis: {
                                type: this.props.state.dataTable.axisY.isCategorical() ? "category" : "linear", // Y-axis type
                                title: this.props.state.dataTable.axisY.name                                    // Y-axis title
                            },
                            boxmode: 'group'                            // Group box plots
                        }}
                    />
                </div>
            </div>
        );
    }
}

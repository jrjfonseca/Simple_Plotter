import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import re
import os
import sys
from scipy.interpolate import splrep, splev
from scipy.signal import savgol_filter

def load_data(file_path: Path, total_mass: float = None, am_percentage: float = None, collector_mass: float = None) -> pd.DataFrame:
    """Load electrochemistry data from CSV/Excel file and process it for analysis.
    
    Args:
        file_path: Path to the data file
        total_mass: Total electrode mass in mg
        am_percentage: Active material percentage (0-100)
        collector_mass: Mass of the current collector in mg
        
    Returns:
        DataFrame containing processed electrochemistry data with standardized column names
    """
    # Load file based on extension
    if file_path.suffix.lower() == '.csv':
        raw_df = pd.read_csv(file_path)
    else:  # Excel files
        # For Excel files, we need to determine which sheets to use
        xl = pd.ExcelFile(file_path)
        
        data_sheet_idx = None
        stats_sheet_idx = None
        
        # Look for sheets with relevant keywords
        for i, sheet_name in enumerate(xl.sheet_names):
            if re.search(r'(channel|data|detail)', sheet_name.lower()):
                data_sheet_idx = i
            elif re.search(r'(stat|summar)', sheet_name.lower()):
                stats_sheet_idx = i
        
        # If not found by name, use heuristics (assume data comes before stats)
        if data_sheet_idx is None and stats_sheet_idx is None:
            # Check content of first few sheets to find data and stats sheets
            for i in range(min(5, len(xl.sheet_names))):
                sample = pd.read_excel(file_path, sheet_name=xl.sheet_names[i], nrows=5)
                cols = [str(col).lower() for col in sample.columns]
                col_str = ' '.join(cols)
                
                # Look for typical column names
                if any(re.search(term, col_str) for term in ['cycle', 'voltage', 'current']):
                    if data_sheet_idx is None:
                        data_sheet_idx = i
                    elif stats_sheet_idx is None and data_sheet_idx != i:
                        stats_sheet_idx = i
                        break
        
        # Default to first and second sheets if still not found
        if data_sheet_idx is None:
            data_sheet_idx = 0
        if stats_sheet_idx is None:
            stats_sheet_idx = min(data_sheet_idx + 1, len(xl.sheet_names) - 1)
        
        # Read the data
        raw_df = pd.read_excel(file_path, sheet_name=xl.sheet_names[data_sheet_idx])
        stats_df = pd.read_excel(file_path, sheet_name=xl.sheet_names[stats_sheet_idx])
    
    # Standardize column names
    std_df = standardize_columns(raw_df)
    
    # Process data if mass parameters are provided
    if total_mass is not None and am_percentage is not None and collector_mass is not None:
        std_df = process_capacity_data(std_df, total_mass, am_percentage, collector_mass)
    
    return std_df

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to work with plotting functions.
    
    Args:
        df: Raw dataframe
        
    Returns:
        DataFrame with standardized column names
    """
    # Create a copy to avoid modifying the original
    std_df = df.copy()
    
    # Create lowercase column string for regex matching
    cols_lower = [col.lower() for col in std_df.columns]
    
    # Map for standardized column names
    col_mapping = {}
    
    # Cycle index
    cycle_pattern = r'cycle[_\s]*index|cycle[_\s]*idx'
    for i, col in enumerate(cols_lower):
        if re.search(cycle_pattern, col):
            col_mapping[std_df.columns[i]] = 'Cycle_Index'
            break
    
    # Voltage
    voltage_pattern = r'voltage[_\s]*\(v\)|voltage\(v\)'
    for i, col in enumerate(cols_lower):
        if re.search(voltage_pattern, col):
            col_mapping[std_df.columns[i]] = 'Voltage'
            break
    
    # Current
    current_pattern = r'current[_\s]*\(a\)|current\(a\)'
    for i, col in enumerate(cols_lower):
        if re.search(current_pattern, col):
            col_mapping[std_df.columns[i]] = 'Current'
            break
    
    # Charge capacity
    charge_cap_pattern = r'charge[_\s]*capacity[_\s]*\(ah\)|charge[_\s]*cap[_\s]*\(ah\)'
    for i, col in enumerate(cols_lower):
        if re.search(charge_cap_pattern, col):
            col_mapping[std_df.columns[i]] = 'Charge_Capacity_Ah'
            break
    
    # Discharge capacity
    discharge_cap_pattern = r'discharge[_\s]*capacity[_\s]*\(ah\)|discharge[_\s]*cap[_\s]*\(ah\)'
    for i, col in enumerate(cols_lower):
        if re.search(discharge_cap_pattern, col):
            col_mapping[std_df.columns[i]] = 'Discharge_Capacity_Ah'
            break
    
    # Test time
    time_pattern = r'test[_\s]*time|time[_\s]*\(s\)|elapsed[_\s]*time'
    for i, col in enumerate(cols_lower):
        if re.search(time_pattern, col):
            col_mapping[std_df.columns[i]] = 'Test_Time'
            break
    
    # dQ/dV
    dqdv_pattern = r'dq[/_\s]*dv|differential[_\s]*capacity'
    for i, col in enumerate(cols_lower):
        if re.search(dqdv_pattern, col):
            col_mapping[std_df.columns[i]] = 'dQdV'
            break
    
    # Step type or infer from current
    step_pattern = r'step[_\s]*type|mode'
    has_step_type = False
    for i, col in enumerate(cols_lower):
        if re.search(step_pattern, col):
            col_mapping[std_df.columns[i]] = 'Step_Type'
            has_step_type = True
            break
    
    # Rename columns based on the mapping
    std_df.rename(columns=col_mapping, inplace=True)
    
    # Add Step_Type column if it doesn't exist
    if 'Step_Type' not in std_df.columns and 'Current' in std_df.columns:
        std_df['Step_Type'] = std_df['Current'].apply(
            lambda x: 'Charge' if x >= 0 else 'Discharge'
        )
    
    # Add capacity column if it doesn't exist
    if 'Capacity' not in std_df.columns:
        if 'Charge_Capacity_Ah' in std_df.columns and 'Discharge_Capacity_Ah' in std_df.columns:
            std_df['Capacity'] = std_df.apply(
                lambda row: row['Charge_Capacity_Ah'] if row['Step_Type'] == 'Charge' 
                            else row['Discharge_Capacity_Ah'],
                axis=1
            )
    
    return std_df

def process_capacity_data(df: pd.DataFrame, total_mass: float, am_percentage: float, collector_mass: float) -> pd.DataFrame:
    """Process capacity data to convert to mAh/g.
    
    Args:
        df: DataFrame with standardized column names
        total_mass: Total electrode mass in mg
        am_percentage: Active material percentage (0-100)
        collector_mass: Mass of the current collector in mg
        
    Returns:
        DataFrame with added mAh/g capacity columns
    """
    # Create a copy to avoid modifying the original
    proc_df = df.copy()
    
    # Calculate normalization factor (mAh/g)
    # Convert Ah to mAh (x1000) and normalize by active material mass in g
    conversion_factor = (1000 / ((total_mass - collector_mass) * am_percentage / 100000))
    
    # Add normalized capacity columns
    if 'Charge_Capacity_Ah' in proc_df.columns:
        proc_df['Charge_Capacity_mAh_g'] = proc_df['Charge_Capacity_Ah'] * conversion_factor
    
    if 'Discharge_Capacity_Ah' in proc_df.columns:
        proc_df['Discharge_Capacity_mAh_g'] = proc_df['Discharge_Capacity_Ah'] * conversion_factor
    
    # Update main capacity column
    if 'Capacity' in proc_df.columns:
        proc_df['Capacity'] = proc_df.apply(
            lambda row: row.get('Charge_Capacity_mAh_g', 0) if row['Step_Type'] == 'Charge' 
                        else row.get('Discharge_Capacity_mAh_g', 0),
            axis=1
        )
    
    return proc_df

def create_plot_title(base_title: str, cell_metadata: dict = None) -> str:
    """Create a standardized plot title with metadata.
    
    Args:
        base_title: The main title of the plot
        cell_metadata: Dictionary with cell metadata to display
        
    Returns:
        Formatted title string with metadata
    """
    if not cell_metadata:
        return base_title
    
    # First line: Main title with Cell ID
    title = f"<b>{base_title}</b> - Cell {cell_metadata.get('cell_id', 'N/A')}"
    
    # Add Cathode Parameters section with slightly larger font
    title += "<br><span style='font-size:14px'><b>Cathode Parameters</b></span>"
    
    # Add cathode specific parameters (omit any missing fields)
    cathode_params = []
    
    if 'cathode_type' in cell_metadata:
        cathode_params.append(f"Active Material: {cell_metadata.get('cathode_type', 'N/A')}")
    
    if 'cathode_composition' in cell_metadata:
        cathode_params.append(f"Composition: {cell_metadata.get('cathode_composition', 'N/A')}")
    
    if 'cathode_mixing_method' in cell_metadata:
        cathode_params.append(f"Mixing Method: {cell_metadata.get('cathode_mixing_method', 'N/A')}")
    
    if 'cathode_mass' in cell_metadata:
        cathode_params.append(f"Mass: {cell_metadata.get('cathode_mass', 'N/A')}")
    
    if 'pressure' in cell_metadata:
        cathode_params.append(f"Pressure: {cell_metadata.get('pressure', 'N/A')}")
    
    # Add cathode parameters if any exist
    if cathode_params:
        title += f"<br><span style='font-size:12px'>{' | '.join(cathode_params)}</span>"
    
    # Add Anode Parameters section
    title += "<br><span style='font-size:14px'><b>Anode Parameters</b></span>"
    
    # Add anode specific parameters
    anode_params = []
    
    if 'anode_type' in cell_metadata:
        anode_params.append(f"Active Material: {cell_metadata.get('anode_type', 'N/A')}")
    
    if 'anode_composition' in cell_metadata:
        anode_params.append(f"Composition: {cell_metadata.get('anode_composition', 'N/A')}")
    
    if 'anode_mixing_method' in cell_metadata:
        anode_params.append(f"Mixing Method: {cell_metadata.get('anode_mixing_method', 'N/A')}")
    
    if 'anode_mass' in cell_metadata:
        anode_params.append(f"Mass: {cell_metadata.get('anode_mass', 'N/A')}")
    
    # Add anode parameters if any exist
    if anode_params:
        title += f"<br><span style='font-size:12px'>{' | '.join(anode_params)}</span>"
    
    # Add Cell Parameters section
    title += "<br><span style='font-size:14px'><b>Cell Parameters</b></span>"
    
    # Add cell parameters
    cell_params = []
    
    if 'electrolyte' in cell_metadata:
        cell_params.append(f"Electrolyte: {cell_metadata.get('electrolyte', 'N/A')}")
    
    if 'electrolyte_quantity' in cell_metadata:
        cell_params.append(f"Electrolyte quantity: {cell_metadata.get('electrolyte_quantity', 'N/A')}")
    
    if 'channel' in cell_metadata:
        cell_params.append(f"Channel: {cell_metadata.get('channel', 'N/A')}")
    
    if 'voltage_range' in cell_metadata:
        cell_params.append(f"Voltage range: {cell_metadata.get('voltage_range', 'N/A')}")
    
    if 'c_rate' in cell_metadata:
        cell_params.append(f"C-rate: {cell_metadata.get('c_rate', 'N/A')}")
    
    if 'date' in cell_metadata:
        cell_params.append(f"Date: {cell_metadata.get('date', 'N/A')}")
    
    # Add cell parameters if any exist
    if cell_params:
        title += f"<br><span style='font-size:12px'>{' | '.join(cell_params)}</span>"
    
    return title

def plot_charge_discharge(df: pd.DataFrame, cycles_to_plot: list = None, 
                          cell_metadata: dict = None, y_range: list = None) -> go.Figure:
    """Plot charge-discharge curves for specified cycles.
    
    Args:
        df: DataFrame containing electrochemistry data
        cycles_to_plot: List of cycle numbers to plot. Defaults to [1, 10, 50, 100]
        cell_metadata: Dictionary with cell metadata to display in the plot
        y_range: Optional list specifying [min, max] for y-axis range
        
    Returns:
        Plotly figure object
    """
    if cycles_to_plot is None:
        cycles_to_plot = [1, 10, 50, 100]
    
    # Create extended color palette for cycles (matplotlib-compatible)
    import plotly.colors as pc
    base_colors = pc.qualitative.Plotly  # These are already hex format
    # Add more colors but ensure they're in hex format (same as publication plots)
    additional_colors = [
        '#A6CEE3', '#1F78B4', '#B2DF8A', '#33A02C', '#FB9A99',
        '#E31A1C', '#FDBF6F', '#FF7F00', '#CAB2D6', '#6A3D9A',
        '#8DD3C7', '#FFFFB3', '#BEBADA', '#FB8072', '#80B1D3',
        '#FDB462', '#B3DE69', '#FCCDE5', '#D9D9D9', '#BC80BD',
        '#CCEBC5', '#FFED6F', '#1B9E77', '#D95F02', '#7570B3'
    ]
    extended_colors = base_colors + additional_colors
    
    # Create figure and disable automatic color cycling to ensure our explicit colors are used
    fig = go.Figure()
    fig.update_layout(colorway=extended_colors)
    
    for i, cycle in enumerate(cycles_to_plot):
        if cycle in df['Cycle_Index'].unique():
            # Use cycle number (not index) to ensure same cycle always gets same color
            color_idx = (cycle - 1) % len(extended_colors)  # cycle-1 so cycle 1 gets first color
            cycle_color = extended_colors[color_idx]
            
            # Debug: Ensure color is properly formatted and consistent
            if not isinstance(cycle_color, str):
                cycle_color = str(cycle_color)
            if not cycle_color.startswith('#'):
                cycle_color = f"#{cycle_color}" if len(cycle_color) == 6 else extended_colors[0]
            
            # Get charge data for this cycle
            charge_mask = (df['Cycle_Index'] == cycle) & (df['Step_Type'] == 'Charge')
            charge_data = df[charge_mask]
            
            # Get discharge data for this cycle
            discharge_mask = (df['Cycle_Index'] == cycle) & (df['Step_Type'] == 'Discharge')
            discharge_data = df[discharge_mask]
            
            # Add trace for charge part
            fig.add_trace(
                go.Scatter(
                    x=charge_data['Capacity'],
                    y=charge_data['Voltage'],
                    name=f'Cycle {cycle}',
                    line=dict(color=cycle_color, width=2),  # Explicit width, solid line
                    legendgroup=f'cycle_{cycle}',
                    showlegend=True
                )
            )
            
            # Add trace for discharge part (same legend group, don't show in legend)
            fig.add_trace(
                go.Scatter(
                    x=discharge_data['Capacity'],
                    y=discharge_data['Voltage'],
                    name=f'Cycle {cycle} Discharge',  # Different name to avoid confusion
                    line=dict(color=cycle_color, width=2, dash='dot'),  # Same color and width, dotted line
                    legendgroup=f'cycle_{cycle}',
                    showlegend=False  # Explicitly hide from legend
                )
            )
    
    # Create title with integrated metadata
    title = create_plot_title("Charge-Discharge Curves", cell_metadata)
    
    # Update layout with better axis visibility and title positioned at the top of the plot
    layout_update = {
        "title": {
            "text": title,
            "font": {"size": 17},  # Further reduced font size
            "x": 0.5,  # Center the title
            "xanchor": "center",
            "y": 0.97,  # Lowered position
            "yanchor": "top"
        },
        "xaxis": {
            "title": "Capacity (mAh/g)",
            "showline": True,
            "linewidth": 2,
            "linecolor": 'black',
            "mirror": True,
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": 'lightgray'
        },
        "yaxis": {
            "title": "Voltage (V)",
            "showline": True,
            "linewidth": 2,
            "linecolor": 'black',
            "mirror": True,
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": 'lightgray'
        },
        "showlegend": True,
        "legend_title": "Cycle",
        "margin": {"t": 220, "r": 30, "l": 80, "b": 80}  # Increased top margin
    }
    
    # Set y-axis range if provided
    if y_range is not None and len(y_range) == 2:
        layout_update["yaxis"]["range"] = y_range
    
    fig.update_layout(**layout_update)
    
    return fig

def plot_capacity_vs_cycle(df: pd.DataFrame, cell_metadata: dict = None, y_range: list = None) -> go.Figure:
    """Plot discharge capacity vs cycle number as a scatter plot."""
    discharge_capacity = df[df['Step_Type'] == 'Discharge'].groupby('Cycle_Index')['Capacity'].max()
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=discharge_capacity.index,
            y=discharge_capacity.values,
            name='Discharge Capacity',
            mode='markers',
            marker=dict(size=10, symbol='circle', color='blue'),
            showlegend=False  # Hide legend label
        )
    )
    
    # Create title with integrated metadata
    title = create_plot_title("Discharge Capacity vs Cycle Number", cell_metadata)
    
    # Update layout with better axis visibility and title positioned at the top of the plot
    layout_update = {
        "title": {
            "text": title,
            "font": {"size": 17},  # Further reduced font size
            "x": 0.5,  # Center the title
            "xanchor": "center",
            "y": 0.97,  # Lowered position
            "yanchor": "top"
        },
        "xaxis": {
            "title": "Cycle Number",
            "showline": True,
            "linewidth": 2,
            "linecolor": 'black',
            "mirror": True,
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": 'lightgray'
        },
        "yaxis": {
            "title": "Capacity (mAh/g)",
            "showline": True,
            "linewidth": 2,
            "linecolor": 'black',
            "mirror": True,
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": 'lightgray'
        },
        "showlegend": False,  # Hide the legend completely
        "margin": {"t": 220, "r": 30, "l": 80, "b": 80}  # Increased top margin
    }
    
    # Set y-axis range if provided
    if y_range is not None and len(y_range) == 2:
        layout_update["yaxis"]["range"] = y_range
    
    fig.update_layout(**layout_update)
    
    return fig

def plot_state_of_health(df: pd.DataFrame, cell_metadata: dict = None, y_range: list = None) -> go.Figure:
    """Plot state of health vs cycle number."""
    discharge_capacity = df[df['Step_Type'] == 'Discharge'].groupby('Cycle_Index')['Capacity'].max()
    initial_capacity = discharge_capacity.iloc[0]
    soh = (discharge_capacity / initial_capacity) * 100
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=soh.index,
            y=soh.values,
            name='State of Health',
            mode='lines+markers',
            marker=dict(size=8),
            line=dict(width=2)
        )
    )
    
    # Create title with integrated metadata
    title = create_plot_title("State of Health vs Cycle Number", cell_metadata)
    
    # Update layout with better axis visibility and title positioned at the top of the plot
    layout_update = {
        "title": {
            "text": title,
            "font": {"size": 17},  # Further reduced font size
            "x": 0.5,  # Center the title
            "xanchor": "center",
            "y": 0.97,  # Lowered position
            "yanchor": "top"
        },
        "xaxis": {
            "title": "Cycle Number",
            "showline": True,
            "linewidth": 2,
            "linecolor": 'black',
            "mirror": True,
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": 'lightgray'
        },
        "yaxis": {
            "title": "State of Health (%)",
            "showline": True,
            "linewidth": 2,
            "linecolor": 'black',
            "mirror": True,
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": 'lightgray'
        },
        "showlegend": True,
        "margin": {"t": 220, "r": 30, "l": 80, "b": 80}  # Increased top margin
    }
    
    # Set y-axis range if provided
    if y_range is not None and len(y_range) == 2:
        layout_update["yaxis"]["range"] = y_range
    
    fig.update_layout(**layout_update)
    
    return fig

def plot_coulombic_efficiency(df: pd.DataFrame, cell_metadata: dict = None, y_range: list = None) -> go.Figure:
    """Plot coulombic efficiency vs cycle number as a scatter plot."""
    discharge_capacity = df[df['Step_Type'] == 'Discharge'].groupby('Cycle_Index')['Capacity'].max()
    charge_capacity = df[df['Step_Type'] == 'Charge'].groupby('Cycle_Index')['Capacity'].max()
    coulombic_efficiency = (discharge_capacity / charge_capacity) * 100
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=coulombic_efficiency.index,
            y=coulombic_efficiency.values,
            name='Coulombic Efficiency',
            mode='markers',
            marker=dict(size=10, symbol='circle', color='red'),
            showlegend=False  # Hide legend
        )
    )
    
    # Create title with integrated metadata
    title = create_plot_title("Coulombic Efficiency vs Cycle Number", cell_metadata)
    
    # Update layout with better axis visibility and title positioned at the top of the plot
    layout_update = {
        "title": {
            "text": title,
            "font": {"size": 17},  # Further reduced font size
            "x": 0.5,  # Center the title
            "xanchor": "center",
            "y": 0.97,  # Lowered position
            "yanchor": "top"
        },
        "xaxis": {
            "title": "Cycle Number",
            "showline": True,
            "linewidth": 2,
            "linecolor": 'black',
            "mirror": True,
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": 'lightgray'
        },
        "yaxis": {
            "title": "Coulombic Efficiency (%)",
            "showline": True,
            "linewidth": 2,
            "linecolor": 'black',
            "mirror": True,
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": 'lightgray'
        },
        "showlegend": False,  # Hide the legend completely
        "margin": {"t": 220, "r": 30, "l": 80, "b": 80}  # Increased top margin
    }
    
    # Set y-axis range if provided
    if y_range is not None and len(y_range) == 2:
        layout_update["yaxis"]["range"] = y_range
    
    fig.update_layout(**layout_update)
    
    return fig

def plot_voltage_time(df: pd.DataFrame, cell_metadata: dict = None, y_range: list = None) -> go.Figure:
    """Plot voltage vs time profile."""
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df['Test_Time'],
            y=df['Voltage'],
            name='Voltage Profile',
            mode='lines',
            line=dict(width=2)
        )
    )
    
    # Create title with integrated metadata
    title = create_plot_title("Voltage vs Time Profile", cell_metadata)
    
    # Update layout with better axis visibility and title positioned at the top of the plot
    layout_update = {
        "title": {
            "text": title,
            "font": {"size": 17},  # Further reduced font size
            "x": 0.5,  # Center the title
            "xanchor": "center",
            "y": 0.97,  # Lowered position
            "yanchor": "top"
        },
        "xaxis": {
            "title": "Time (h)",
            "showline": True,
            "linewidth": 2,
            "linecolor": 'black',
            "mirror": True,
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": 'lightgray'
        },
        "yaxis": {
            "title": "Voltage (V)",
            "showline": True,
            "linewidth": 2,
            "linecolor": 'black',
            "mirror": True,
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": 'lightgray'
        },
        "showlegend": True,
        "margin": {"t": 220, "r": 30, "l": 80, "b": 80}  # Increased top margin
    }
    
    # Set y-axis range if provided
    if y_range is not None and len(y_range) == 2:
        layout_update["yaxis"]["range"] = y_range
    
    fig.update_layout(**layout_update)
    
    return fig

def plot_differential_capacity(df: pd.DataFrame, cycles: list = [1, 2],
                            cell_metadata: dict = None,
                            voltage_range: list = None,
                            polynomial_spline: int = 3, s_spline: float = 1e-5,
                            polyorder_1: int = 5, window_size_1: int = 101,
                            polyorder_2: int = 5, window_size_2: int = 1001,
                            final_smooth: bool = True) -> go.Figure:
    """Plot differential capacity (dQ/dV) curves for multiple cycles.
    
    Args:
        df: DataFrame containing electrochemistry data
        cycles: List of cycle numbers to plot. Defaults to [1, 2]
        cell_metadata: Dictionary with cell metadata to display in the plot
        voltage_range: Optional list specifying [min, max] for voltage axis
        polynomial_spline: Order of the spline interpolation for the capacity-voltage curve
        s_spline: Smoothing factor for the spline interpolation
        polyorder_1: Order of the polynomial for the first smoothing filter
        window_size_1: Size of the window for the first smoothing filter
        polyorder_2: Order of the polynomial for the second smoothing filter
        window_size_2: Size of the window for the second smoothing filter
        final_smooth: Whether to apply final smoothing to the dQ/dV curve
        
    Returns:
        Plotly figure object
    """
    if not cycles:
        raise ValueError("cycles list cannot be empty")
    
    # Create the plot
    fig = go.Figure()
    
    # Create extended color palette for cycles (consistent with charge-discharge plots)
    import plotly.colors as pc
    base_colors = pc.qualitative.Plotly  # These are already hex format
    # Add more colors but ensure they're in hex format (same as other plots)
    additional_colors = [
        '#A6CEE3', '#1F78B4', '#B2DF8A', '#33A02C', '#FB9A99',
        '#E31A1C', '#FDBF6F', '#FF7F00', '#CAB2D6', '#6A3D9A',
        '#8DD3C7', '#FFFFB3', '#BEBADA', '#FB8072', '#80B1D3',
        '#FDB462', '#B3DE69', '#FCCDE5', '#D9D9D9', '#BC80BD',
        '#CCEBC5', '#FFED6F', '#1B9E77', '#D95F02', '#7570B3'
    ]
    extended_colors = base_colors + additional_colors
    
    # Process each cycle
    for cycle_idx, cycle in enumerate(cycles):
        cycle_mask = df['Cycle_Index'] == cycle
        cycle_data = df[cycle_mask]
        
        # Use cycle number (not index) to ensure same cycle always gets same color
        color_idx = (cycle - 1) % len(extended_colors)  # cycle-1 so cycle 1 gets first color
        cycle_color = extended_colors[color_idx]
        
        # Process both charge and discharge
        for step_type in ['Charge', 'Discharge']:
            step_mask = cycle_data['Step_Type'] == step_type
            step_data = cycle_data[step_mask]
            
            # Use raw capacity values (Ah) instead of normalized ones
            capacity = step_data['Charge_Capacity_Ah'].values if step_type == 'Charge' else step_data['Discharge_Capacity_Ah'].values
            voltage = step_data['Voltage'].values
            
            # Process data using the original method
            # Group by voltage and calculate mean capacity
            df_step = pd.DataFrame({'Capacity': capacity, 'Voltage': voltage})
            unique_v = df_step.astype(float).groupby('Voltage').mean().index
            unique_v_cap = df_step.astype(float).groupby('Voltage').mean()['Capacity']
            
            # Create smooth voltage points
            x_volt = np.linspace(min(voltage), max(voltage), num=int(1e4))
            
            # First spline fit
            f_lit = splrep(unique_v, unique_v_cap, k=1, s=0.0)
            y_cap = splev(x_volt, f_lit)
            smooth_cap = savgol_filter(y_cap, window_size_1, polyorder_1)
            
            # Second spline fit and differentiation
            f_smooth = splrep(x_volt, smooth_cap, k=polynomial_spline, s=s_spline)
            dqdv = splev(x_volt, f_smooth, der=1)
            
            # Final smoothing if requested
            if final_smooth:
                dqdv = savgol_filter(dqdv, window_size_2, polyorder_2)
            
            # Add the dQ/dV trace
            fig.add_trace(
                go.Scatter(
                    x=x_volt,
                    y=dqdv,
                    name=f'{step_type} - Cycle {cycle}',
                    line=dict(
                        color=cycle_color,  # Use consistent cycle color
                        width=2,
                        dash='solid' if step_type == 'Charge' else 'dot'
                    ),
                    mode='lines',
                    legendgroup=f'cycle_{cycle}',
                    showlegend=True
                )
            )
    
    # Create title with integrated metadata
    cycles_str = ', '.join(map(str, cycles))
    title = create_plot_title(f"Differential Capacity (dQ/dV) Curves - Cycles {cycles_str}", cell_metadata)
    
    # Update layout with better axis visibility and title positioned at the top of the plot
    layout_update = {
        "title": {
            "text": title,
            "font": {"size": 17},  # Further reduced font size
            "x": 0.5,  # Center the title
            "xanchor": "center",
            "y": 0.97,  # Lowered position
            "yanchor": "top"
        },
        "xaxis": {
            "title": "Voltage (V)",
            "showline": True,
            "linewidth": 2,
            "linecolor": 'black',
            "mirror": True,
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": 'lightgray'
        },
        "yaxis": {
            "title": "dQ/dV (Ah/V)",
            "showline": True,
            "linewidth": 2,
            "linecolor": 'black',
            "mirror": True,
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": 'lightgray'
        },
        "showlegend": True,
        "legend": {
            "orientation": "h",  # Horizontal legend
            "yanchor": "bottom",
            "y": -0.25,  # Move legend further down
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(255, 255, 255, 0.9)",  # Semi-transparent white background
            "bordercolor": "black",
            "borderwidth": 1
        },
        "margin": {"t": 220, "r": 30, "l": 80, "b": 200}  # Increased top margin, maintaining bottom margin for legend
    }
    
    # Set voltage range if provided
    if voltage_range is not None and len(voltage_range) == 2:
        layout_update["xaxis"]["range"] = voltage_range
    
    fig.update_layout(**layout_update)
    
    return fig

def save_plot(fig: go.Figure, output_path: Path, save_png: bool = True, width: int = 1000, height: int = 700, dpi: int = 300) -> Path:
    """Save plotly figure to HTML and optionally PNG file with high resolution.
    
    Args:
        fig: Plotly figure object
        output_path: Path where to save the HTML file
        save_png: Whether to also save as PNG
        width: Image width in pixels (for PNG)
        height: Image height in pixels (for PNG)
        dpi: Dots per inch for PNG export (higher values = better print quality)
        
    Returns:
        Path to the saved HTML file
    """
    # Make sure the directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a copy of the figure for PNG export to avoid modifying the original
    if save_png:
        png_fig = go.Figure(fig)
        # Improve layout for PNG export
        png_fig.update_layout(
            width=width,
            height=height,
            font=dict(size=14),  # Reasonably sized font
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=80, r=80, t=180, b=80)
        )
        
        # Make lines thicker for better visibility in PNG
        for data in png_fig.data:
            if hasattr(data, 'line') and data.line is not None:
                if hasattr(data.line, 'width') and data.line.width is not None:
                    data.line.width = data.line.width * 1.5
                else:
                    data.line.width = 3
            
            if hasattr(data, 'marker') and data.marker is not None:
                if hasattr(data.marker, 'size') and data.marker.size is not None:
                    data.marker.size = data.marker.size * 1.2
                else:
                    data.marker.size = 10
    
    # Save HTML (original figure)
    output_html = output_path.with_suffix('.html')
    fig.write_html(str(output_html))
    
    # Save PNG if requested
    if save_png:
        png_path = output_path.with_suffix('.png')
        try:
            # Calculate appropriate scale based on DPI
            # Standard screen is 96 DPI, so scale is relative to that
            scale = dpi / 96
            
            # Use kaleido for higher quality with explicit DPI setting
            png_fig.write_image(
                str(png_path), 
                scale=scale,
                width=width,
                height=height,
                engine="kaleido",
                format="png"
            )
            
            print(f"High-quality PNG saved to {png_path} (DPI: {dpi})")
        except Exception as e:
            print(f"Warning: Error with high-quality export: {str(e)}")
            try:
                # Try alternative approach
                import plotly.io as pio
                pio.write_image(
                    png_fig, 
                    str(png_path), 
                    format='png',
                    width=width, 
                    height=height
                )
                print(f"PNG saved using alternative method to {png_path}")
            except Exception as e2:
                print(f"Warning: Could not save PNG image. {str(e2)}")
                print("To enable high-quality PNG export, install required dependencies:")
                print("pip install kaleido pillow")
            
    return output_html

def plot_combined_performance(df: pd.DataFrame, cycles_to_plot: list = None, 
                               cell_metadata: dict = None, voltage_range: list = None,
                               capacity_range: list = None, exclude_last_cycle: bool = False) -> go.Figure:
    """Create a side-by-side plot of charge-discharge curves and capacity vs cycle.
    
    Args:
        df: DataFrame containing electrochemistry data
        cycles_to_plot: List of cycle numbers to plot. Defaults to [1, 10, 50, 100]
        cell_metadata: Dictionary with cell metadata to display in the plot
        voltage_range: Optional list specifying [min, max] for voltage axis
        capacity_range: Optional list specifying [min, max] for capacity axis
        exclude_last_cycle: If True, excludes the last cycle from the capacity vs cycle plot
    """
    if cycles_to_plot is None:
        cycles_to_plot = [1, 10, 50, 100]
    
    # Create a figure with secondary y-axis
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=('Charge-Discharge Curves', 'Capacity vs Cycle'),
                       horizontal_spacing=0.15,
                       vertical_spacing=0.3)  # Further increased vertical spacing for title
    
    # Create color scale for cycles
    import plotly.colors as pc
    colors = pc.qualitative.Plotly
    
    # Plot charge-discharge curves (left subplot)
    for i, cycle in enumerate(cycles_to_plot):
        if cycle in df['Cycle_Index'].unique():
            color_idx = i % len(colors)
            cycle_color = colors[color_idx]
            
            # Get charge data for this cycle
            charge_mask = (df['Cycle_Index'] == cycle) & (df['Step_Type'] == 'Charge')
            charge_data = df[charge_mask]
            
            # Get discharge data for this cycle
            discharge_mask = (df['Cycle_Index'] == cycle) & (df['Step_Type'] == 'Discharge')
            discharge_data = df[discharge_mask]
            
            # Add trace for charge part
            fig.add_trace(
                go.Scatter(
                    x=charge_data['Capacity'],
                    y=charge_data['Voltage'],
                    name=f'Cycle {cycle}',
                    line=dict(color=cycle_color),
                    legendgroup=f'cycle_{cycle}',
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Add trace for discharge part (same legend group, don't show in legend)
            fig.add_trace(
                go.Scatter(
                    x=discharge_data['Capacity'],
                    y=discharge_data['Voltage'],
                    name=f'Cycle {cycle}',
                    line=dict(color=cycle_color, dash='dot'),
                    legendgroup=f'cycle_{cycle}',
                    showlegend=False
                ),
                row=1, col=1
            )
    
    # Plot capacity data (right subplot)
    discharge_capacity = df[df['Step_Type'] == 'Discharge'].groupby('Cycle_Index')['Capacity'].max()
    
    # Exclude last cycle if requested
    if exclude_last_cycle and len(discharge_capacity) > 1:
        discharge_capacity = discharge_capacity.iloc[:-1]
    
    fig.add_trace(
        go.Scatter(
            x=discharge_capacity.index,
            y=discharge_capacity.values,
            mode='markers',
            marker=dict(size=10, symbol='circle', color='blue'),
            showlegend=False  # Don't show in legend
        ),
        row=1, col=2
    )
    
    # Create title with integrated metadata
    title = create_plot_title("Battery Performance Overview", cell_metadata)
    
    # Update layout with better axis visibility and title positioned at the top of the plot
    layout_update = {
        "title": {
            "text": title,
            "font": {"size": 17},  # Further reduced font size
            "x": 0.5,  # Center the title
            "xanchor": "center",
            "y": 0.97,  # Lowered position
            "yanchor": "top"
        },
        "showlegend": True,
        "legend": {
            "orientation": "h",  # Horizontal legend
            "yanchor": "bottom",
            "y": -0.25,  # Move legend further down
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(255, 255, 255, 0.9)",  # Semi-transparent white background
            "bordercolor": "black",
            "borderwidth": 1
        },
        "margin": {"t": 240, "r": 30, "l": 80, "b": 200}  # Further increased top margin for combined plot
    }
    
    # Update axes for both subplots
    fig.update_xaxes(
        title_text="Capacity (mAh/g)",
        title_standoff=20,  # Add space between axis and its title
        showline=True, linewidth=2, linecolor='black', mirror=True,
        showgrid=True, gridwidth=1, gridcolor='lightgray',
        row=1, col=1
    )
    
    fig.update_xaxes(
        title_text="Cycle Number",
        title_standoff=20,  # Add space between axis and its title
        showline=True, linewidth=2, linecolor='black', mirror=True,
        showgrid=True, gridwidth=1, gridcolor='lightgray',
        row=1, col=2
    )
    
    fig.update_yaxes(
        title_text="Voltage (V)",
        title_standoff=20,  # Add space between axis and its title
        showline=True, linewidth=2, linecolor='black', mirror=True,
        showgrid=True, gridwidth=1, gridcolor='lightgray',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Capacity (mAh/g)",
        title_standoff=20,  # Add space between axis and its title
        showline=True, linewidth=2, linecolor='black', mirror=True,
        showgrid=True, gridwidth=1, gridcolor='lightgray',
        row=1, col=2
    )
    
    # Set axis ranges if provided
    if voltage_range is not None and len(voltage_range) == 2:
        fig.update_yaxes(range=voltage_range, row=1, col=1)
    
    if capacity_range is not None and len(capacity_range) == 2:
        fig.update_yaxes(range=capacity_range, row=1, col=2)
    
    fig.update_layout(**layout_update)
    
    return fig 

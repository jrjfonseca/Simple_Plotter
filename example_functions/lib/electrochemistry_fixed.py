import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import re
import os
import sys

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
        
        # Try to identify the data and stats sheets
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
    
    # Create color scale for cycles
    import plotly.colors as pc
    colors = pc.qualitative.Plotly
    
    fig = go.Figure()
    
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
                )
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
                )
            )
    
    # Create title with integrated metadata
    title = "Charge-Discharge Curves"
    if cell_metadata:
        metadata_str = f"Cell {cell_metadata.get('cell_id', 'N/A')} | {cell_metadata.get('cathode_type', 'N/A')}/{cell_metadata.get('anode_type', 'N/A')} | {cell_metadata.get('cathode_composition', 'N/A')} | {cell_metadata.get('c_rate', 'N/A')}"
        title = f"<b>{title}</b><br><span style='font-size:16px'>{metadata_str}</span><br><span style='font-size:14px'>Electrolyte: {cell_metadata.get('electrolyte', 'N/A')}</span>"
    
    # Update layout with better axis visibility and title positioned closer to plot area
    layout_update = {
        "title": {
            "text": title,
            "font": {"size": 22},
            "x": 0.0,  # Left align
            "xanchor": "left",
            "y": 0.85,  # Lower position to be closer to the plot area
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
        "legend_title": "Cycle",  # Use original style legend title
        "margin": {"t": 170, "r": 30, "l": 80, "b": 80}  # Extra top margin for multi-line title
    }
    
    # Set y-axis range if provided
    if y_range is not None and len(y_range) == 2:
        layout_update["yaxis"]["range"] = y_range
    
    fig.update_layout(**layout_update)
    
    return fig

def plot_capacity_vs_cycle(df: pd.DataFrame, cell_metadata: dict = None, y_range: list = None) -> go.Figure:
    """Plot discharge capacity vs cycle number as a scatter plot.
    
    Args:
        df: DataFrame containing electrochemistry data
        cell_metadata: Dictionary with cell metadata to display in the plot
        y_range: Optional list specifying [min, max] for y-axis range
        
    Returns:
        Plotly figure object
    """
    discharge_capacity = df[df['Step_Type'] == 'Discharge'].groupby('Cycle_Index')['Capacity'].max()
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=discharge_capacity.index,
            y=discharge_capacity.values,
            name='Discharge Capacity',
            mode='markers',
            marker=dict(size=10, symbol='circle', color='blue'),
        )
    )
    
    # Create title with integrated metadata
    title = "Discharge Capacity vs Cycle Number"
    if cell_metadata:
        metadata_str = f"Cell {cell_metadata.get('cell_id', 'N/A')} | {cell_metadata.get('cathode_type', 'N/A')}/{cell_metadata.get('anode_type', 'N/A')} | {cell_metadata.get('cathode_composition', 'N/A')} | {cell_metadata.get('c_rate', 'N/A')}"
        title = f"<b>{title}</b><br><span style='font-size:16px'>{metadata_str}</span><br><span style='font-size:14px'>Electrolyte: {cell_metadata.get('electrolyte', 'N/A')}</span>"
    
    # Update layout with better axis visibility and title positioned closer to plot area
    layout_update = {
        "title": {
            "text": title,
            "font": {"size": 22},
            "x": 0.0,  # Left align
            "xanchor": "left",
            "y": 0.85,  # Lower position to be closer to the plot area
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
        "showlegend": True,
        "margin": {"t": 170, "r": 30, "l": 80, "b": 80}  # Extra top margin for multi-line title
    }
    
    # Set y-axis range if provided
    if y_range is not None and len(y_range) == 2:
        layout_update["yaxis"]["range"] = y_range
    
    fig.update_layout(**layout_update)
    
    return fig

def plot_state_of_health(df: pd.DataFrame, cell_metadata: dict = None, y_range: list = None) -> go.Figure:
    """Plot state of health vs cycle number.
    
    Args:
        df: DataFrame containing electrochemistry data
        cell_metadata: Dictionary with cell metadata to display in the plot
        y_range: Optional list specifying [min, max] for y-axis range
        
    Returns:
        Plotly figure object
    """
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
    title = "State of Health vs Cycle Number"
    if cell_metadata:
        metadata_str = f"Cell {cell_metadata.get('cell_id', 'N/A')} | {cell_metadata.get('cathode_type', 'N/A')}/{cell_metadata.get('anode_type', 'N/A')} | {cell_metadata.get('cathode_composition', 'N/A')} | {cell_metadata.get('c_rate', 'N/A')}"
        title = f"<b>{title}</b><br><span style='font-size:16px'>{metadata_str}</span><br><span style='font-size:14px'>Electrolyte: {cell_metadata.get('electrolyte', 'N/A')}</span>"
    
    # Update layout with better axis visibility and title positioned closer to plot area
    layout_update = {
        "title": {
            "text": title,
            "font": {"size": 22},
            "x": 0.0,  # Left align
            "xanchor": "left",
            "y": 0.85,  # Lower position to be closer to the plot area
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
        "margin": {"t": 170, "r": 30, "l": 80, "b": 80}  # Extra top margin for multi-line title
    }
    
    # Set y-axis range if provided
    if y_range is not None and len(y_range) == 2:
        layout_update["yaxis"]["range"] = y_range
    
    fig.update_layout(**layout_update)
    
    return fig

def plot_coulombic_efficiency(df: pd.DataFrame, cell_metadata: dict = None, y_range: list = None) -> go.Figure:
    """Plot coulombic efficiency vs cycle number as a scatter plot.
    
    Args:
        df: DataFrame containing electrochemistry data
        cell_metadata: Dictionary with cell metadata to display in the plot
        y_range: Optional list specifying [min, max] for y-axis range
        
    Returns:
        Plotly figure object
    """
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
        )
    )
    
    # Create title with integrated metadata
    title = "Coulombic Efficiency vs Cycle Number"
    if cell_metadata:
        metadata_str = f"Cell {cell_metadata.get('cell_id', 'N/A')} | {cell_metadata.get('cathode_type', 'N/A')}/{cell_metadata.get('anode_type', 'N/A')} | {cell_metadata.get('cathode_composition', 'N/A')} | {cell_metadata.get('c_rate', 'N/A')}"
        title = f"<b>{title}</b><br><span style='font-size:16px'>{metadata_str}</span><br><span style='font-size:14px'>Electrolyte: {cell_metadata.get('electrolyte', 'N/A')}</span>"
    
    # Update layout with better axis visibility and title positioned closer to plot area
    layout_update = {
        "title": {
            "text": title,
            "font": {"size": 22},
            "x": 0.0,  # Left align
            "xanchor": "left",
            "y": 0.85,  # Lower position to be closer to the plot area
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
        "showlegend": True,
        "margin": {"t": 170, "r": 30, "l": 80, "b": 80}  # Extra top margin for multi-line title
    }
    
    # Set y-axis range if provided
    if y_range is not None and len(y_range) == 2:
        layout_update["yaxis"]["range"] = y_range
    
    fig.update_layout(**layout_update)
    
    return fig

def plot_differential_capacity(df: pd.DataFrame, cycles_to_plot: list = None, 
                               cell_metadata: dict = None, y_range: list = None) -> go.Figure:
    """Plot differential capacity analysis (dQ/dV).
    
    Args:
        df: DataFrame containing electrochemistry data
        cycles_to_plot: List of cycle numbers to plot. Required.
        cell_metadata: Dictionary with cell metadata to display in the plot
        y_range: Optional list specifying [min, max] for y-axis range
        
    Returns:
        Plotly figure object
    """
    if cycles_to_plot is None or len(cycles_to_plot) == 0:
        # Default to first cycle if none specified
        cycles_to_plot = [1]
        print("No cycles specified for dQ/dV plot, defaulting to cycle 1")
    
    # Create color scale for cycles
    import plotly.colors as pc
    colors = pc.qualitative.Plotly
    
    fig = go.Figure()
    
    # Check if we have a pre-calculated dQ/dV column
    has_dqdv_column = 'dQdV' in df.columns
    
    for i, cycle in enumerate(cycles_to_plot):
        if cycle not in df['Cycle_Index'].unique():
            print(f"Warning: Cycle {cycle} not found in data, skipping")
            continue
            
        color_idx = i % len(colors)
        cycle_color = colors[color_idx]
        
        # For discharge data only (for dQ/dV analysis)
        discharge_data = df[(df['Cycle_Index'] == cycle) & (df['Step_Type'] == 'Discharge')]
        
        if len(discharge_data) < 3:
            print(f"Warning: Not enough data points for cycle {cycle}, skipping")
            continue
            
        if has_dqdv_column:
            # Use pre-calculated dQ/dV values
            voltage = discharge_data['Voltage'].values
            dqdv = discharge_data['dQdV'].values
        else:
            # Calculate dQ/dV values from capacity and voltage
        voltage = discharge_data['Voltage'].values
        capacity = discharge_data['Capacity'].values
            
            if len(voltage) > 5 and len(capacity) > 5:
                # Smooth the data to reduce noise
                from scipy.signal import savgol_filter
                
                try:
                    # Try to smooth the data using Savitzky-Golay filter
                    # Window size must be odd and less than data length
                    window_size = min(11, (len(voltage) // 2) * 2 - 1)
                    if window_size < 5:
                        window_size = 5
                    
                    capacity_smooth = savgol_filter(capacity, window_size, 3)
                    # Calculate dQ/dV using numpy gradient on smoothed data
                    dqdv = np.gradient(capacity_smooth, voltage)
                except (ImportError, ValueError) as e:
                    print(f"Warning: Error in smoothing, using raw gradient: {e}")
                    # Fall back to simple gradient
                    dqdv = np.gradient(capacity, voltage)
            else:
                # Not enough data to smooth, use raw gradient
        dqdv = np.gradient(capacity, voltage)
        
        # Add trace for this cycle
        fig.add_trace(
            go.Scatter(
                x=voltage,
                y=dqdv,
                name=f'Cycle {cycle}',
                mode='lines',
                line=dict(color=cycle_color, width=2)
            )
        )
    
    # Create title with integrated metadata
    title = "Differential Capacity Analysis (dQ/dV)"
    if cell_metadata:
        metadata_str = f"Cell {cell_metadata.get('cell_id', 'N/A')} | {cell_metadata.get('cathode_type', 'N/A')}/{cell_metadata.get('anode_type', 'N/A')} | {cell_metadata.get('cathode_composition', 'N/A')} | {cell_metadata.get('c_rate', 'N/A')}"
        title = f"<b>{title}</b><br><span style='font-size:16px'>{metadata_str}</span><br><span style='font-size:14px'>Electrolyte: {cell_metadata.get('electrolyte', 'N/A')}</span>"
    
    # Update layout with better axis visibility and title positioned closer to plot area
    layout_update = {
        "title": {
            "text": title,
            "font": {"size": 22},
            "x": 0.0,  # Left align
            "xanchor": "left",
            "y": 0.85,  # Lower position to be closer to the plot area
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
            "title": "dQ/dV",
            "showline": True,
            "linewidth": 2,
            "linecolor": 'black',
            "mirror": True,
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": 'lightgray'
        },
        "showlegend": True,
        "legend_title": "Cycle",  # Add cycle legend title for consistency
        "margin": {"t": 170, "r": 30, "l": 80, "b": 80}  # Extra top margin for multi-line title
    }
    
    # Set y-axis range if provided
    if y_range is not None and len(y_range) == 2:
        layout_update["yaxis"]["range"] = y_range
    
    fig.update_layout(**layout_update)
    
    return fig

def plot_voltage_time(df: pd.DataFrame, cell_metadata: dict = None, y_range: list = None) -> go.Figure:
    """Plot voltage vs time profile.
    
    Args:
        df: DataFrame containing electrochemistry data
        cell_metadata: Dictionary with cell metadata to display in the plot
        y_range: Optional list specifying [min, max] for y-axis range
        
    Returns:
        Plotly figure object
    """
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
    title = "Voltage vs Time Profile"
    if cell_metadata:
        metadata_str = f"Cell {cell_metadata.get('cell_id', 'N/A')} | {cell_metadata.get('cathode_type', 'N/A')}/{cell_metadata.get('anode_type', 'N/A')} | {cell_metadata.get('cathode_composition', 'N/A')} | {cell_metadata.get('c_rate', 'N/A')}"
        title = f"<b>{title}</b><br><span style='font-size:16px'>{metadata_str}</span><br><span style='font-size:14px'>Electrolyte: {cell_metadata.get('electrolyte', 'N/A')}</span>"
    
    # Update layout with better axis visibility and title positioned closer to plot area
    layout_update = {
        "title": {
            "text": title,
            "font": {"size": 22},
            "x": 0.0,  # Left align
            "xanchor": "left",
            "y": 0.85,  # Lower position to be closer to the plot area
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
        "margin": {"t": 170, "r": 30, "l": 80, "b": 80}  # Extra top margin for multi-line title
    }
    
    # Set y-axis range if provided
    if y_range is not None and len(y_range) == 2:
        layout_update["yaxis"]["range"] = y_range
    
    fig.update_layout(**layout_update)
    
    return fig

def plot_differential_capacity_2(df: pd.DataFrame, cycles_to_plot: list = None, 
                                cell_metadata: dict = None, y_range: list = None) -> go.Figure:
    """Plot differential capacity analysis (dQ/dV) using current and time data.
    
    Args:
        df: DataFrame containing electrochemistry data
        cycles_to_plot: List of cycle numbers to plot. Required.
        cell_metadata: Dictionary with cell metadata to display in the plot
        y_range: Optional list specifying [min, max] for y-axis range
        
    Returns:
        Plotly figure object
    """
    if cycles_to_plot is None or len(cycles_to_plot) == 0:
        # Default to first cycle if none specified
        cycles_to_plot = [1]
        print("No cycles specified for dQ/dV plot, defaulting to cycle 1")
    
    # Create color scale for cycles
    import plotly.colors as pc
    colors = pc.qualitative.Plotly
    
    fig = go.Figure()
    
    # Identify time column using regex
    time_col = None
    for col in df.columns:
        if isinstance(col, str) and re.search(r'step[_\s]*time|time[_\s]*\(s\)', col.lower()):
            time_col = col
            break
    
    if time_col is None and 'Test_Time' in df.columns:
        time_col = 'Test_Time'
        print("Using Test_Time column for time data")
    elif time_col is None:
        print("Warning: Step time column not found, cannot calculate dQ/dV")
        return fig
        
    # Identify current column using regex
    current_col = None
    for col in df.columns:
        if isinstance(col, str) and re.search(r'current[_\s]*\(a\)|current\(a\)', col.lower()):
            current_col = col
            break
    
    if current_col is None and 'Current' in df.columns:
        current_col = 'Current'
        print("Using Current column for current data")
    elif current_col is None:
        print("Warning: Current column not found, cannot calculate dQ/dV")
        return fig
    
    # Define step types to include
    step_types = ['Charge', 'Discharge']
    
    # Process each cycle
    for i, cycle in enumerate(cycles_to_plot):
        if cycle not in df['Cycle_Index'].unique():
            print(f"Warning: Cycle {cycle} not found in data, skipping")
            continue
        
        color_idx = i % len(colors)
        cycle_color = colors[color_idx]
        
        # Process each step type (charge and discharge)
        for step_type in step_types:
            step_data = df[(df['Cycle_Index'] == cycle) & (df['Step_Type'] == step_type)]
            
            if len(step_data) < 5:
                print(f"Warning: Not enough {step_type.lower()} data points for cycle {cycle}, skipping")
                continue
            
            # Extract relevant data
            voltage = step_data['Voltage'].values
            current = step_data[current_col].values
            time = step_data[time_col].values
            
            # Calculate step time differences
            time_diff = np.diff(time)
            time_diff = np.append(time_diff, time_diff[-1])  # Duplicate last value for length matching
            
            # Compute cumulative charge (q = I * t)
            # For discharge, use absolute values (already negative)
            # For charge, use as-is (positive)
            charge_increments = current * time_diff
            q = np.cumsum(charge_increments)
            
            if len(voltage) > 5 and len(q) > 5:
                try:
                    # Try to smooth the data using Savitzky-Golay filter
                    from scipy.signal import savgol_filter
                    
                    # Window size must be odd and less than data length
                    window_size = min(11, (len(voltage) // 2) * 2 - 1)
                    if window_size < 5:
                        window_size = 5
                    
                    # Smooth the cumulative charge data
                    q_smooth = savgol_filter(q, window_size, 3)
                    
                    # Calculate dQ/dV using numpy gradient on smoothed data
                    dqdv = np.gradient(q_smooth, voltage)
                    
                    # Optionally smooth the final dQ/dV curve
                    dqdv_smooth = savgol_filter(dqdv, window_size, 3)
                    
                    # Use the smoothed dQ/dV
                    dqdv = dqdv_smooth
                    
                except (ImportError, ValueError) as e:
                    print(f"Warning: Error in smoothing, using raw gradient: {e}")
                    # Fall back to simple gradient
                    dqdv = np.gradient(q, voltage)
            else:
                # Not enough data to smooth, use raw gradient
                dqdv = np.gradient(q, voltage)
            
            # Define line style based on step type
            line_style = 'solid' if step_type == 'Discharge' else 'dash'
            
            # Add trace for this cycle and step type
            fig.add_trace(
                go.Scatter(
                    x=voltage,
                    y=dqdv,
                    name=f'Cycle {cycle} ({step_type})',
                    mode='lines',
                    line=dict(color=cycle_color, width=2, dash=line_style)
                )
            )
    
    # Create title with integrated metadata
    title = "Differential Capacity Analysis (dQ/dV) - From Current & Time"
    if cell_metadata:
        metadata_str = f"Cell {cell_metadata.get('cell_id', 'N/A')} | {cell_metadata.get('cathode_type', 'N/A')}/{cell_metadata.get('anode_type', 'N/A')} | {cell_metadata.get('cathode_composition', 'N/A')} | {cell_metadata.get('c_rate', 'N/A')}"
        title = f"<b>{title}</b><br><span style='font-size:16px'>{metadata_str}</span><br><span style='font-size:14px'>Electrolyte: {cell_metadata.get('electrolyte', 'N/A')}</span>"
    
    # Update layout with better axis visibility and title positioned closer to plot area
    layout_update = {
        "title": {
            "text": title,
            "font": {"size": 22},
            "x": 0.0,  # Left align
            "xanchor": "left",
            "y": 0.85,  # Lower position to be closer to the plot area
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
            "title": "dQ/dV",
            "showline": True,
            "linewidth": 2,
            "linecolor": 'black',
            "mirror": True,
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": 'lightgray'
        },
        "showlegend": True,
        "legend_title": "Cycle & Step Type",  # Update legend title to reflect both cycle and step type
        "margin": {"t": 170, "r": 30, "l": 80, "b": 80}  # Extra top margin for multi-line title
    }
    
    # Set y-axis range if provided
    if y_range is not None and len(y_range) == 2:
        layout_update["yaxis"]["range"] = y_range
    
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

# Example usage:
# Import libraries
from pathlib import Path
import os
import sys


# Import our electrochemistry module

# Define data and output paths - using absolute paths to avoid errors
base_dir = parent_dir  # The Experiments directory
data_dir = base_dir / "Raw_Data" / "Eletrochemestry"
output_base_dir = base_dir / "Processed_Data" / "Eletrochemestry"

# Print paths for debugging
print(f"Current working directory: {os.getcwd()}")
print(f"Looking for data in: {data_dir}")
print(f"Will save results to: {output_base_dir}")

# Check if file exists before loading
file_path = data_dir / "P148_CC-C20_CV-C40_Channel_8_Wb_1.xlsx"
if not file_path.exists():
    print(f"ERROR: File not found: {file_path}")
    print(f"Available files in {data_dir}:")
    for f in data_dir.glob("*"):
        print(f"  - {f.name}")
else:
    print(f"Found file: {file_path}")
    
    # Set cell metadata for plotting
    cell_metadata = {
        'cell_id': 'P148',
        'cathode_type': 'LFO-C',
        'anode_type': 'Li metal',
        'cathode_composition': '75:20:5 (AM:C:Binder)',
        'electrolyte': '1M LiPF6 in EC:EMC (1:1) + 5% VC',
        'c_rate': '0.05C'
    }
    
    # Simple data loading (column names will be auto-detected)
    df = load_data(file_path)
    
    # Data loading with capacity normalization (converts Ah to mAh/g)
    df_normalized = load_data(
        file_path,
        total_mass=20.3,      # Total electrode mass in mg
        am_percentage=75,     # Active material percentage (75%)
        collector_mass=0      # Current collector mass in mg
    )
    
    # Plot individual graphs with metadata and custom y-axis ranges
    # Note: When using normalized data, capacity will be in mAh/g
    cd_fig = plot_charge_discharge(
        df_normalized, 
        cycles_to_plot=[1, 2, 3, 5, 10, 25, 50], 
        cell_metadata=cell_metadata,
        y_range=[1.5, 3.5]  # Set voltage range to match expected range (1.5-3.5V)
    )
    
    # Discharge capacity vs cycle number (scatter plot without lines)
    cap_fig = plot_capacity_vs_cycle(
        df_normalized,
        cell_metadata=cell_metadata,
        y_range=[200, 350]  # Set capacity range from 0 to 350 mAh/g
    )
    
    soh_fig = plot_state_of_health(
        df_normalized,
        cell_metadata=cell_metadata,
        y_range=[50, 105]  # Display SOH between 50% and 105%
    )
    
    # Coulombic efficiency vs cycle number (scatter plot without lines)
    ce_fig = plot_coulombic_efficiency(
        df_normalized,
        cell_metadata=cell_metadata,
        y_range=[80, 101]  # Display CE between 80% and 101%
    )
    
    dqdv_fig = plot_differential_capacity(
        df_normalized, 
        cycles_to_plot=[1, 10],
        cell_metadata=cell_metadata
        # No y-range set - will auto-scale
    )
    
    vt_fig = plot_voltage_time(
        df_normalized,
        cell_metadata=cell_metadata,
        y_range=[1.5, 3.5]  # Set voltage range to match expected range (1.5-3.5V)
    )
    
    # Example of using the alternative dQ/dV calculation method
    dqdv2_fig = plot_differential_capacity_2(
        df_normalized, 
        cycles_to_plot=[1, 10],
        cell_metadata=cell_metadata
        # No y-range set - will auto-scale
    )
    
    # Save plots to output directory (both HTML and high-resolution PNG)
    output_dir = output_base_dir / "P148"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save plots with reasonable size but high DPI for quality
    print("Generating plots...")
    save_plot(
        cd_fig, 
        output_dir / "charge_discharge", 
        save_png=True,
        dpi=300  # High DPI for quality, but standard size
    )
    save_plot(
        cap_fig, 
        output_dir / "capacity", 
        save_png=True,
        dpi=300
    )
    save_plot(
        soh_fig, 
        output_dir / "soh", 
        save_png=True,
        dpi=300
    )
    save_plot(
        ce_fig, 
        output_dir / "coulombic_efficiency", 
        save_png=True,
        dpi=300
    )
    save_plot(
        dqdv_fig, 
        output_dir / "dqdv", 
        save_png=True,
        dpi=300
    )
    save_plot(
        vt_fig, 
        output_dir / "voltage_time", 
        save_png=True,
        dpi=300
    )
    save_plot(
        dqdv2_fig, 
        output_dir / "dqdv_current_time", 
        save_png=True,
        dpi=300
    )
    
    print(f"All plots saved to {output_dir}")

# Example of processing multiple cells
# Uncomment and modify the cell IDs and file patterns as needed
"""
# Sample metadata for different cells
cell_metadata_templates = {
    "P148": {
        'cell_id': 'P148',
        'cathode_type': 'LFO-C',
        'anode_type': 'Li metal',
        'cathode_composition': '75:20:5 (AM:C:Binder)',
        'electrolyte': '1M LiPF6 in EC:EMC (1:1) + 5% VC',
        'c_rate': '0.05C'
    },
    "P162": {
        'cell_id': 'P162',
        'cathode_type': 'LFO-C',
        'anode_type': 'Graphite',
        'cathode_composition': '75:20:5 (AM:C:Binder)',
        'electrolyte': '1M LiPF6 in EC:EMC (1:1) + 5% VC',
        'c_rate': '0.1C'
    }
}

cell_ids = ["P148", "P162", "P166", "P167"]
for cell_id in cell_ids:
    # Find files matching the pattern for this cell
    pattern = f"{cell_id}*.xlsx"
    matching_files = list(data_dir.glob(pattern))
    
    if matching_files:
        file_path = matching_files[0]  # Use the first matching file
        print(f"Processing {cell_id} with file: {file_path.name}")
        
        # Get metadata for this cell (or use default if not available)
        if cell_id in cell_metadata_templates:
            cell_metadata = cell_metadata_templates[cell_id]
        else:
            cell_metadata = {
                'cell_id': cell_id,
                'cathode_type': 'LFO-C',
                'anode_type': 'Li',
                'cathode_composition': '75:20:5',
                'electrolyte': '1M LiPF6 EC:EMC',
                'c_rate': 'N/A'
            }
        
        # Load and process data
        df = load_data(
            file_path,
            total_mass=20.3,
            am_percentage=75,
            collector_mass=0
        )
        
        # Create output directory for this cell
        cell_dir = output_base_dir / cell_id
        cell_dir.mkdir(parents=True, exist_ok=True)
        
        # Create and save plots with metadata
        save_plot(
            plot_charge_discharge(
                df, 
                cycles_to_plot=[1, 5, 10], 
                cell_metadata=cell_metadata,
                y_range=[1.5, 3.5]  # Standard voltage range
            ), 
            cell_dir / "charge_discharge",
            save_png=True,
            dpi=300
        )
        save_plot(
            plot_capacity_vs_cycle(
                df, 
                cell_metadata=cell_metadata,
                y_range=[0, 350]  # Capacity range
            ),
            cell_dir / "capacity",
            save_png=True,
            dpi=300
        )
        
        # Add other plots
        save_plot(
            plot_state_of_health(
                df, 
                cell_metadata=cell_metadata,
                y_range=[50, 105]  # SOH range
            ),
            cell_dir / "soh",
            save_png=True,
            dpi=300
        )
        save_plot(
            plot_coulombic_efficiency(
                df, 
                cell_metadata=cell_metadata,
                y_range=[80, 101]  # CE range
            ),
            cell_dir / "coulombic_efficiency",
            save_png=True,
            dpi=300
        )
        
        print(f"Finished processing {cell_id}")
    else:
        print(f"No files found for {cell_id}")
""" 
